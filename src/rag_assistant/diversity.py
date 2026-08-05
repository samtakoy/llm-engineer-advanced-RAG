"""Отбор разнообразной выдачи поверх уже оценённых фрагментов.

MMR отбирает из оценённых кандидатов, вычитая из оценки похожесть кандидата
на уже отобранное.

Отличие от режима MMR у векторного хранилища (флаг `--mmr`): там разнообразие считается
до реранкера, релевантностью служит косинус вопроса и фрагмента, а выстроенный порядок
реранкер потом перестраивает по-своему. Здесь (флаг `--mmr2`) релевантность — оценка
cross-encoder, и отбор идёт последним шагом пайплайна.

Дополнительно к score нужны векторы узлов - они берутся из готовой коллекции ChromaDB, а не считаются заново.

Обе половины формулы растягиваются на [0, 1] по крайним значениям пачки кандидатов:
без этого вес отбора ни на что не влияет, см. stretch_to_unit_range.
"""
import math
from typing import Callable, Dict, Hashable, List, Sequence, Set, Tuple, TypeVar

import chromadb
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

from rag_assistant.config import AppConfig
from rag_assistant.index import find_collection

# Вектора узла в коллекции не нашлось: похожесть считать не на чем, и узел проходит
# отбор как непохожий ни на что.
UNKNOWN_SIMILARITY = 0.0

KeyType = TypeVar("KeyType", bound = Hashable)


class DiversityWeightMissing(RuntimeError):
    """Отбор по разнообразию включён, а вес разнообразия не задан."""


def resolve_mmr_threshold(config: AppConfig) -> float:
    """Возвращает вес MMR, требуя, чтобы он был задан.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Долю веса, отданную релевантности.

    Исключения:
        DiversityWeightMissing: настройка MMR_THRESHOLD пуста.
    """
    if config.mmr_threshold is None:
        raise DiversityWeightMissing(
            "Отбору по разнообразию нужен вес: задайте MMR_THRESHOLD в .env "
            "(1.0 — только релевантность, 0.0 — только непохожесть фрагментов).",
        )

    return config.mmr_threshold


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Считает косинус угла между векторами.

    Аргументы:
        first: первый вектор.
        second: второй вектор.

    Возвращает:
        Косинус в пределах [-1, 1], либо 0.0, если хотя бы один вектор нулевой.
    """
    dot_product = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def stretch_to_unit_range(values: Dict[KeyType, float]) -> Dict[KeyType, float]:
    """Растягивает величины на отрезок [0, 1] по крайним значениям пачки.

    Обе половины формулы MMR формально лежат в [0, 1], но занимают в нём разные
    участки: оценки cross-encoder на одном вопросе укладываются в 0.02…0.12, а
    косинусы между кандидатами держатся около 0.7 — соседние фрагменты одного отчёта
    похожи все. Без растяжения штраф крупнее вклада релевантности на порядок,
    и вес отбора ни на что не влияет.

    Аргументы:
        values: величины по ключам.

    Возвращает:
        Те же ключи с величинами в [0, 1]. Если все величины равны, различить
        их нечем и все обращаются в ноль.
    """
    if not values:
        return {}

    lowest = min(values.values())
    highest = max(values.values())

    if highest == lowest:
        return {key: 0.0 for key in values}

    return {
        key: (value - lowest) / (highest - lowest)
        for key, value in values.items()
    }


def measure_similarities(
    nodes: List[NodeWithScore],
    embeddings: Dict[str, List[float]],
) -> Dict[Tuple[str, str], float]:
    """Считает похожесть каждой пары кандидатов, растянутую на [0, 1].

    Пары считаются один раз до отбора: отбор перебирает их многократно, а состав
    кандидатов при этом не меняется.

    Аргументы:
        nodes: оценённые кандидаты.
        embeddings: векторы узлов по идентификаторам.

    Возвращает:
        Похожесть по паре идентификаторов, упорядоченной по возрастанию. Пары
        с узлом, вектора которого нет, в выдачу не попадают.
    """
    known = [node for node in nodes if node.node.node_id in embeddings]
    pairs = {
        pair_key(first.node.node_id, second.node.node_id): cosine_similarity(
            embeddings[first.node.node_id],
            embeddings[second.node.node_id],
        )
        for position, first in enumerate(known)
        for second in known[position + 1:]
    }

    return stretch_to_unit_range(pairs)


def pair_key(first_id: str, second_id: str) -> Tuple[str, str]:
    """Собирает ключ пары узлов, не зависящий от порядка.

    Аргументы:
        first_id: идентификатор одного узла.
        second_id: идентификатор другого узла.

    Возвращает:
        Пару идентификаторов, упорядоченную по возрастанию.
    """
    return (first_id, second_id) if first_id <= second_id else (second_id, first_id)


def similarity_to_selected(
    candidate: NodeWithScore,
    selected: List[NodeWithScore],
    similarities: Dict[Tuple[str, str], float],
) -> float:
    """Находит, насколько кандидат похож на ближайший из уже отобранных узлов.

    Аргументы:
        candidate: узел, который рассматривается к отбору.
        selected: узлы, уже попавшие в выдачу.
        similarities: похожесть по паре идентификаторов.

    Возвращает:
        Наибольшую похожесть с отобранными узлами, либо 0.0, если сравнивать
        не с чем.
    """
    return max(
        (
            similarities[pair_key(candidate.node.node_id, node.node.node_id)]
            for node in selected
            if pair_key(candidate.node.node_id, node.node.node_id) in similarities
        ),
        default = UNKNOWN_SIMILARITY,
    )


def mmr_score(
    candidate: NodeWithScore,
    selected: List[NodeWithScore],
    relevances: Dict[str, float],
    similarities: Dict[Tuple[str, str], float],
    mmr_threshold: float,
) -> float:
    """Считает оценку кандидата с учётом уже отобранного.

    Аргументы:
        candidate: узел, который рассматривается к отбору.
        selected: узлы, уже попавшие в выдачу.
        relevances: релевантность узлов по идентификаторам, растянутая на [0, 1].
        similarities: похожесть по паре идентификаторов, растянутая на [0, 1].
        mmr_threshold: доля веса, отданная релевантности, остальное — непохожести.

    Возвращает:
        Оценку: чем выше, тем раньше узел попадёт в выдачу.
    """
    relevance = relevances.get(candidate.node.node_id, 0.0)
    similarity = similarity_to_selected(
        candidate = candidate,
        selected = selected,
        similarities = similarities,
    )

    return mmr_threshold * relevance - (1.0 - mmr_threshold) * similarity


def select_by_mmr(
    nodes: List[NodeWithScore],
    embeddings: Dict[str, List[float]],
    top_k: int,
    mmr_threshold: float,
) -> List[NodeWithScore]:
    """Отбирает узлы жадно: на каждом шаге лучший с поправкой на уже отобранное.

    Первым уходит узел с наибольшей оценкой — до отбора похожесть не с чем считать,
    и поправка равна нулю. Дальше каждый следующий узел выбирается из оставшихся.

    Аргументы:
        nodes: оценённые кандидаты.
        embeddings: векторы узлов по идентификаторам.
        top_k: сколько узлов оставить.
        mmr_threshold: доля веса, отданная релевантности, остальное — непохожести.

    Возвращает:
        Отобранные узлы в порядке отбора.
    """
    relevances = stretch_to_unit_range(
        {node.node.node_id: node.score or 0.0 for node in nodes},
    )
    similarities = measure_similarities(nodes = nodes, embeddings = embeddings)

    selected: List[NodeWithScore] = []
    selected_ids: Set[str] = set()

    while len(selected) < top_k:
        remaining = [node for node in nodes if node.node.node_id not in selected_ids]

        if not remaining:
            break

        best = max(
            remaining,
            key = lambda candidate: mmr_score(
                candidate = candidate,
                selected = selected,
                relevances = relevances,
                similarities = similarities,
                mmr_threshold = mmr_threshold,
            ),
        )
        selected.append(best)
        selected_ids.add(best.node.node_id)

    return selected


class DiversityPostprocessor(BaseNodePostprocessor):
    """Оставляет top_k фрагментов, отбирая их по MMR из оценённых кандидатов.

    Оценки узлов не меняются: постпроцессор отбирает и переставляет, а ссылки
    под ответом продолжают показывать релевантность, посчитанную реранкером.

    Атрибуты:
        top_k: сколько фрагментов уходит модели.
        mmr_threshold: доля веса, отданная релевантности, остальное — непохожести.
        embedding_lookup: выдаёт векторы узлов по их идентификаторам.
    """

    top_k: int = Field(description = "Сколько фрагментов остаётся после отбора.")
    mmr_threshold: float = Field(description = "Доля веса, отданная релевантности.")
    embedding_lookup: Callable[[List[str]], Dict[str, List[float]]] = Field(
        description = "Векторы узлов по идентификаторам.",
    )

    @classmethod
    def class_name(cls) -> str:
        """Имя класса для сериализации llama-index.

        Возвращает:
            Устойчивое имя, по которому постпроцессор восстанавливается из настроек.
        """
        return "DiversityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:
        """Отбирает разнообразную выдачу из принесённых кандидатов.

        Аргументы:
            nodes: кандидаты с оценками релевантности.
            query_bundle: вопрос. Не используется: релевантность уже посчитана,
                а разнообразие меряется между узлами.

        Возвращает:
            Не более top_k узлов, отобранных по MMR.
        """
        if len(nodes) <= self.top_k:
            return list(nodes)

        return select_by_mmr(
            nodes = nodes,
            embeddings = self.embedding_lookup([node.node.node_id for node in nodes]),
            top_k = self.top_k,
            mmr_threshold = self.mmr_threshold,
        )


def create_embedding_lookup(config: AppConfig) -> Callable[[List[str]], Dict[str, List[float]]]:
    """Создаёт чтение векторов узлов из готовой коллекции ChromaDB.

    Векторы уже посчитаны при сборке индекса, поэтому берутся с диска: повторный
    прогон эмбеддера дал бы то же самое, но на каждый запрос.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Функцию, отдающую векторы по идентификаторам узлов. Узел, которого
        в коллекции не оказалось, в выдачу не попадает.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))
    collection = find_collection(client = client, name = config.chroma_collection)

    def lookup(node_ids: List[str]) -> Dict[str, List[float]]:
        if collection is None:
            return {}

        records = collection.get(ids = node_ids, include = ["embeddings"])
        embeddings = records.get("embeddings")

        if embeddings is None or len(embeddings) == 0:
            return {}

        return {
            node_id: [float(value) for value in embedding]
            for node_id, embedding in zip(records["ids"], embeddings)
        }

    return lookup


def create_diversity_postprocessor(
    config: AppConfig,
    mmr_threshold: float,
) -> DiversityPostprocessor:
    """Собирает отбор по разнообразию поверх готового индекса.

    Аргументы:
        config: конфигурация приложения.
        mmr_threshold: доля веса, отданная релевантности, остальное — непохожести.

    Возвращает:
        Постпроцессор, оставляющий TOP_K фрагментов.
    """
    return DiversityPostprocessor(
        top_k = config.top_k,
        mmr_threshold = mmr_threshold,
        embedding_lookup = create_embedding_lookup(config),
    )

"""Поиск по словам: индекс BM25 строится один раз, фильтр применяется на запрос.

Векторный поиск отвечает на смысл, а буквальное совпадение теряет: редкое имя
собственное, код, число размазаны по вектору вместе со всей темой фрагмента. BM25
ищет ровно по словам и потому дополняет его.

Индекс живёт в оперативной памяти рядом с приложением, в отличие от векторов, которые
лежат в Chroma. Разделение на «построить» и «искать» здесь не деталь реализации:
фильтр по метаданным приходит вместе с вопросом, а перестраивать индекс на каждый
вопрос — работа, растущая с размером корпуса.
"""
from typing import List

import Stemmer
import bm25s
import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.utils import build_metadata_filter_fn

# Язык корпуса. Стеммер сводит словоформы («госпитализаций» и «госпитализация» —
# одно слово), список стоп-слов убирает предлоги и союзы, которые есть в любом
# фрагменте и потому ничего не различают.
LANGUAGE = "russian"


class LexicalIndex:
    """Индекс BM25 по узлам корпуса."""

    def __init__(self, nodes: List[BaseNode]) -> None:
        """Строит индекс по словам.

        Аргументы:
            nodes: узлы, по которым идёт поиск, — весь корпус целиком.
        """
        self.nodes = nodes
        self.stemmer = Stemmer.Stemmer(LANGUAGE)
        # Метаданные в текст не берутся: имя файла стоит в каждом узле корпуса,
        # и поиск отвечал бы на слова из названия документа.
        corpus_tokens = bm25s.tokenize(
            [node.get_content(metadata_mode = MetadataMode.NONE) for node in nodes],
            stopwords = LANGUAGE,
            stemmer = self.stemmer,
            show_progress = False,
        )
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress = False)

    def search(self, question: str, filters: MetadataFilters | None, top_k: int) -> List[NodeWithScore]:
        """Ищет узлы, в которых встречаются слова вопроса.

        Аргументы:
            question: вопрос пользователя.
            filters: отбор по метаданным либо None — искать по всему корпусу.
            top_k: сколько узлов вернуть.

        Возвращает:
            Узлы с ненулевой оценкой, от большей к меньшей. Узлы без совпадений
            и узлы, не прошедшие отбор, в выдачу не попадают.
        """
        query_tokens = bm25s.tokenize(
            question,
            stopwords = LANGUAGE,
            stemmer = self.stemmer,
            show_progress = False,
        )
        indexes, scores = self.bm25.retrieve(
            query_tokens,
            k = min(top_k, len(self.nodes)),
            weight_mask = self.build_weight_mask(filters),
            show_progress = False,
        )

        # Отбор обнуляет оценку, но узел из индекса не убирает: когда подходящих
        # меньше запрошенного числа, места добираются нулями. Их и отсеиваем —
        # заодно уходят узлы, где ни одно слово вопроса не встретилось.
        return [
            NodeWithScore(node = self.nodes[int(index)], score = float(score))
            for index, score in zip(indexes[0], scores[0])
            if score > 0
        ]

    def build_weight_mask(self, filters: MetadataFilters | None) -> np.ndarray | None:
        """Собирает маску отбора по метаданным.

        Аргументы:
            filters: отбор по метаданным либо None.

        Возвращает:
            Массив весов по узлам корпуса: единица у подходящих, ноль у остальных.
            None, если отбора нет.
        """
        if filters is None:
            return None

        nodes_by_id = {node.node_id: node for node in self.nodes}
        matches = build_metadata_filter_fn(lambda node_id: nodes_by_id[node_id].metadata, filters)

        return np.array([int(matches(node.node_id)) for node in self.nodes])


def create_lexical_index(index: VectorStoreIndex) -> LexicalIndex:
    """Строит индекс по словам поверх узлов векторного индекса.

    Тексты берутся из докстора: векторное хранилище отдаёт вектора, а не слова.

    Аргументы:
        index: векторный индекс документов вместе с докстором.

    Возвращает:
        Готовый к поиску индекс по словам.
    """
    return LexicalIndex(nodes = list(index.docstore.docs.values()))


class LexicalRetriever(BaseRetriever):
    """Отдаёт узлы индекса BM25 по вопросу."""

    def __init__(
        self,
        lexical_index: LexicalIndex,
        filters: MetadataFilters | None,
        top_k: int,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        """Связывает готовый индекс с параметрами одного поиска.

        Аргументы:
            lexical_index: построенный индекс по словам.
            filters: отбор по метаданным либо None — искать по всему корпусу.
            top_k: сколько узлов забирает поиск.
            callback_manager: сборщик событий поиска, нужен базовому классу.
        """
        self.lexical_index = lexical_index
        self.filters = filters
        self.top_k = top_k
        super().__init__(callback_manager = callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Ищет по словам вопроса.

        Аргументы:
            query_bundle: вопрос в обёртке фреймворка.

        Возвращает:
            Узлы с ненулевой оценкой.
        """
        return self.lexical_index.search(
            question = query_bundle.query_str,
            filters = self.filters,
            top_k = self.top_k,
        )

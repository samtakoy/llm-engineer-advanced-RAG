"""Движок вопрос-ответ: поиск по индексу плюс ответ модели с ссылками."""
from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from rag_assistant.config import AppConfig
from rag_assistant.lexical import LexicalIndex, LexicalRetriever
from rag_assistant.prompts import QA_TEMPLATE


@dataclass(frozen = True)
class Source:
    """Фрагмент документа, попавший в контекст ответа.

    Атрибуты:
        citation: ссылка для показа пользователю, например "отчёт.pdf, стр. 12".
        text: текст фрагмента.
        score: близость фрагмента к вопросу.
        file_name: имя исходного файла.
        page_label: номер страницы либо None, если источник без страниц.
        page_end: последняя страница фрагмента, если таблица сшита через границу
            листа. None — фрагмент лежит на одной странице.
    """

    citation: str
    text: str
    score: float
    file_name: str
    page_label: str | None
    page_end: str | None


@dataclass(frozen = True)
class Answer:
    """Ответ ассистента вместе с источниками.

    Атрибуты:
        text: текст ответа.
        sources: фрагменты, на которых ответ построен.
    """

    text: str
    sources: List[Source]


def retrieval_top_k(
    config: AppConfig,
    reranker: SentenceTransformerRerank | None,
    diversity: bool,
) -> int:
    """Определяет, сколько фрагментов должен принести поиск.

    Аргументы:
        config: конфигурация приложения.
        reranker: реранкер либо None, если порядок выдачи остаётся за поиском.
        diversity: True — выдачу отбирает разнообразие.

    Возвращает:
        Число кандидатов с запасом, когда выдачу будет переставлять реранкер либо
        отбирать разнообразие, иначе ровно столько, сколько уйдёт модели.
    """
    if reranker is None and not diversity:
        return config.top_k

    return config.candidate_top_k


def rerank_top_n(config: AppConfig, diversity: bool) -> int:
    """Определяет, сколько фрагментов оставляет реранкер.

    Аргументы:
        config: конфигурация приложения.
        diversity: True — после реранкера идёт отбор по разнообразию.

    Возвращает:
        Всех кандидатов, когда выдачу будет отбирать разнообразие: урезав список
        сразу до top_k, реранкер не оставил бы отбору выбора. Иначе ровно столько,
        сколько уйдёт модели.
    """
    return config.candidate_top_k if diversity else config.top_k


def create_vector_retriever(
    index: VectorStoreIndex,
    filters: MetadataFilters | None,
    top_k: int,
    mmr_threshold: float | None,
) -> BaseRetriever:
    """Создаёт поиск по близости векторов.

    Аргументы:
        index: векторный индекс документов.
        filters: отбор документов по метаданным либо None — искать по всему корпусу.
        top_k: сколько фрагментов забирает поиск.
        mmr_threshold: доля веса, отданная близости к вопросу, остальное уходит
            непохожести фрагментов друг на друга. None — отбирать только по близости.

    Возвращает:
        Поиск по векторам.
    """
    if mmr_threshold is None:
        return index.as_retriever(similarity_top_k = top_k, filters = filters)

    # Разнообразие считается по векторам, поэтому режим задаётся самому хранилищу:
    # оно достаёт кандидатов с запасом и отбирает из них непохожие друг на друга.
    return index.as_retriever(
        similarity_top_k = top_k,
        filters = filters,
        vector_store_query_mode = VectorStoreQueryMode.MMR,
        vector_store_kwargs = {"mmr_threshold": mmr_threshold},
    )


def create_retriever(
    index: VectorStoreIndex,
    filters: MetadataFilters | None,
    top_k: int,
    lexical_index: LexicalIndex | None,
    mmr_threshold: float | None,
) -> BaseRetriever:
    """Создаёт поиск по корпусу: одну ветку либо объединение веток.

    Аргументы:
        index: векторный индекс документов вместе с докстором.
        filters: отбор документов по метаданным либо None — искать по всему корпусу.
            Отбор идёт до поиска, поэтому выдача набирается уже внутри
            отобранных документов.
        top_k: сколько фрагментов забирает поиск.
        lexical_index: индекс поиска по словам либо None — искать только векторно.
        mmr_threshold: доля веса, отданная близости к вопросу, остальное уходит
            непохожести фрагментов друг на друга. None — отбирать только по близости.

    Возвращает:
        Поиск, готовый отдавать фрагменты по вопросу.
    """
    branches: List[BaseRetriever] = [
        create_vector_retriever(
            index = index,
            filters = filters,
            top_k = top_k,
            mmr_threshold = mmr_threshold,
        ),
    ]

    if lexical_index is not None:
        branches.append(
            LexicalRetriever(
                lexical_index = lexical_index,
                filters = filters,
                top_k = top_k,
            )
        )

    if len(branches) == 1:
        return branches[0]

    return QueryFusionRetriever(
        retrievers = branches,
        # Ранги выдач складываются по взаимному рангу: фрагмент, стоящий высоко
        # в нескольких ветках, поднимается выше лидера любой одной из них.
        mode = FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k = top_k,
        # Один запрос означает «искать ровно тем вопросом, что задан». Иначе
        # объединение сперва просит модель придумать похожие формулировки.
        num_queries = 1,
        use_async = False,
    )


class RagEngine:
    """Отвечает на вопросы по проиндексированным документам."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: SentenceTransformerRerank | None,
        diversity: BaseNodePostprocessor | None,
    ) -> None:
        """Создаёт движок запросов поверх готового поиска.
        Поиск собирается снаружи.

        Аргументы:
            retriever: поиск, отдающий фрагменты по вопросу.
            reranker: реранкер либо None, если порядок выдачи остаётся за поиском.
            diversity: отбор по разнообразию либо None, если выдача идёт как есть.
                Стоит после реранкера: отбирает из того, что тот уже оценил.
        """
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever = retriever,
            text_qa_template = QA_TEMPLATE,
            node_postprocessors = [
                postprocessor
                for postprocessor in (reranker, diversity)
                if postprocessor is not None
            ],
        )

    def ask(self, question: str) -> Answer:
        """Отвечает на вопрос по документам.

        Аргументы:
            question: вопрос пользователя.

        Возвращает:
            Ответ модели и фрагменты, на которых он построен.
        """
        response = self.query_engine.query(question)

        return Answer(
            text = str(response),
            sources = [to_source(node) for node in response.source_nodes],
        )


def to_source(node: NodeWithScore) -> Source:
    """Собирает ссылку на источник из метаданных найденного узла.

    Аргументы:
        node: узел, который вернул поиск.

    Возвращает:
        Источник с готовой ссылкой на файл и страницу.
    """
    file_name = node.metadata.get("file_name", "документ")
    page_label = node.metadata.get("page_label")
    # Узел со сшитой таблицей лежит на двух листах, и ссылка называет оба: иначе
    # число со второго листа подписано страницей, на которой его нет.
    page_end = node.metadata.get("page_end")
    pages = f"{page_label}-{page_end}" if page_end and page_end != page_label else page_label
    citation = f"{file_name}, стр. {pages}" if page_label else file_name

    return Source(
        citation = citation,
        text = node.text,
        score = node.score or 0.0,
        file_name = file_name,
        page_label = page_label,
        page_end = page_end,
    )

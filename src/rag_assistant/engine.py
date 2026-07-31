"""Движок вопрос-ответ: поиск по индексу плюс ответ модели с ссылками."""
from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters

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
    """

    citation: str
    text: str
    score: float
    file_name: str
    page_label: str | None


@dataclass(frozen = True)
class Answer:
    """Ответ ассистента вместе с источниками.

    Атрибуты:
        text: текст ответа.
        sources: фрагменты, на которых ответ построен.
    """

    text: str
    sources: List[Source]


def retrieval_top_k(config: AppConfig, reranker: SentenceTransformerRerank | None) -> int:
    """Определяет, сколько фрагментов должен принести поиск.

    Аргументы:
        config: конфигурация приложения.
        reranker: реранкер либо None, если порядок выдачи остаётся за поиском.

    Возвращает:
        Число кандидатов с запасом, когда фрагменты будут переставлены реранкером,
        иначе ровно столько, сколько уйдёт модели.
    """
    return config.candidate_top_k if reranker is not None else config.top_k


def create_retriever(
    index: VectorStoreIndex,
    config: AppConfig,
    filters: MetadataFilters | None,
    top_k: int,
    lexical_index: LexicalIndex | None,
) -> AutoMergingRetriever:
    """Создаёт поиск, склеивающий найденные мелкие фрагменты в крупные.

    Ищет по листьям, потому что векторизованы только они. Если в выдачу попало
    больше половины листьев одного родителя, они заменяются родителем целиком:
    факт, разорванный границей чанка, так возвращается собранным.

    Аргументы:
        index: векторный индекс документов вместе с докстором.
        config: конфигурация приложения.
        filters: отбор документов по метаданным либо None — искать по всему корпусу.
            Отбор идёт до поиска соседей, поэтому выдача набирается уже внутри
            отобранных документов.
        top_k: сколько фрагментов забирает поиск.
        lexical_index: индекс поиска по словам либо None — искать только векторно.

    Возвращает:
        Поиск, готовый отдавать фрагменты по вопросу.
    """
    vector_retriever = index.as_retriever(similarity_top_k = top_k, filters = filters)
    found: BaseRetriever = vector_retriever

    if lexical_index is not None:
        # Ветки объединяются до склейки листьев: иначе в общий список попали бы
        # и склеенный родитель от векторного поиска, и его же лист от поиска по
        # словам — один и тот же текст двумя строками, занимающими два места.
        found = QueryFusionRetriever(
            retrievers = [
                vector_retriever,
                LexicalRetriever(
                    lexical_index = lexical_index,
                    filters = filters,
                    top_k = top_k,
                ),
            ],
            # Ранги двух выдач складываются по взаимному рангу: фрагмент, стоящий
            # высоко в обеих, поднимается выше лидера одной из них.
            mode = FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k = top_k,
            # Один запрос означает «искать ровно тем вопросом, что задан». Иначе
            # объединение сперва просит модель придумать похожие формулировки.
            num_queries = 1,
            use_async = False,
        )

    return AutoMergingRetriever(
        vector_retriever = found,
        storage_context = index.storage_context,
        verbose = False,
    )


class RagEngine:
    """Отвечает на вопросы по проиндексированным документам."""

    def __init__(
        self,
        index: VectorStoreIndex,
        config: AppConfig,
        filters: MetadataFilters | None,
        reranker: SentenceTransformerRerank | None,
        lexical_index: LexicalIndex | None,
    ) -> None:
        """Создаёт движок запросов поверх индекса.

        Аргументы:
            index: векторный индекс документов.
            config: конфигурация приложения.
            filters: отбор документов по метаданным либо None — искать по всему корпусу.
            reranker: реранкер либо None, если порядок выдачи остаётся за поиском.
            lexical_index: индекс поиска по словам либо None — искать только векторно.
        """
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever = create_retriever(
                index = index,
                config = config,
                filters = filters,
                top_k = retrieval_top_k(config = config, reranker = reranker),
                lexical_index = lexical_index,
            ),
            text_qa_template = QA_TEMPLATE,
            node_postprocessors = [reranker] if reranker is not None else [],
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
    citation = f"{file_name}, стр. {page_label}" if page_label else file_name

    return Source(
        citation = citation,
        text = node.text,
        score = node.score or 0.0,
        file_name = file_name,
        page_label = page_label,
    )

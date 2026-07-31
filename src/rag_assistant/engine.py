"""Движок вопрос-ответ: поиск по индексу плюс ответ модели с ссылками."""
from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters

from rag_assistant.config import AppConfig
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


def create_retriever(
    index: VectorStoreIndex,
    config: AppConfig,
    filters: MetadataFilters | None,
) -> AutoMergingRetriever:
    """Создаёт поиск, склеивающий найденные мелкие фрагменты в крупные.

    Ищет по листьям, потому что векторизованы только они. Если в выдачу попало
    больше половины листьев одного родителя, они заменяются родителем целиком:
    факт, разорванный границей чанка, так возвращается собранным.

    Аргументы:
        index: векторный индекс документов вместе с докстором.
        config: конфигурация приложения.
        filters: отбор документов по метаданным либо None — искать по всему корпусу.
            Отбор идёт до поиска соседей, поэтому top_k набирается уже внутри
            отобранных документов.

    Возвращает:
        Поиск, готовый отдавать фрагменты по вопросу.
    """
    return AutoMergingRetriever(
        vector_retriever = index.as_retriever(similarity_top_k = config.top_k, filters = filters),
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
    ) -> None:
        """Создаёт движок запросов поверх индекса.

        Аргументы:
            index: векторный индекс документов.
            config: конфигурация приложения.
            filters: отбор документов по метаданным либо None — искать по всему корпусу.
        """
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever = create_retriever(index = index, config = config, filters = filters),
            text_qa_template = QA_TEMPLATE,
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

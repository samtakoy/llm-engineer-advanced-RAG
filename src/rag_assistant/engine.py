"""Движок вопрос-ответ: поиск по индексу плюс ответ модели с ссылками."""
from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from rag_assistant.config import AppConfig
from rag_assistant.prompts import QA_TEMPLATE


@dataclass(frozen = True)
class Source:
    """Фрагмент документа, попавший в контекст ответа.

    Атрибуты:
        citation: ссылка для показа пользователю, например "отчёт.pdf, стр. 12".
        text: текст фрагмента.
        score: близость фрагмента к вопросу.
    """

    citation: str
    text: str
    score: float


@dataclass(frozen = True)
class Answer:
    """Ответ ассистента вместе с источниками.

    Атрибуты:
        text: текст ответа.
        sources: фрагменты, на которых ответ построен.
    """

    text: str
    sources: List[Source]


class RagEngine:
    """Отвечает на вопросы по проиндексированным документам."""

    def __init__(self, index: VectorStoreIndex, config: AppConfig) -> None:
        """Создаёт движок запросов поверх индекса.

        Аргументы:
            index: векторный индекс документов.
            config: конфигурация приложения.
        """
        self.query_engine = index.as_query_engine(
            similarity_top_k = config.top_k,
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

    return Source(citation = citation, text = node.text, score = node.score or 0.0)

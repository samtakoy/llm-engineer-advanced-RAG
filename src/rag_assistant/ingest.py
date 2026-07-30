"""Загрузка документов из папки: один Document на страницу PDF."""
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

from rag_assistant.config import AppConfig
from rag_assistant.filters import drop_service_pages

# Метаданные для служебных нужд: в ссылке не нужны, в эмбеддинге мешают.
SERVICE_METADATA_KEYS = ["file_path", "file_size", "creation_date", "last_modified_date"]


def load_documents(config: AppConfig) -> List[Document]:
    """Читает папку с документами.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Список Document, по одному на страницу PDF, без служебных страниц.
        В метаданных file_name и page_label — из них собирается ссылка в ответе.
    """
    pages = SimpleDirectoryReader(
        input_dir = str(config.documents_dir),
        recursive = True,
    ).load_data()

    documents = drop_service_pages(pages)
    hide_service_metadata(documents)

    return documents


def hide_service_metadata(documents: List[Document]) -> None:
    """Прячет служебные метаданные от эмбеддера и от модели.

    Аргументы:
        documents: документы, меняются на месте.

    Возвращает:
        None.
    """
    for document in documents:
        document.excluded_embed_metadata_keys = SERVICE_METADATA_KEYS
        document.excluded_llm_metadata_keys = SERVICE_METADATA_KEYS

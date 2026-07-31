"""Загрузка документов из папки: один Document на страницу PDF."""
from pathlib import Path
from typing import List

import pymupdf4llm
from llama_index.core import Document

from rag_assistant.config import AppConfig
from rag_assistant.ingest.filters import drop_service_pages
from rag_assistant.ingest.normalize import normalize_page

# Путь к файлу нужен только при отладке: ни в ссылке, ни в ответе он не участвует.
# Номер страницы модель видеть должна — по нему она проставляет ссылку, — но в вектор
# он попадать не должен: у всех узлов это одинаковое по форме поле со случайным для
# смысла числом, то есть шум.
EXCLUDED_FROM_EMBEDDING = ("file_path", "page_label")
EXCLUDED_FROM_PROMPT = ("file_path",)


def load_documents(config: AppConfig) -> List[Document]:
    """Читает папку с документами.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Список Document, по одному на страницу PDF, без служебных страниц.
        В метаданных file_name и page_label — из них собирается ссылка в ответе.
    """
    pages = []
    for pdf_path in sorted(config.documents_dir.rglob("*.pdf")):
        pages.extend(read_pdf(pdf_path))

    documents = drop_service_pages(pages)
    hide_service_metadata(documents)

    return documents


def read_pdf(pdf_path: Path) -> List[Document]:
    """Разбирает один PDF в markdown, по одному Document на страницу.

    Идентификатор документа собирается из имени файла и номера страницы:
    по нему индекс узнаёт страницу при повторной загрузке.

    Аргументы:
        pdf_path: путь к файлу.

    Возвращает:
        Страницы файла в markdown с метаданными.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks = True, show_progress = False)

    return [
        Document(
            doc_id = f"{pdf_path.name}:{page['metadata']['page_number']}",
            text = normalize_page(page["text"]),
            metadata = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "page_label": str(page["metadata"]["page_number"]),
            },
        )
        for page in pages
    ]


def hide_service_metadata(documents: List[Document]) -> None:
    """Прячет служебные метаданные от эмбеддера и от модели.

    Аргументы:
        documents: документы, меняются на месте.

    Возвращает:
        None.
    """
    for document in documents:
        document.excluded_embed_metadata_keys = list(EXCLUDED_FROM_EMBEDDING)
        document.excluded_llm_metadata_keys = list(EXCLUDED_FROM_PROMPT)

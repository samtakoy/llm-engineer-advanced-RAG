"""Загрузка документов из папки: страницы PDF и собранные из них файлы.

Страница — единица чтения и адресации: по её номеру строится ссылка в ответе.
Файл — единица нарезки: стек заголовков, порядок
чтения и таблицы не должны рваться на границе листа.

Номер страницы при нарезке файла не теряется: узел получает его по своему смещению
в тексте файла, этим занят `page_attribution`.
"""
from pathlib import Path
from typing import List

import pymupdf4llm
from llama_index.core import Document
from pymupdf4llm.ocr import OCRMode

from rag_assistant.config import AppConfig
from rag_assistant.ingest.filters import drop_service_pages
from rag_assistant.ingest.normalize import normalize_page
from rag_assistant.ingest.page_map import PAGE_JOIN
from rag_assistant.ingest.table_continuation import drop_continuation_separators

# Разметка документов для фильтрации по метаданным. Ведётся руками: имена файлов
# у отчётов разных лет устроены по-разному, а ошибка разметки молча сузила бы поиск.
# Файла нет в словаре — документ индексируется без полей фильтрации.
DOCUMENT_METADATA = {
    "FY2024_Issuer's report.pdf": {"year": "2024"},
    "Отчет эмитента МКПАО ЮМГ 12м2025.pdf": {"year": "2025"},
}

# Путь к файлу нужен только при отладке: ни в ссылке, ни в ответе он не участвует.
# Номер страницы модель видеть должна — по нему она проставляет ссылку, — но в вектор
# он попадать не должен: у всех узлов это одинаковое по форме поле со случайным для
# смысла числом, то есть шум.
# Поля фильтрации в вектор не идут: год отчёта в тексте чанка сбивал бы поиск
# по годам, которые отчёт тоже описывает — сравнительные колонки за прошлый период.
EXCLUDED_FROM_EMBEDDING = ("file_path", "page_label", "page_end", "year")
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

    documents = drop_continuation_separators(drop_service_pages(pages))
    hide_service_metadata(documents)

    return documents


def merge_pages_into_files(pages: List[Document]) -> List[Document]:
    """Собирает страницы в документы по файлу — то, что уходит в нарезку.

    Номера страниц в метаданных не остаётся: у файла её нет. Узлам её проставляет
    `page_attribution` по смещению в тексте.

    Аргументы:
        pages: страницы корпуса в любом порядке.

    Возвращает:
        По одному Document на файл, страницы внутри — в порядке чтения.
    """
    by_file = {}
    for page in pages:
        by_file.setdefault(page.metadata["file_name"], []).append(page)

    documents = []

    for file_name, file_pages in sorted(by_file.items()):
        file_pages.sort(key = lambda page: int(page.metadata["page_label"]))
        first = file_pages[0]
        documents.append(
            Document(
                doc_id = file_name,
                text = PAGE_JOIN.join(page.text for page in file_pages),
                metadata = {
                    key: value
                    for key, value in first.metadata.items()
                    if key not in ("page_label", "page_end")
                },
            )
        )

    hide_service_metadata(documents)

    return documents


def read_pdf(pdf_path: Path) -> List[Document]:
    """Разбирает один PDF в markdown, по одному Document на страницу.

    Идентификатор документа собирается из имени файла и номера страницы:
    по нему индекс узнаёт страницу при повторной загрузке.

    Аргументы:
        pdf_path: путь к файлу.

    Возвращает:
        Страницы файла в markdown с метаданными: ссылка на источник плюс поля
        разметки документа, по которым отбирает фильтр поиска.
    """
    # Отчеты содержат текстовый слой, распознавание не нужно. По умолчанию pymupdf4llm
    # включает его для страниц, которые счел картинками, причем с английской моделью
    # языка, — на русском документе это молча дало бы мусор в индексе.
    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks = True,
        show_progress = False,
        use_ocr = OCRMode.NEVER,
    )

    return [
        Document(
            doc_id = f"{pdf_path.name}:{page['metadata']['page_number']}",
            text = normalize_page(page["text"]),
            metadata = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "page_label": str(page["metadata"]["page_number"]),
                **DOCUMENT_METADATA.get(pdf_path.name, {}),
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

"""Карта страниц: где в тексте файла кончается один лист и начинается следующий.

Нарезка работает с файлом целиком — иначе стек заголовков, порядок чтения и таблицы
рвутся на каждой границе листа. Номер страницы при этом терять нельзя: он стоит
в ссылке ответа. Карта хранит границы листов в символах, и по ней узел получает
страницу своего начала и своего конца.

Страницы склеиваются одним переводом строки: пустая строка между ними разорвала бы
таблицу, продолжающуюся на следующем листе.
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from llama_index.core import Document

# Чем склеиваются страницы в текст файла.
PAGE_JOIN = "\n"


@dataclass(frozen = True)
class PageSpan:
    """Границы страницы в тексте файла.

    Атрибуты:
        number: номер страницы.
        start: смещение первого символа страницы.
        end: смещение за последним символом страницы.
    """

    number: int
    start: int
    end: int


def build_file_text(pages: Sequence[Document]) -> Tuple[str, List[PageSpan]]:
    """Склеивает страницы одного файла в сплошной текст.

    Аргументы:
        pages: страницы файла в порядке чтения.

    Возвращает:
        Текст файла и границы страниц в нём.
    """
    parts: List[str] = []
    spans: List[PageSpan] = []
    offset = 0

    for page in pages:
        text = page.text
        spans.append(
            PageSpan(
                number = int(page.metadata["page_label"]),
                start = offset,
                end = offset + len(text),
            )
        )
        parts.append(text)
        offset += len(text) + len(PAGE_JOIN)

    return PAGE_JOIN.join(parts), spans


def page_at(spans: Sequence[PageSpan], offset: int) -> int:
    """Находит страницу, которой принадлежит смещение.

    Аргументы:
        spans: границы страниц файла в порядке чтения.
        offset: смещение символа в тексте файла.

    Возвращает:
        Номер страницы. Смещение, попавшее на склейку между листами, относится
        к предыдущей странице: там кончается её текст.
    """
    found = spans[0].number

    for span in spans:
        if offset < span.start:
            break

        found = span.number

    return found

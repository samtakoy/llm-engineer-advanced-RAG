"""Проставление страниц узлам, нарезанным из файла целиком.

Узел режется из сплошного текста файла, поэтому страницу он получает по своему месту
в этом тексте: страница начала идёт в `page_label`, страница конца — в `page_end`,
если узел лёг на два листа и больше. Ссылка в ответе печатает диапазон, метрика
зачитывает любую страницу из него.

Место узла ищется так же, как его ищет сам llama-index в `_postprocess_parsed_nodes`:
поиском текста узла в тексте родителя с продвижением позиции по порядку чтения.
Перекрытие соседних узлов этому не мешает — позиция двигается на начало найденного,
а не на конец.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

from llama_index.core import Document
from llama_index.core.schema import BaseNode, MetadataMode

from rag_assistant.ingest.page_map import PageSpan, build_file_text, page_at

# Поля метаданных с первой и последней страницей узла.
PAGE_LABEL_KEY = "page_label"
PAGE_END_KEY = "page_end"


@dataclass(frozen = True)
class PageAttribution:
    """Счёт проставленных страниц.

    Атрибуты:
        located: узлов, нашедших своё место в тексте файла.
        missed: узлов, места не нашедших. Такой узел остаётся без страницы,
            и ссылка на него сошлётся на файл целиком.
        multipage: узлов, легших на два листа и больше.
    """

    located: int
    missed: int
    multipage: int


def assign_pages(nodes: Sequence[BaseNode], pages: Sequence[Document]) -> PageAttribution:
    """Проставляет узлам страницы по их месту в тексте файла.

    Аргументы:
        nodes: узлы после нарезки, в порядке чтения.
        pages: страницы корпуса, из которых собирался текст файлов.

    Возвращает:
        Счёт проставленных страниц. Метаданные узлов меняются на месте.
    """
    texts, spans = build_file_maps(pages)
    cursors: Dict[str, int] = {}
    located = missed = multipage = 0

    for node in nodes:
        file_name = node.metadata.get("file_name", "")
        text = texts.get(file_name)

        if text is None:
            missed += 1
            continue

        content = node.get_content(metadata_mode = MetadataMode.NONE)
        bounds = locate(text = text, content = content, cursor = cursors.get(file_name, 0))

        if bounds is None:
            missed += 1
            continue

        start, end = bounds
        cursors[file_name] = start + 1
        located += 1

        first = page_at(spans = spans[file_name], offset = start)
        last = page_at(spans = spans[file_name], offset = end - 1)
        node.metadata[PAGE_LABEL_KEY] = str(first)

        if last > first:
            node.metadata[PAGE_END_KEY] = str(last)
            multipage += 1

    return PageAttribution(located = located, missed = missed, multipage = multipage)


def locate(text: str, content: str, cursor: int) -> tuple[int, int] | None:
    """Находит место узла в тексте файла.

    Ищется не весь текст узла, а его первая и последняя непустые строки: цепочка
    парсеров переставляет пробелы между строками, а нарезка таблицы повторяет шапку
    в каждой части, и дословно такой текст в файле не встречается.

    Аргументы:
        text: текст файла.
        content: текст узла.
        cursor: позиция, с которой идёт поиск: узлы разбираются по порядку чтения.

    Возвращает:
        Границы узла в тексте файла либо None, если место не нашлось.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        return None

    end = text.find(lines[-1], cursor)

    if end < 0:
        return None

    start = text.find(lines[0], cursor)

    # Повторённая шапка таблицы стоит в файле выше своей части и уже пройдена
    # курсором: тогда началом узла считается его последняя строка.
    if start < 0 or start > end:
        start = end

    return start, end + len(lines[-1])


def build_file_maps(
    pages: Sequence[Document],
) -> tuple[Dict[str, str], Dict[str, List[PageSpan]]]:
    """Собирает текст и границы страниц по каждому файлу.

    Аргументы:
        pages: страницы корпуса в любом порядке.

    Возвращает:
        Текст файла и границы его страниц по имени файла.
    """
    by_file: Dict[str, List[Document]] = {}

    for page in pages:
        by_file.setdefault(page.metadata["file_name"], []).append(page)

    texts: Dict[str, str] = {}
    spans: Dict[str, List[PageSpan]] = {}

    for file_name, file_pages in by_file.items():
        file_pages.sort(key = lambda page: int(page.metadata["page_label"]))
        texts[file_name], spans[file_name] = build_file_text(file_pages)

    return texts, spans

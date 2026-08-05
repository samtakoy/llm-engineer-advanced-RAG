"""Снятие лишней шапки у таблицы, продолжающейся на следующем листе.

Разбор PDF ищет таблицы на каждой странице отдельно, поэтому продолжению таблицы
он ставит свою строку-разделитель. В сплошном тексте файла она оказывается посреди
таблицы и разрывает разметку: строки после второго разделителя перестают быть
той же таблицей.

Текст при этом остаётся на своей странице — двигать его нельзя, номер страницы
узла считается по смещению в файле.
"""
from typing import List, Sequence

from llama_index.core import Document

# Строка таблицы markdown и символы, из которых состоит строка-разделитель.
TABLE_LINE_PREFIX = "|"
SEPARATOR_CHARACTERS = set("|-: ")


def drop_continuation_separators(pages: Sequence[Document]) -> List[Document]:
    """Убирает строку-разделитель у страниц, продолжающих таблицу.

    Аргументы:
        pages: страницы корпуса в любом порядке.

    Возвращает:
        Те же страницы в порядке чтения. У страницы, начинающейся продолжением
        таблицы, снята строка-разделитель.
    """
    ordered = sorted(pages, key = reading_order_key)

    for previous, current in zip(ordered, ordered[1:]):
        if continues_table(previous = previous, current = current):
            current.set_content(strip_leading_separator(current.text))

    return ordered


def reading_order_key(document: Document) -> tuple:
    """Возвращает ключ сортировки страниц в порядке чтения.

    Аргументы:
        document: страница корпуса.

    Возвращает:
        Имя файла и номер страницы.
    """
    return document.metadata["file_name"], int(document.metadata["page_label"])


def is_table_line(line: str) -> bool:
    """Сообщает, является ли строка строкой таблицы markdown.

    Аргументы:
        line: строка текста.

    Возвращает:
        True, если строка начинается с вертикальной черты.
    """
    return line.strip().startswith(TABLE_LINE_PREFIX)


def is_separator_line(line: str) -> bool:
    """Сообщает, является ли строка разделителем под шапкой таблицы.

    Аргументы:
        line: строка таблицы markdown.

    Возвращает:
        True, если строка состоит только из черт, дефисов и двоеточий.
    """
    stripped = line.strip()

    return bool(stripped) and set(stripped) <= SEPARATOR_CHARACTERS and "-" in stripped


def count_columns(line: str) -> int:
    """Считает число колонок в строке таблицы.

    Аргументы:
        line: строка вида «|ячейка|ячейка|».

    Возвращает:
        Количество ячеек.
    """
    return line.strip().count(TABLE_LINE_PREFIX) - 1


def filled_lines(text: str) -> List[str]:
    """Возвращает непустые строки текста.

    Аргументы:
        text: текст страницы.

    Возвращает:
        Строки без пустых.
    """
    return [line for line in text.splitlines() if line.strip()]


def continues_table(previous: Document, current: Document) -> bool:
    """Решает, продолжает ли страница таблицу предыдущей.

    Аргументы:
        previous: предыдущая страница.
        current: очередная страница.

    Возвращает:
        True, если это одна таблица на двух листах: тот же файл, соседние страницы,
        предыдущая кончается строкой таблицы, текущая начинается ею же — без подписи
        и заголовка над таблицей, — и число колонок совпадает.
    """
    if previous.metadata["file_name"] != current.metadata["file_name"]:
        return False

    if int(current.metadata["page_label"]) != int(previous.metadata["page_label"]) + 1:
        return False

    tail = filled_lines(previous.text)
    head = filled_lines(current.text)

    if not tail or not head or not is_table_line(tail[-1]) or not is_table_line(head[0]):
        return False

    return count_columns(tail[-1]) == count_columns(head[0])


def strip_leading_separator(text: str) -> str:
    """Убирает строку-разделитель в начале страницы.

    Аргументы:
        text: текст страницы в markdown.

    Возвращает:
        Текст без разделителя среди первых строк таблицы. Прочие строки остаются
        на своих местах: их смещение задаёт номер страницы узла.
    """
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        if not is_table_line(line):
            break

        if is_separator_line(line):
            return "\n".join(lines[:index] + lines[index + 1 :])

    return text

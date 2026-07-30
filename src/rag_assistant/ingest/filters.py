"""Отбор страниц, которые не должны попасть в индекс."""
from typing import List

from llama_index.core import Document

# Заголовок оглавления стоит в самом начале страницы.
CONTENTS_HEADINGS = ("оглавление", "содержание")
HEADING_ZONE_CHARS = 150

# Отточие — цепочка точек между названием раздела и номером страницы.
DOT_LEADER = "...."
MIN_DOT_LEADERS = 10


def drop_service_pages(documents: List[Document]) -> List[Document]:
    """Убирает страницы, не несущие содержания.

    Аргументы:
        documents: страницы всех документов.

    Возвращает:
        Страницы без оглавления.
    """
    return [document for document in documents if not is_contents_page(document)]


def is_contents_page(document: Document) -> bool:
    """Определяет, является ли страница оглавлением.

    Оглавление дословно совпадает с вопросами, сформулированными как название
    раздела, но фактов не содержит и вытесняет из выдачи страницы с ответом.

    Аргументы:
        document: страница документа.

    Возвращает:
        True, если страница похожа на оглавление.
    """
    return starts_with_contents_heading(document) or consists_of_dot_leaders(document)


def starts_with_contents_heading(document: Document) -> bool:
    """Проверяет, открывается ли страница заголовком оглавления.

    Аргументы:
        document: страница документа.

    Возвращает:
        True, если в начале страницы стоит слово «оглавление» или «содержание».
    """
    heading_zone = document.text[:HEADING_ZONE_CHARS].lower()

    return any(heading in heading_zone for heading in CONTENTS_HEADINGS)


def consists_of_dot_leaders(document: Document) -> bool:
    """Проверяет, набрана ли страница строками с отточием.

    Так выглядит продолжение оглавления: заголовка на нём уже нет,
    а строки «название раздела ....... 12» остались.

    Аргументы:
        document: страница документа.

    Возвращает:
        True, если строк с отточием больше порога.
    """
    return document.text.count(DOT_LEADER) >= MIN_DOT_LEADERS

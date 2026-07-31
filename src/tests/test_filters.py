"""Проверка отсева страниц, не несущих содержания."""
from llama_index.core import Document

from rag_assistant.ingest.filters import drop_service_pages, is_contents_page


def make_page(text: str, page_label: str = "1") -> Document:
    """Собирает страницу документа для проверки.

    Аргументы:
        text: текст страницы.
        page_label: номер страницы.

    Возвращает:
        Document с теми же метаданными, что ставит загрузчик.
    """
    return Document(
        text = text,
        metadata = {"file_name": "отчёт.pdf", "page_label": page_label},
    )


def test_page_starting_with_contents_heading_is_dropped() -> None:
    """Первая страница оглавления опознаётся по заголовку."""
    page = make_page("# ОГЛАВЛЕНИЕ\n\nРаздел 1. Управленческий отчёт")

    assert is_contents_page(page)


def test_contents_heading_is_case_insensitive() -> None:
    """Регистр заголовка значения не имеет."""
    assert is_contents_page(make_page("## Содержание\n\nРаздел 1"))


def test_contents_word_below_heading_zone_does_not_drop_page() -> None:
    """Слово «содержание» в теле страницы — это обычный текст, а не оглавление."""
    page = make_page("А" * 200 + "\n\nраскрывается содержание договора")

    assert not is_contents_page(page)


def test_page_of_dot_leaders_is_dropped() -> None:
    """Продолжение оглавления заголовка уже не имеет, но отточия на нём остались."""
    page = make_page("\n".join(f"Раздел {number} ........ {number}" for number in range(12)))

    assert is_contents_page(page)


def test_few_dot_leaders_do_not_drop_page() -> None:
    """Пара отточий в обычном тексте страницу не отсеивает."""
    page = make_page("Показатель .... 12\nДругой показатель .... 15")

    assert not is_contents_page(page)


def test_ordinary_page_survives() -> None:
    """Содержательная страница остаётся в индексе."""
    page = make_page("Консолидированная выручка Группы составила 253,7 млн евро.")

    assert not is_contents_page(page)


def test_drop_service_pages_keeps_order_and_removes_only_contents() -> None:
    """Из потока страниц выпадает только оглавление, порядок остальных сохраняется."""
    pages = [
        make_page("Введение", page_label = "1"),
        make_page("# Оглавление\nРаздел 1", page_label = "2"),
        make_page("Выручка составила 253,7 млн евро", page_label = "3"),
    ]

    kept = drop_service_pages(pages)

    assert [page.metadata["page_label"] for page in kept] == ["1", "3"]

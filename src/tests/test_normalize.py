"""Проверка починки markdown после разбора PDF."""
from rag_assistant.ingest.normalize import (
    demote_false_heading,
    normalize_page,
    replace_bullet_marks,
    strip_emphasis,
    unwrap_single_column_tables,
)


def test_strip_emphasis_unwraps_bold_italic() -> None:
    """Жирный курсив вокруг обычного абзаца снимается."""
    assert strip_emphasis("**_Выручка выросла._**") == "Выручка выросла."


def test_strip_emphasis_spans_line_breaks() -> None:
    """Выделение снимается и когда абзац разорван переносом строки."""
    assert strip_emphasis("**_первая\nвторая_**") == "первая\nвторая"


def test_strip_emphasis_keeps_plain_bold() -> None:
    """Обычный жирный текст не трогается: под замену идёт только жирный курсив."""
    assert strip_emphasis("**Раздел 1**") == "**Раздел 1**"


def test_demote_false_heading_drops_marks_on_lowercase_start() -> None:
    """Заголовок со строчной буквы — это абзац, перенесённый с прошлой страницы."""
    assert demote_false_heading("#### деятельности эмитента") == "деятельности эмитента"


def test_demote_false_heading_keeps_real_heading() -> None:
    """Настоящий заголовок начинается с прописной буквы и остаётся заголовком."""
    assert demote_false_heading("## Раздел 1") == "## Раздел 1"


def test_demote_false_heading_looks_past_empty_lines() -> None:
    """Пустые строки перед заголовком не мешают правилу сработать."""
    assert demote_false_heading("\n\n# продолжение абзаца") == "\n\nпродолжение абзаца"


def test_demote_false_heading_checks_only_first_line() -> None:
    """Правило смотрит первую непустую строку: ниже по странице заголовки не трогает."""
    text = "## Раздел 1\n#### продолжение"
    assert demote_false_heading(text) == text


def test_replace_bullet_marks_normalizes_all_variants() -> None:
    """Маркеры «•» и «▪» приводятся к дефису."""
    assert replace_bullet_marks("• первый\n▪ второй") == "- первый\n- второй"


def test_replace_bullet_marks_converts_question_mark_bullet() -> None:
    """Отдельно стоящий «?» — это маркер списка, потерявшийся при экспорте из Word."""
    assert replace_bullet_marks("\n? пункт списка") == "\n- пункт списка"


def test_replace_bullet_marks_keeps_real_question_mark() -> None:
    """Настоящий вопросительный знак стоит вплотную к слову и остаётся на месте."""
    assert replace_bullet_marks("Какая выручка? Ответ ниже.") == "Какая выручка? Ответ ниже."


def test_unwrap_single_column_tables_turns_frame_into_list() -> None:
    """Рамка вокруг списка приходит таблицей из одной колонки и разворачивается обратно."""
    text = "|- первый|\n|---|\n|- второй|"
    assert unwrap_single_column_tables(text) == "- первый\n- второй"


def test_unwrap_single_column_tables_glues_continuation_rows() -> None:
    """Строка без маркера продолжает предыдущий пункт, а не начинает новый."""
    text = "|- первый пункт|\n|---|\n|продолжение пункта|"
    assert unwrap_single_column_tables(text) == "- первый пункт продолжение пункта"


def test_unwrap_single_column_tables_replaces_line_breaks_in_cells() -> None:
    """Перенос строки внутри ячейки становится пробелом."""
    text = "|- первый<br>пункт|"
    assert unwrap_single_column_tables(text) == "- первый пункт"


def test_unwrap_single_column_tables_keeps_real_tables() -> None:
    """Настоящая таблица с несколькими колонками остаётся нетронутой."""
    text = "|Показатель|2024|\n|---|---|\n|Выручка|253.7|"
    assert unwrap_single_column_tables(text) == text


def test_unwrap_single_column_tables_keeps_surrounding_text() -> None:
    """Текст до и после таблицы сохраняется вместе с пустыми строками."""
    text = "До таблицы\n\n|Показатель|2024|\n|---|---|\n\nПосле таблицы"
    assert unwrap_single_column_tables(text) == text


def test_unwrap_single_column_tables_handles_table_at_end_of_page() -> None:
    """Таблица последней строкой страницы не теряется и не тянет за собой лишний перенос."""
    assert unwrap_single_column_tables("Абзац\n|- пункт|") == "Абзац\n- пункт"


def test_normalize_page_applies_whole_chain() -> None:
    """Все правила отрабатывают вместе: выделение, ложный заголовок, рамка-таблица, маркеры."""
    text = "#### продолжение абзаца\n\n**_Выделенный текст._**\n\n|• пункт|\n|---|"
    expected = "продолжение абзаца\n\nВыделенный текст.\n\n- пункт"

    assert normalize_page(text) == expected


def test_normalize_page_keeps_empty_page_empty() -> None:
    """Пустая страница не превращается в мусор."""
    assert normalize_page("") == ""

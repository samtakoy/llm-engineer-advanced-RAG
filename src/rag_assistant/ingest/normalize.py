"""Починка markdown после разбора PDF."""
import re
from typing import List

# Жирный курсив, которым разбор помечает обычные абзацы отчёта.
EMPHASIS_PATTERN = re.compile(r"\*\*_(.+?)_\*\*", re.DOTALL)

# Маркер списка, превратившийся в «?» при экспорте отчёта из Word.
QUESTION_MARK_BULLET_PATTERN = re.compile(r"(?<=\s)\?(?=\s)")

# Прочие символы, которыми в отчёте набраны маркеры списка.
BULLET_CHARACTER_PATTERN = re.compile(r"[•▪]")

# Строка markdown-таблицы и строка-разделитель под её шапкой.
TABLE_ROW_PATTERN = re.compile(r"^\|.*\|\s*$")
TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s:|-]+\|\s*$")

# Заголовок markdown.
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(?P<text>.*)")


def normalize_page(text: str) -> str:
    """Приводит страницу к читаемому markdown.

    Аргументы:
        text: страница как её выдал разбор PDF.

    Возвращает:
        Текст без лишних выделений и ложных заголовков, со списками вместо
        одноколоночных таблиц и с дефисом в маркерах списка.
    """
    text = strip_emphasis(text)
    text = demote_false_heading(text)
    text = unwrap_single_column_tables(text)

    return replace_bullet_marks(text)


def demote_false_heading(text: str) -> str:
    """Снимает заголовок с первой строки страницы, если она продолжает абзац.

    Разбор помечает заголовком строку, набранную крупнее основного текста,
    и ошибается на абзаце, перенесённом с предыдущей страницы. Такая строка
    начинается со строчной буквы — заголовки в отчёте начинаются с прописной.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором ложный заголовок стал обычным абзацем.
    """
    lines = text.split("\n")

    for index, line in enumerate(lines):
        if not line.strip():
            continue

        heading = HEADING_PATTERN.match(line.strip())
        if heading and heading.group("text").lstrip()[:1].islower():
            lines[index] = heading.group("text")
        break

    return "\n".join(lines)


def strip_emphasis(text: str) -> str:
    """Снимает жирный курсив с обычного текста.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст без обрамления «**_ _**».
    """
    return EMPHASIS_PATTERN.sub(r"\1", text)


def replace_bullet_marks(text: str) -> str:
    """Приводит маркеры списка к дефису.

    В отчёте встречаются три вида маркера: дефис, «•» и «?», получившийся
    из маркера при экспорте из Word. Настоящий вопросительный знак стоит
    вплотную к слову, поэтому под замену попадает только «?» между пробелами.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором маркером списка везде служит дефис.
    """
    return BULLET_CHARACTER_PATTERN.sub("-", QUESTION_MARK_BULLET_PATTERN.sub("-", text))


def unwrap_single_column_tables(text: str) -> str:
    """Разворачивает одноколоночные таблицы обратно в список.

    Одна колонка означает, что таблицей стала рамка вокруг маркированного
    списка. Настоящих таблиц с одной колонкой в отчёте не бывает.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором такие таблицы заменены списком, а настоящие
        оставлены как есть.
    """
    lines: List[str] = []
    table_rows: List[str] = []

    for line in text.split("\n") + [""]:
        if TABLE_ROW_PATTERN.match(line):
            table_rows.append(line)
            continue

        lines.extend(render_table(table_rows))
        table_rows = []
        lines.append(line)

    return "\n".join(lines[:-1])


def render_table(rows: List[str]) -> List[str]:
    """Готовит строки таблицы к выводу.

    Аргументы:
        rows: подряд идущие строки одной таблицы.

    Возвращает:
        Те же строки, если колонок больше одной. Иначе — список, в котором
        пункт начинается с дефиса, а строки-продолжения приклеены к пункту.
    """
    if not rows or count_columns(rows[0]) > 1:
        return rows

    lines: List[str] = []

    for row in rows:
        if TABLE_SEPARATOR_PATTERN.match(row):
            continue

        cell = row.strip().strip("|").replace("<br>", " ").strip()
        if cell.startswith("-"):
            lines.append(f"- {cell.lstrip('-').strip()}")
        elif lines:
            lines[-1] = f"{lines[-1]} {cell}"
        else:
            lines.append(cell)

    return lines


def count_columns(row: str) -> int:
    """Считает число колонок в строке markdown-таблицы.

    Аргументы:
        row: строка вида «|ячейка|ячейка|».

    Возвращает:
        Количество ячеек.
    """
    return row.strip().count("|") - 1

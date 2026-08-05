"""Починка markdown после разбора PDF."""
import re
from typing import List

# Жирный курсив, которым разбор помечает обычные абзацы отчёта.
EMPHASIS_PATTERN = re.compile(r"\*\*_(.+?)_\*\*", re.DOTALL)

# Жирный, которым разбор помечает имена полей формы: «|**Полное фирменное**
# **наименование эмитента:**|». Снимается после разметки заголовков — до неё
# по этому знаку строка отличается от обычного абзаца.
BOLD_PATTERN = re.compile(r"\*\*")

# Маркер списка, превратившийся в «?» при экспорте отчёта из Word.
QUESTION_MARK_BULLET_PATTERN = re.compile(r"(?<=\s)\?(?=\s)")

# Прочие символы, которыми в отчёте набраны маркеры списка.
BULLET_CHARACTER_PATTERN = re.compile(r"[•▪]")

# Строка markdown-таблицы и строка-разделитель под её шапкой.
TABLE_ROW_PATTERN = re.compile(r"^\|.*\|\s*$")
TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s:|-]+\|\s*$")

# Заголовок markdown.
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(?P<text>.*)")

# Номер раздела в начале заголовка: «1.», «1.4.», «1.4.1.». Глубина номера задаёт
# уровень заголовка — разбор PDF ставит уровень по размеру шрифта, поэтому «Раздел 4»
# и «4.1.» у него оба второго уровня, то есть братья, и вложенности разделов нет.
# Одно число без точки номером раздела не считается: так набрана подпись таблицы
# «10 крупнейших частных многопрофильных клиник в России», и она становилась
# заголовком первого уровня, под который уходил весь остаток документа.
SECTION_NUMBER_PATTERN = re.compile(r"^\**\s*(?P<number>\d+(?:\.\d+)+|\d+(?=\.))\.?(?:\s|\*|$)")

# Заголовок раздела верхнего уровня: «Раздел 4. Дополнительные сведения...».
SECTION_WORD_PATTERN = re.compile(r"^\**\s*Раздел\s+\d+")

# Заголовок, набранный курсивом целиком: в отчёте курсивом набран обычный текст,
# и такая строка — абзац, а не заголовок.
ITALIC_HEADING_PATTERN = re.compile(r"^\*\*_.*_\*\*$")

# Начало строки, которая продолжает предыдущую: строчная буква или открытая скобка.
# Заголовки отчёта начинаются с прописной.
HEADING_TAIL_PATTERN = re.compile(r"^[(a-zа-яё]")

# Подпись поля формы раскрытия: «Основной государственный регистрационный номер (ОГРН):».
# Разбор PDF помечает её заголовком, потому что она набрана жирным.
FIELD_LABEL_MARK = ":"

# Предельный уровень заголовка markdown.
MAX_HEADING_LEVEL = 6

# Перенос строки внутри ячейки таблицы: разбор PDF ставит его в длинных заголовках.
CELL_BREAK = "<br>"


def normalize_page(text: str) -> str:
    """Приводит страницу к читаемому markdown.

    Аргументы:
        text: страница как её выдал разбор PDF.

    Возвращает:
        Текст без колонтитула, лишних выделений и ложных заголовков, со списками
        вместо одноколоночных таблиц и с дефисом в маркерах списка.
    """
    text = strip_page_footer(text)
    text = retag_headings(text)
    text = strip_emphasis(text)
    text = demote_false_heading(text)
    text = unwrap_single_column_tables(text)
    text = flatten_table_cells(text)
    text = join_wrapped_table_rows(text)

    return replace_bullet_marks(text)


def strip_page_footer(text: str) -> str:
    """Убирает колонтитул с номером страницы.

    Номер страницы стоит отдельной строкой в начале или в конце листа. В тексте он
    не значит ничего, а таблицу, продолжающуюся на следующем листе, отрезает
    от продолжения: страница кончается не строкой таблицы, а номером.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст без строки-номера по краям страницы. Число внутри страницы остаётся:
        там оно часть содержания.
    """
    lines = text.split("\n")
    filled = [index for index, line in enumerate(lines) if line.strip()]

    if not filled:
        return text

    return "\n".join(
        line
        for index, line in enumerate(lines)
        if not (index in (filled[0], filled[-1]) and line.strip().isdigit())
    )


def flatten_table_cells(text: str) -> str:
    """Убирает перенос строки внутри ячеек таблицы.

    Разбор PDF ставит «<br>» там, где текст не поместился в ширину колонки.
    Для эмбеддера это разметка посреди слова, а не перенос.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором строки таблиц не содержат «<br>», а подряд идущие
        пробелы сжаты до одного.
    """
    return "\n".join(
        re.sub(r"[ \t]{2,}", " ", line.replace(CELL_BREAK, " "))
        if line.strip().startswith("|")
        else line
        for line in text.split("\n")
    )


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


def retag_headings(text: str) -> str:
    """Переставляет заголовки: ложные снимает, настоящим ставит уровень по номеру.

    Разбор PDF помечает заголовком всё, что набрано крупнее или жирнее основного
    текста: подписи полей формы раскрытия, куски абзацев, перенесённые хвосты
    заголовков. Нарезка режет по каждому такому знаку, и «ОГРН: 1233900014985»
    становится узлом в 76 символов. Уровень разбор ставит по размеру шрифта,
    поэтому «Раздел 4» и «4.1.» оказываются братьями, вложенности разделов нет,
    и путь заголовков в метаданных узла пуст.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором ложные заголовки стали абзацами, а у настоящих уровень
        соответствует глубине номера раздела.
    """
    lines = []

    for line in text.split("\n"):
        heading = HEADING_PATTERN.match(line.strip())

        if heading is None:
            lines.append(line)
            continue

        heading_text = heading.group("text").strip()
        level = heading_level(heading_text)
        lines.append(heading_text if level is None else f"{'#' * level} {heading_text}")

    return "\n".join(lines)


def heading_level(heading_text: str) -> int | None:
    """Решает, заголовок ли это, и какого уровня.

    Правила разбираются по порядку: номер раздела сильнее всех прочих признаков,
    иначе заголовок «1.4.1. Основные показатели, рассчитываемые на основе
    консолидированной отчётности:» потерялся бы из-за двоеточия.

    Аргументы:
        heading_text: текст заголовка без знаков решётки.

    Возвращает:
        Уровень заголовка либо None, если это не заголовок и решётки надо снять:
        строка набрана курсивом целиком, продолжает предыдущую или подписывает
        поле формы раскрытия.
    """
    if SECTION_WORD_PATTERN.match(heading_text):
        return 1

    number = SECTION_NUMBER_PATTERN.match(heading_text)
    if number is not None:
        return min(number.group("number").count(".") + 1, MAX_HEADING_LEVEL)

    if ITALIC_HEADING_PATTERN.match(heading_text):
        return None

    if HEADING_TAIL_PATTERN.match(heading_text.lstrip("*_ ")):
        return None

    if FIELD_LABEL_MARK in heading_text:
        return None

    return 2


def strip_emphasis(text: str) -> str:
    """Снимает выделение с текста страницы.

    Жирным разбор помечает имена полей формы, причём разрывает его посреди имени:
    «**Полное фирменное** **наименование эмитента:**».

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст без обрамления «**_ _**» и без жирного.
    """
    return BOLD_PATTERN.sub("", EMPHASIS_PATTERN.sub(r"\1", text))


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


def join_wrapped_table_rows(text: str) -> str:
    """Склеивает строку таблицы с продолжением её ячейки.

    Текст ячейки, не влезший в ширину колонки, разбор PDF выносит следующей строкой
    таблицы, а остальные колонки в ней оставляет пустыми: «|||среднегодовому размеру|||».
    Логическая строка оказывается разбита на несколько, и числа стоят только в первой.

    Продолжением считается строка с единственной заполненной ячейкой, у которой
    в предыдущей строке та же ячейка тоже заполнена. Иначе это начало своей строки:
    таблицы отчёта содержат строки, где заполнена только первая колонка.

    Аргументы:
        text: страница в markdown.

    Возвращает:
        Текст, в котором продолжения ячеек приклеены к своим строкам.
    """
    lines: List[str] = []

    for line in text.split("\n"):
        continued = continuation_column(previous = lines[-1] if lines else "", line = line)

        if continued is None:
            lines.append(line)
            continue

        cells = split_row(lines[-1])
        cells[continued] = f"{cells[continued]} {split_row(line)[continued]}".strip()
        lines[-1] = "|" + "|".join(cells) + "|"

    return "\n".join(lines)


def continuation_column(previous: str, line: str) -> int | None:
    """Находит колонку, продолжение которой стоит в строке.

    Аргументы:
        previous: предыдущая строка страницы.
        line: очередная строка страницы.

    Возвращает:
        Номер колонки либо None, если строка продолжением не является.
    """
    if not TABLE_ROW_PATTERN.match(previous) or not TABLE_ROW_PATTERN.match(line):
        return None

    if TABLE_SEPARATOR_PATTERN.match(previous) or TABLE_SEPARATOR_PATTERN.match(line):
        return None

    previous_cells = split_row(previous)
    cells = split_row(line)

    if len(previous_cells) != len(cells):
        return None

    filled = [index for index, cell in enumerate(cells) if cell.strip()]

    if len(filled) != 1 or not previous_cells[filled[0]].strip():
        return None

    return filled[0]


def split_row(row: str) -> List[str]:
    """Разбирает строку таблицы markdown на ячейки.

    Аргументы:
        row: строка вида «|ячейка|ячейка|».

    Возвращает:
        Ячейки строки. Пустая ячейка остаётся пустой строкой: по её месту
        определяется колонка.
    """
    stripped = row.strip()

    return stripped[1:-1].split("|") if stripped.startswith("|") else stripped.split("|")


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

        cell = row.strip().strip("|").replace(CELL_BREAK, " ").strip()
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

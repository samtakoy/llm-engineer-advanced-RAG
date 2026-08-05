"""Нарезка таблицы по строкам с повтором шапки.

`MarkdownNodeParser` ставит границы по заголовкам и про таблицы не знает, а `SentenceSplitter`
дорезает по предложениям: таблица, не влезшая в окно эмбеддера, рвётся посреди строк,
и во второй части остаются числа без имён колонок — «|Амортизация|22,6|16,5|» не говорит
ни про год, ни про единицы.

Здесь таблица режется по своим строкам, и каждая часть начинается шапкой. Часть остаётся
читаемой сама по себе, а не превращается в столбец чисел.

Шаг ставится между `MarkdownNodeParser` и `SentenceSplitter`: границы по структуре
сначала, предел длины последним.
"""
from typing import Callable, Iterator, List, Tuple

from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
from pydantic import Field

# Строка таблицы markdown.
TABLE_LINE_PREFIX = "|"

# Из чего состоит строка-разделитель под шапкой: «|---|---|».
SEPARATOR_CHARACTERS = set("|-: ")


class TableRowSplitter(MetadataAwareTextSplitter):
    """Режет таблицы по строкам, повторяя шапку в каждой части.

    Текст вокруг таблиц не трогает: его длину ограничивает следующий шаг цепочки.

    Атрибуты:
        chunk_size: предел длины части в токенах эмбеддера.
        tokenizer: счётчик токенов той модели, которая будет векторизовать текст.
    """

    chunk_size: int = Field(description = "Предел длины части в токенах эмбеддера.")
    tokenizer: Callable[[str], List[int]] = Field(
        exclude = True,
        description = "Счётчик токенов модели эмбеддингов.",
    )

    @classmethod
    def class_name(cls) -> str:
        """Имя класса для сериализации llama-index.

        Возвращает:
            Устойчивое имя, по которому парсер восстанавливается из настроек.
        """
        return "TableRowSplitter"

    @classmethod
    def create(
        cls,
        chunk_size: int,
        tokenizer: Callable[[str], List[int]],
    ) -> "TableRowSplitter":
        """Собирает сплиттер.

        Аргументы:
            chunk_size: предел длины части в токенах эмбеддера.
            tokenizer: счётчик токенов модели эмбеддингов.

        Возвращает:
            Готовый сплиттер.
        """
        return cls(chunk_size = chunk_size, tokenizer = tokenizer)

    def fits(self, lines: List[str], limit: int) -> bool:
        """Проверяет, помещаются ли строки в отведённый предел.

        Аргументы:
            lines: строки таблицы.
            limit: предел длины в токенах эмбеддера.

        Возвращает:
            True, если их длина в токенах не больше предела.
        """
        return len(self.tokenizer("\n".join(lines))) <= limit

    def read_blocks(self, text: str) -> Iterator[Tuple[bool, List[str]]]:
        """Разбирает текст на таблицы и всё остальное.

        Аргументы:
            text: текст узла в markdown.

        Возвращает:
            Пары «это таблица, строки блока» в порядке чтения. Пустая строка
            блок не закрывает: разбор PDF ставит её и посреди таблицы, а таблица,
            разорванная на этом месте, теряет шапку.
        """
        current: List[str] = []
        current_is_table = False

        for line in text.splitlines():
            if not line.strip():
                current.append(line)
                continue

            line_is_table = line.strip().startswith(TABLE_LINE_PREFIX)

            if current and line_is_table != current_is_table:
                yield current_is_table, current
                current = []

            current_is_table = line_is_table
            current.append(line)

        if current:
            yield current_is_table, current

    def split_header(self, rows: List[str]) -> Tuple[List[str], List[str]]:
        """Отделяет шапку таблицы от её тела.

        Аргументы:
            rows: строки таблицы в порядке чтения.

        Возвращает:
            Шапку и тело. Шапка — строки до разделителя включительно, а если
            разделителя нет — первая строка: без неё колонки безымянные.
        """
        for index, row in enumerate(rows):
            stripped = row.strip()

            if set(stripped) <= SEPARATOR_CHARACTERS and "-" in stripped:
                return rows[: index + 1], rows[index + 1 :]

        return rows[:1], rows[1:]

    def split_table(self, rows: List[str], limit: int) -> List[str]:
        """Режет таблицу на части, помещающиеся в отведённый предел.

        Аргументы:
            rows: строки таблицы в порядке чтения.
            limit: предел длины части в токенах эмбеддера.

        Возвращает:
            Таблицу целиком, если она помещается, иначе её части. Каждая часть
            начинается шапкой. Строка, которая с шапкой не помещается и одна,
            отдаётся как есть: дорезать её по строкам таблицы больше нечем,
            это сделает предел длины следующим шагом.
        """
        if self.fits(lines = rows, limit = limit):
            return ["\n".join(rows)]

        header, body = self.split_header(rows)
        parts: List[str] = []
        current: List[str] = []

        for row in body:
            if current and not self.fits(lines = header + current + [row], limit = limit):
                parts.append("\n".join(header + current))
                current = []

            current.append(row)

        if current:
            parts.append("\n".join(header + current))

        return parts

    def split_text(self, text: str) -> List[str]:
        """Режет текст, дробя длинные таблицы по строкам.

        Аргументы:
            text: текст узла в markdown.

        Возвращает:
            Части текста в пределах окна эмбеддера.
        """
        return self.split_within(text = text, limit = self.chunk_size)

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """Режет текст, оставляя место под метаданные узла.

        Метаданные уходят в вектор вместе с текстом и занимают часть окна. Без этого
        запаса часть таблицы, влезающая по тексту, не влезает как узел, и её дорезает
        предел длины — по предложениям, посреди строк, снова отрывая числа от шапки.

        Аргументы:
            text: текст узла в markdown.
            metadata_str: метаданные узла в том виде, в котором они идут в вектор.

        Возвращает:
            Части текста, помещающиеся в окно вместе с метаданными.
        """
        reserved = len(self.tokenizer(metadata_str)) if metadata_str else 0

        return self.split_within(text = text, limit = self.chunk_size - reserved)

    def split_within(self, text: str, limit: int) -> List[str]:
        """Режет текст, дробя таблицы, не влезающие в предел.

        Аргументы:
            text: текст узла в markdown.
            limit: предел длины части в токенах эмбеддера.

        Возвращает:
            Сам текст, если он в предел помещается: резать нечего, а лишняя граница
            отрывала бы таблицу от подписи над ней. Иначе — части, где таблица
            разбита по строкам и каждая часть начинается шапкой, а текст вокруг
            таблиц отдан как есть: его длину ограничивает следующий шаг цепочки.
        """
        if self.fits(lines = [text], limit = limit):
            return [text]

        parts: List[str] = []

        for is_table, lines in self.read_blocks(text):
            if is_table:
                parts.extend(self.split_table(rows = lines, limit = limit))
            else:
                parts.append("\n".join(lines))

        return [part for part in parts if part.strip()]

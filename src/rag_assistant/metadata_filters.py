"""Фильтр поиска по метаданным документов.

Фильтр задаётся человеком явно строкой вида "year:2024|year:2025", из вопроса ничего
не выводится. Пары с одинаковым ключом расширяют выбор по полю, пары с разными
ключами сужают выдачу каждая своим условием.

Chroma применяет такой фильтр до поиска соседей, поэтому top_k набирается уже
из отобранных документов, а не обрезается после поиска по всему корпусу.

Значения полей — строки, включая год. Так тег из командной строки попадает в фильтр
как есть, без догадок о типе поля: несовпадение типа Chroma не считает ошибкой и
молча отдаёт пустую выдачу. Цена — сравнения по порядку («с 2024 года и новее»)
недоступны: числовые операторы Chroma принимает только от чисел.
"""
import re
from typing import Dict, List, Sequence

from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

# Пары перечисляются через вертикальную черту или запятую: "year:2024|year:2025".
PAIR_SEPARATORS = re.compile(r"[|,]")
KEY_VALUE_SEPARATOR = ":"


class TagFormatError(ValueError):
    """Тег записан не в виде «ключ:значение»."""


def parse_tags(value: str | None) -> List[str]:
    """Разбирает строку тегов, переданную из командной строки.

    Аргументы:
        value: теги вида "year:2024|year:2025" либо None, если фильтр не задан.

    Возвращает:
        Теги в порядке перечисления, без пустых значений.
    """
    if not value:
        return []

    return [tag.strip() for tag in PAIR_SEPARATORS.split(value) if tag.strip()]


def group_tags(tags: Sequence[str]) -> Dict[str, List[str]]:
    """Группирует теги по имени поля метаданных.

    Аргументы:
        tags: теги вида "ключ:значение".

    Возвращает:
        Значения по каждому полю, поля и значения — в порядке перечисления.

    Исключения:
        TagFormatError: тег записан без разделителя между ключом и значением.
    """
    grouped: Dict[str, List[str]] = {}

    for tag in tags:
        key, separator, value = tag.partition(KEY_VALUE_SEPARATOR)
        if not separator or not key.strip() or not value.strip():
            raise TagFormatError(
                f"Тег «{tag}» записан не как «ключ:значение», например year:2024",
            )
        grouped.setdefault(key.strip(), []).append(value.strip())

    return grouped


def build_field_filter(key: str, values: Sequence[str]) -> MetadataFilter:
    """Собирает условие по одному полю метаданных.

    Аргументы:
        key: имя поля метаданных.
        values: значения поля, перечисленные в тегах.

    Возвращает:
        Условие «поле равно значению» либо «поле входит в перечисленные значения».
    """
    if len(values) == 1:
        return MetadataFilter(key = key, value = values[0], operator = FilterOperator.EQ)

    return MetadataFilter(key = key, value = list(values), operator = FilterOperator.IN)


def build_filters(tags: Sequence[str]) -> MetadataFilters | None:
    """Собирает фильтр поиска из тегов.

    Аргументы:
        tags: теги вида "ключ:значение".

    Возвращает:
        Фильтр, где условия по разным полям соединены «и», либо None, если тегов
        нет. None означает поиск по всему корпусу.

    Исключения:
        TagFormatError: тег записан без разделителя между ключом и значением.
    """
    grouped = group_tags(tags)

    if not grouped:
        return None

    return MetadataFilters(
        filters = [build_field_filter(key = key, values = values) for key, values in grouped.items()],
        condition = FilterCondition.AND,
    )

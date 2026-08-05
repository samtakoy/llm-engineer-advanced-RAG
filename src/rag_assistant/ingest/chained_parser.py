"""Цепочка парсеров: выдача первого идёт на вход второму.

Стандартные парсеры llama-index делят обязанности: `MarkdownNodeParser` ставит границы
по заголовкам, `SentenceSplitter` соблюдает предел и не знает
про структуру.
Собранные в цепочку: границу ставит документ, длина остаётся ограничением.
"""
from typing import Any, List, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode
from pydantic import Field


class ChainedNodeParser(NodeParser):
    """Применяет парсеры по очереди, передавая узлы дальше по цепочке.

    Метаданные узла наследуются на каждом шаге, поэтому ссылка на страницу доходит
    до конца цепочки. Ссылка на исходный документ у итогового узла указывает
    на промежуточный узел предыдущего шага, а не на страницу.

    Атрибуты:
        parsers: парсеры в порядке применения.
    """

    parsers: List[NodeParser] = Field(description = "Парсеры в порядке применения.")

    @classmethod
    def class_name(cls) -> str:
        """Имя класса для сериализации llama-index.

        Возвращает:
            Устойчивое имя, по которому парсер восстанавливается из настроек.
        """
        return "ChainedNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Прогоняет узлы через все парсеры цепочки.

        Аргументы:
            nodes: документы либо узлы предыдущего шага.
            show_progress: показывать ход работы.
            **kwargs: прочие аргументы парсеров.

        Возвращает:
            Узлы после последнего парсера цепочки.
        """
        current = list(nodes)

        for parser in self.parsers:
            current = parser.get_nodes_from_documents(
                current,
                show_progress = show_progress,
            )

        return current

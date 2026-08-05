"""Склейка узла, в котором остался один заголовок.

`MarkdownNodeParser` ставит границу перед каждым заголовком, поэтому два заголовка
подряд — «Раздел 4» и следом «4.1.» — дают узел из одной строки. Искать по нему нечего,
а место в выдаче он занимает наравне с содержательным.

Заголовок приклеивается к следующему узлу: там лежит текст, к которому он относится.
"""
from typing import Any, List, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode

# Знак заголовка markdown.
HEADING_PREFIX = "#"


class HeadingMergeParser(NodeParser):
    """Приклеивает узлы из одних заголовков к следующему узлу той же страницы."""

    @classmethod
    def class_name(cls) -> str:
        """Имя класса для сериализации llama-index.

        Возвращает:
            Устойчивое имя, по которому парсер восстанавливается из настроек.
        """
        return "HeadingMergeParser"

    def is_heading_only(self, node: BaseNode) -> bool:
        """Решает, состоит ли узел из одних заголовков.

        Аргументы:
            node: узел после нарезки по заголовкам.

        Возвращает:
            True, если в узле нет ни одной строки, кроме заголовков и пустых.
        """
        lines = [line.strip() for line in node.get_content().splitlines() if line.strip()]

        return bool(lines) and all(line.startswith(HEADING_PREFIX) for line in lines)

    def same_page(self, node: BaseNode, other: BaseNode) -> bool:
        """Сравнивает страницы двух узлов.

        Аргументы:
            node: первый узел.
            other: второй узел.

        Возвращает:
            True, если оба узла с одной страницы одного файла.
        """
        keys = ("file_name", "page_label")

        return all(node.metadata.get(key) == other.metadata.get(key) for key in keys)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Склеивает заголовки с идущим за ними текстом.

        Аргументы:
            nodes: узлы после нарезки по заголовкам, в порядке чтения.
            show_progress: показывать ход работы, здесь не используется.
            **kwargs: прочие аргументы парсеров.

        Возвращает:
            Узлы, в которых заголовок стоит вместе со своим текстом. Заголовок
            в конце страницы остаётся отдельным узлом: приклеить его к тексту
            следующей страницы значило бы соврать в номере страницы.
        """
        merged: List[BaseNode] = []
        headings: List[str] = []

        for index, node in enumerate(nodes):
            following = nodes[index + 1] if index + 1 < len(nodes) else None

            if (
                self.is_heading_only(node)
                and following is not None
                and self.same_page(node = node, other = following)
            ):
                headings.append(node.get_content())
                continue

            if headings:
                node.set_content("\n\n".join([*headings, node.get_content()]))
                headings = []

            merged.append(node)

        return merged

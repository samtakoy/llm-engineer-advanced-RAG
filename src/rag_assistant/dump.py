"""Выгрузка нарезки в markdown: по файлу на документ, узлы в порядке чтения.

Показывает то, что уйдёт в индекс, до векторизации: границы узлов и метаданные
каждого из них.
"""
from pathlib import Path
from typing import Dict, List, Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode


def write_corpus(
    documents: Sequence[Document],
    node_parser: SentenceSplitter,
    target_dir: Path,
) -> Dict[str, int]:
    """Режет документы и записывает нарезку в markdown, по файлу на документ.

    Аргументы:
        documents: документы, прочитанные из папки.
        node_parser: сплиттер, тот же, что режет при индексации.
        target_dir: папка, в которую пишутся файлы.

    Возвращает:
        Число узлов по каждому исходному файлу.
    """
    target_dir.mkdir(parents = True, exist_ok = True)

    nodes_by_source = group_nodes_by_source(
        node_parser.get_nodes_from_documents(list(documents)),
    )
    written = {}

    for source, nodes in nodes_by_source.items():
        target_path = target_dir / f"{Path(source).stem}.md"
        target_path.write_text(render_source(source = source, nodes = nodes), encoding = "utf-8")
        written[source] = len(nodes)

    return written


def group_nodes_by_source(nodes: Sequence[BaseNode]) -> Dict[str, List[BaseNode]]:
    """Группирует узлы по исходному файлу и раскладывает в порядке чтения.

    Аргументы:
        nodes: узлы всех документов.

    Возвращает:
        Узлы по имени исходного файла, внутри — по возрастанию страницы
        и позиции в тексте страницы.
    """
    grouped: Dict[str, List[BaseNode]] = {}

    for node in nodes:
        source = node.metadata.get("file_name", "документ")
        grouped.setdefault(source, []).append(node)

    for source_nodes in grouped.values():
        source_nodes.sort(key = reading_order_key)

    return grouped


def reading_order_key(node: BaseNode) -> tuple:
    """Собирает ключ сортировки узла в порядке чтения документа.

    Аргументы:
        node: узел документа.

    Возвращает:
        Пару «номер страницы, позиция в тексте страницы». Номер страницы
        приводится к числу: как строка он дал бы порядок 1, 10, 11, 2.
    """
    page_label = str(node.metadata.get("page_label", "")).strip()
    page_number = int(page_label) if page_label.isdigit() else 0

    return page_number, node.start_char_idx or 0


def render_source(source: str, nodes: Sequence[BaseNode]) -> str:
    """Собирает markdown-файл одного документа.

    Аргументы:
        source: имя исходного файла.
        nodes: узлы этого файла в порядке чтения.

    Возвращает:
        Текст файла.
    """
    pages = {node.metadata.get("page_label") for node in nodes}
    characters = sum(len(node.text) for node in nodes)
    first = nodes[0]

    header = [
        f"# {source}",
        "",
        f"Узлов: {len(nodes)} · страниц: {len(pages)} · символов: {characters}",
        "",
        f"В эмбеддинг уходят метаданные: {', '.join(visible_metadata_keys(first, MetadataMode.EMBED))}",
        f"В промпт модели уходят: {', '.join(visible_metadata_keys(first, MetadataMode.LLM))}",
    ]
    blocks = [render_node(node = node, number = number) for number, node in enumerate(nodes, 1)]

    return "\n".join(header) + "\n\n" + "\n\n".join(blocks) + "\n"


def visible_metadata_keys(node: BaseNode, metadata_mode: MetadataMode) -> List[str]:
    """Определяет, какие метаданные видны в выбранном режиме.

    Аргументы:
        node: узел документа.
        metadata_mode: режим видимости метаданных.

    Возвращает:
        Имена полей, попадающих в текст узла в этом режиме.
    """
    excluded = {
        MetadataMode.EMBED: node.excluded_embed_metadata_keys,
        MetadataMode.LLM: node.excluded_llm_metadata_keys,
    }.get(metadata_mode, [])

    return [key for key in node.metadata if key not in excluded]


def render_node(node: BaseNode, number: int) -> str:
    """Собирает раздел одного узла.

    Аргументы:
        node: узел документа.
        number: порядковый номер узла в документе.

    Возвращает:
        Заголовок узла, все его метаданные и текст.
    """
    page_label = node.metadata.get("page_label", "—")
    metadata_lines = [f"- {key}: {value}" for key, value in node.metadata.items()]

    return "\n".join([
        f"## Узел {number} · стр. {page_label} · {len(node.text)} симв.",
        "",
        *metadata_lines,
        "",
        "```text",
        node.text,
        "```",
    ])

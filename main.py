"""Точка входа проекта.

Команды:
    uv run main.py              — поднять веб-чат;
    uv run main.py reindex      — перестроить индекс с нуля и выйти;
    uv run main.py ask "вопрос" — один вопрос в терминале;
    uv run main.py parse        — выгрузить нарезку в parsed/ для просмотра.

Поиск можно ограничить разметкой документов:
    uv run main.py ask "вопрос" --tag "year:2025"
"""
import argparse
import sys
from functools import partial

from llama_index.core import VectorStoreIndex

from rag_assistant.config import AppConfig
from rag_assistant.dump import write_corpus
from rag_assistant.engine import RagEngine
from rag_assistant.index import open_index
from rag_assistant.index_signature import IndexSettingsChanged
from rag_assistant.ingest import load_documents
from rag_assistant.metadata_filters import TagFormatError, build_filters, parse_tags
from rag_assistant.models import configure_global_settings, create_node_parser, create_reranker
from rag_assistant.ui import build_app


def prepare_index(config: AppConfig, rebuild: bool) -> VectorStoreIndex:
    """Поднимает нижнюю часть пайплайна: модели, документы, индекс.

    Аргументы:
        config: конфигурация приложения.
        rebuild: True — перестроить индекс с нуля.

    Возвращает:
        Индекс, готовый отдавать движок запросов.
    """
    configure_global_settings(config)

    return open_index(
        config = config,
        load_documents = partial(load_documents, config = config),
        rebuild = rebuild,
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Возвращает:
        Аргументы с полями command и question.
    """
    parser = argparse.ArgumentParser(description = "RAG-ассистент по годовым отчётам")
    parser.add_argument(
        "command",
        nargs = "?",
        choices = ["ui", "ask", "reindex", "parse"],
        default = "ui",
        help = "ui — веб-чат (по умолчанию), ask — один вопрос в терминале, "
               "reindex — перестроить индекс с нуля и выйти, "
               "parse — выгрузить нарезку в parsed/",
    )
    parser.add_argument("question", nargs = "?", default = None, help = "вопрос для команды ask")
    parser.add_argument(
        "--tag",
        default = None,
        help = 'фильтр по метаданным документов: "year:2024" — один отчёт, '
               '"year:2024|year:2025" — любой из двух. Разные поля сужают выдачу '
               'каждое своим условием',
    )

    arguments = parser.parse_args()

    if arguments.command == "ask" and not arguments.question:
        parser.error('команде ask нужен вопрос: uv run main.py ask "текст вопроса"')

    return arguments


def main() -> None:
    """Запускает выбранный сценарий.

    Возвращает:
        None.
    """
    arguments = parse_arguments()
    config = AppConfig.from_env()

    if arguments.command == "parse":
        written = write_corpus(
            documents = load_documents(config),
            node_parser = create_node_parser(config),
            target_dir = config.parsed_dir,
        )
        print(f"Записано {sum(written.values())} узлов в {config.parsed_dir}")
        for source, count in sorted(written.items()):
            print(f"  {source:<40} {count:>5}")
        return

    try:
        filters = build_filters(parse_tags(arguments.tag))
        index = prepare_index(config = config, rebuild = arguments.command == "reindex")
    except (TagFormatError, IndexSettingsChanged) as mismatch:
        # Ожидаемая ситуация, а не сбой: показываем причину без стека вызовов.
        print(mismatch, file = sys.stderr)
        raise SystemExit(1)

    # Движок собирается только под запросы: команде reindex он не нужен.
    if arguments.command == "reindex":
        return

    engine = RagEngine(
        index = index,
        config = config,
        filters = filters,
        reranker = create_reranker(config),
    )

    if arguments.command == "ask":
        answer = engine.ask(arguments.question)
        print(answer.text)
        print("\nИсточники:")
        for source in answer.sources:
            print(f"  {source.citation}  (score {source.score:.3f})")
        return

    build_app(engine).launch()


if __name__ == "__main__":
    main()

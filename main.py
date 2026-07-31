"""Точка входа проекта.

Команды:
    uv run main.py              — поднять веб-чат;
    uv run main.py --reindex    — перестроить индекс и поднять чат;
    uv run main.py ask "вопрос" — один вопрос в терминале;
    uv run main.py parse        — выгрузить нарезку в parsed/ для просмотра.
"""
import argparse
import sys
from functools import partial

from rag_assistant.config import AppConfig
from rag_assistant.dump import write_corpus
from rag_assistant.engine import RagEngine
from rag_assistant.index import open_index
from rag_assistant.index_signature import IndexSettingsChanged
from rag_assistant.ingest import load_documents
from rag_assistant.models import configure_global_settings, create_node_parser
from rag_assistant.ui import build_app


def create_engine(config: AppConfig, reindex: bool) -> RagEngine:
    """Поднимает пайплайн: модели, документы, индекс, движок запросов.

    Аргументы:
        config: конфигурация приложения.
        reindex: True — перестроить индекс с нуля.

    Возвращает:
        Готовый к запросам движок.
    """
    configure_global_settings(config)
    index = open_index(
        config = config,
        load_documents = partial(load_documents, config = config),
        rebuild = reindex,
    )

    return RagEngine(index = index, config = config)


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Возвращает:
        Аргументы с полями command, question, reindex.
    """
    parser = argparse.ArgumentParser(description = "RAG-ассистент по годовым отчётам")
    parser.add_argument(
        "command",
        nargs = "?",
        choices = ["ui", "ask", "parse"],
        default = "ui",
        help = "ui — веб-чат (по умолчанию), ask — один вопрос в терминале, "
               "parse — выгрузить нарезку в parsed/",
    )
    parser.add_argument("question", nargs = "?", default = None, help = "вопрос для команды ask")
    parser.add_argument("--reindex", action = "store_true", help = "перестроить индекс с нуля")

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
        engine = create_engine(config = config, reindex = arguments.reindex)
    except IndexSettingsChanged as mismatch:
        # Ожидаемая ситуация, а не сбой: показываем причину без стека вызовов.
        print(mismatch, file = sys.stderr)
        raise SystemExit(1)

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

"""Точка входа проекта.

Команды:
    uv run main.py ask "вопрос" — один вопрос в терминале;
    uv run main.py reindex      — перестроить индекс с нуля и выйти;
    uv run main.py parse        — выгрузить в parsed/ документы после чистки и нарезку.

Поиск можно ограничить разметкой документов:
    uv run main.py ask "вопрос" --tag "year:2025"

Техники поиска включаются флагами, как на стенде: --rerank, --hybrid, --mmr.
Флаг только включает технику, настройка берётся из .env: RERANK_MODEL,
MMR_THRESHOLD.
"""
import argparse
import sys
from functools import partial
from typing import List

from llama_index.core import VectorStoreIndex

from rag_assistant.config import AppConfig
from rag_assistant.dump import write_cleaned_documents, write_corpus
from rag_assistant.engine import RagEngine, create_retriever, retrieval_top_k
from rag_assistant.index import open_index
from rag_assistant.index_signature import IndexSettingsChanged
from rag_assistant.ingest import load_documents
from rag_assistant.lexical import create_lexical_index
from rag_assistant.metadata_filters import TagFormatError, build_filters, parse_tags
from rag_assistant.models import configure_global_settings, create_node_parser, create_reranker


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
        Аргументы с полями command, question, tag и флагами техник поиска.
    """
    parser = argparse.ArgumentParser(description = "RAG-ассистент по годовым отчётам")
    parser.add_argument(
        "command",
        choices = ["ask", "reindex", "parse"],
        help = "ask — один вопрос в терминале, "
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
    parser.add_argument(
        "--rerank",
        action = "store_true",
        help = "переставлять найденные фрагменты cross-encoder из RERANK_MODEL",
    )
    parser.add_argument(
        "--hybrid",
        action = "store_true",
        help = "искать векторно и по словам сразу, объединяя выдачи",
    )
    parser.add_argument(
        "--mmr",
        action = "store_true",
        help = "разбавлять выдачу непохожими фрагментами, вес из MMR_THRESHOLD",
    )

    arguments = parser.parse_args()

    if arguments.command == "ask" and not arguments.question:
        parser.error('команде ask нужен вопрос: uv run main.py ask "текст вопроса"')

    # Техники поиска работают на запросе, а reindex и parse запроса не делают.
    # Молча принятый флаг создал бы впечатление, что индекс собран как-то иначе.
    if arguments.command != "ask" and enabled_modes(arguments):
        parser.error(
            f"флаги {', '.join('--' + name for name in enabled_modes(arguments))} "
            f"относятся к поиску и работают только с командой ask",
        )

    return arguments


def enabled_modes(arguments: argparse.Namespace) -> List[str]:
    """Перечисляет техники поиска, включённые флагами.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Имена включённых техник в порядке применения.
    """
    return [name for name in ("rerank", "hybrid", "mmr") if getattr(arguments, name)]


def main() -> None:
    """Запускает выбранный сценарий.

    Возвращает:
        None.
    """
    arguments = parse_arguments()
    config = AppConfig.from_env()

    if arguments.command == "parse":
        documents = load_documents(config)
        pages = write_cleaned_documents(documents = documents, target_dir = config.parsed_dir)
        written = write_corpus(
            documents = documents,
            node_parser = create_node_parser(config),
            target_dir = config.parsed_dir,
        )
        print(f"Записано {sum(pages.values())} страниц после чистки и {sum(written.values())} узлов в {config.parsed_dir}")
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

    # Движок собирается только под запросы: сборке индекса он не нужен.
    if arguments.command == "reindex":
        return

    reranker = create_reranker(config) if arguments.rerank else None
    engine = RagEngine(
        retriever = create_retriever(
            index = index,
            filters = filters,
            top_k = retrieval_top_k(
                config = config,
                reranker = reranker,
            ),
            lexical_index = create_lexical_index(index) if arguments.hybrid else None,
            mmr_threshold = config.mmr_threshold if arguments.mmr else None,
        ),
        reranker = reranker,
    )

    if arguments.command == "ask":
        answer = engine.ask(arguments.question)
        print(answer.text)
        print("\nИсточники:")
        for source in answer.sources:
            print(f"  {source.citation}  (score {source.score:.3f})")
        return


if __name__ == "__main__":
    main()

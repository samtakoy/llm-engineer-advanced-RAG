"""Прогон контрольных вопросов и снятие метрик.

Команды:
    uv run python -m eval.run retrieval    — только поиск, без обращений к модели;
    uv run python -m eval.run full         — поиск, ответы и разбор ссылок;
    uv run python -m eval.run full --judge — то же плюс оценка ответов моделью.

Режим retrieval быстрый, потому что модель не вызывается: на нём и подбираются настройки
поиска. Полный прогон нужен для отчёта и для метрик, которые считаются по тексту ответа.
"""
import argparse
import sys
from functools import partial
from typing import List, Set

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM

from eval.cases import QUESTIONS_PATH, Case, Page, load_cases
from eval.judge import judge_answer
from eval.metrics import measure, summarize
from eval.report import print_summary, print_table, save_report
from eval.results import CaseRun, to_pages
from rag_assistant.config import AppConfig
from rag_assistant.engine import RagEngine, Source, to_source
from rag_assistant.index import find_collection, open_index
from rag_assistant.index_signature import IndexSettingsChanged
from rag_assistant.ingest import load_documents
from rag_assistant.models import configure_global_settings, create_llm


def load_known_pages(config: AppConfig) -> Set[Page]:
    """Собирает все страницы, лежащие в индексе.

    Нужны, чтобы отличить ссылку на выдуманную страницу от ссылки на существующую,
    но не попавшую в контекст.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Страницы всех проиндексированных узлов.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))
    collection = find_collection(client = client, name = config.chroma_collection)
    records = collection.get(include = ["metadatas"])

    return {
        Page(document = metadata["file_name"], number = int(metadata["page_label"]))
        for metadata in records["metadatas"]
        if str(metadata.get("page_label", "")).isdigit()
    }


def retrieve_sources(index: VectorStoreIndex, config: AppConfig, question: str) -> List[Source]:
    """Возвращает фрагменты по вопросу, не обращаясь к модели.

    Аргументы:
        index: векторный индекс документов.
        config: конфигурация приложения.
        question: вопрос.

    Возвращает:
        Фрагменты в порядке выдачи поиска.
    """
    retriever = index.as_retriever(similarity_top_k = config.top_k)

    return [to_source(node) for node in retriever.retrieve(question)]


def run_case(
    case: Case,
    index: VectorStoreIndex,
    config: AppConfig,
    known_pages: Set[Page],
    engine: RagEngine | None,
    judge: LLM | None,
) -> CaseRun:
    """Прогоняет один вопрос.

    Аргументы:
        case: контрольный вопрос.
        index: векторный индекс документов.
        config: конфигурация приложения.
        known_pages: все страницы, лежащие в индексе.
        engine: движок вопрос-ответ либо None, если ответы не нужны.
        judge: модель-судья либо None, если оценка не нужна.

    Возвращает:
        Результат прогона вопроса.
    """
    if engine is None:
        sources = retrieve_sources(index = index, config = config, question = case.question)
        answer = ""
    else:
        response = engine.ask(case.question)
        sources = response.sources
        answer = response.text

    metrics = measure(
        case = case,
        retrieved = to_pages(sources),
        context = " ".join(source.text for source in sources),
        answer = answer,
        known_pages = known_pages,
    )
    # На вопрос-провокацию судью не зовём: его эталон — сам отказ, раскладывать такой
    # эталон на факты бессмысленно, а верность отказа уже меряет refusal_rate.
    needs_judge = judge is not None and answer and not case.expected_refusal
    verdict = (
        judge_answer(
            llm = judge,
            case = case,
            answer = answer,
            contexts = [source.text for source in sources],
        )
        if needs_judge
        else None
    )

    return CaseRun(
        case = case,
        sources = sources,
        answer = answer,
        metrics = metrics,
        verdict = verdict,
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Возвращает:
        Аргументы с полями mode, judge, name.
    """
    parser = argparse.ArgumentParser(description = "Прогон контрольных вопросов")
    parser.add_argument(
        "mode",
        nargs = "?",
        choices = ["retrieval", "full"],
        default = "retrieval",
        help = "retrieval — только поиск (по умолчанию), full — с ответами модели",
    )
    parser.add_argument("--judge", action = "store_true", help = "оценить ответы моделью")
    parser.add_argument("--name", default = "baseline", help = "имя снимка в docs/eval")

    arguments = parser.parse_args()

    if arguments.judge and arguments.mode != "full":
        parser.error("судья оценивает ответы, поэтому нужен режим full")

    return arguments


def main() -> None:
    """Прогоняет набор вопросов и сохраняет отчёт.

    Возвращает:
        None.
    """
    arguments = parse_arguments()
    config = AppConfig.from_env()
    configure_global_settings(config)

    try:
        index = open_index(
            config = config,
            load_documents = partial(load_documents, config = config),
            rebuild = False,
        )
    except IndexSettingsChanged as mismatch:
        print(mismatch, file = sys.stderr)
        raise SystemExit(1)

    cases = load_cases(QUESTIONS_PATH)
    known_pages = load_known_pages(config)
    engine = RagEngine(index = index, config = config) if arguments.mode == "full" else None
    judge = create_llm(config) if arguments.judge else None

    runs = []
    for case in cases:
        print(f"[{case.number:>2}/{len(cases)}] {case.question[:60]}")
        runs.append(
            run_case(
                case = case,
                index = index,
                config = config,
                known_pages = known_pages,
                engine = engine,
                judge = judge,
            )
        )

    summary = summarize(
        cases,
        [run.metrics for run in runs],
        [run.verdict for run in runs],
    )
    print_table(runs)
    print_summary(summary, runs)
    save_report(runs = runs, summary = summary, config = config, name = arguments.name)


if __name__ == "__main__":
    main()

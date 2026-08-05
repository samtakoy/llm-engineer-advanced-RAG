"""Прогон контрольных вопросов и снятие метрик.

Команды:
    uv run python -m eval.run retrieval    — только поиск, без обращений к модели;
    uv run python -m eval.run full         — поиск, ответы и разбор ссылок;
    uv run python -m eval.run full --judge — то же плюс оценка ответов моделью.

Флаг --filters ограничивает поиск разметкой документов из поля tags у вопроса,
--rerank переставляет найденное cross-encoder, --hybrid добавляет к векторному
поиску лексический, --mmr разбавляет выдачу непохожими фрагментами, --mmr2 делает
то же самое отбором из оценённых реранкером кандидатов. Прогон с флагом и без него
под разными именами (--name) и даёт сравнение.

Режим retrieval быстрый, потому что модель не вызывается: на нём и подбираются настройки
поиска. Полный прогон нужен для отчёта и для метрик, которые считаются по тексту ответа.
"""
import argparse
import sys
import time
from functools import partial
from typing import List, Set

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from eval.cases import QUESTIONS_PATH, Case, Page, load_cases
from eval.judge import judge_answer
from eval.metrics import measure, summarize
from eval.report import SnapshotBelongsToOtherSettings, print_summary, print_table, save_report
from eval.results import CaseRun, page_range, to_page_groups
from rag_assistant.config import AppConfig
from rag_assistant.diversity import (
    DiversityWeightMissing,
    create_diversity_postprocessor,
    resolve_mmr_threshold,
)
from rag_assistant.engine import (
    RagEngine,
    Source,
    create_retriever,
    rerank_top_n,
    retrieval_top_k,
    to_source,
)
from rag_assistant.index import find_collection, open_index
from rag_assistant.index_signature import IndexSettingsChanged
from rag_assistant.ingest import load_documents
from rag_assistant.lexical import LexicalIndex, create_lexical_index
from rag_assistant.metadata_filters import build_filters
from rag_assistant.models import configure_global_settings, create_llm, create_reranker


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
        Page(document = metadata["file_name"], number = number)
        for metadata in records["metadatas"]
        if str(metadata.get("page_label", "")).isdigit()
        for number in page_range(
            first = str(metadata["page_label"]),
            last = str(metadata.get("page_end", "")) or None,
        )
    }


def retrieve_sources(
    retriever: BaseRetriever,
    question: str,
    reranker: SentenceTransformerRerank | None,
    diversity: BaseNodePostprocessor | None,
) -> List[Source]:
    """Возвращает фрагменты по вопросу, не обращаясь к модели.

    Аргументы:
        retriever: поиск, отдающий фрагменты по вопросу.
        question: вопрос.
        reranker: реранкер либо None, если порядок выдачи остаётся за поиском.
        diversity: отбор по разнообразию либо None, если выдача идёт как есть.

    Возвращает:
        Фрагменты в том порядке, в котором их увидела бы модель.
    """
    nodes = retriever.retrieve(question)

    # Режим retrieval идёт мимо движка запросов, а постпроцессоры применяет он.
    # Без этого вызова режимы retrieval и full мерили бы разные пайплайны.
    for postprocessor in (reranker, diversity):
        if postprocessor is not None:
            nodes = postprocessor.postprocess_nodes(nodes, query_str = question)

    return [to_source(node) for node in nodes]


def run_case(
    case: Case,
    index: VectorStoreIndex,
    config: AppConfig,
    known_pages: Set[Page],
    filters: MetadataFilters | None,
    reranker: SentenceTransformerRerank | None,
    diversity: BaseNodePostprocessor | None,
    lexical_index: LexicalIndex | None,
    mmr_threshold: float | None,
    with_answer: bool,
    judge: LLM | None,
) -> CaseRun:
    """Прогоняет один вопрос.

    Аргументы:
        case: контрольный вопрос.
        index: векторный индекс документов.
        config: конфигурация приложения.
        known_pages: все страницы, лежащие в индексе.
        filters: отбор документов по метаданным либо None — искать по всему корпусу.
        reranker: реранкер либо None, если порядок выдачи остаётся за поиском.
        diversity: отбор по разнообразию после реранкера либо None.
        lexical_index: индекс поиска по словам либо None — искать только векторно.
        mmr_threshold: вес близости против разнообразия либо None — только близость.
        with_answer: True — спросить модель, False — снять только выдачу поиска.
        judge: модель-судья либо None, если оценка не нужна.

    Возвращает:
        Результат прогона вопроса.
    """
    # Замеряется работа пайплайна: поиск и ответ модели.
    # Судья остаётся снаружи — он часть замера, а не того, что меряют.
    started = time.perf_counter()

    retriever = create_retriever(
        index = index,
        filters = filters,
        top_k = retrieval_top_k(
            config = config,
            reranker = reranker,
            diversity = diversity is not None,
        ),
        lexical_index = lexical_index,
        mmr_threshold = mmr_threshold,
    )

    if with_answer:
        response = RagEngine(
            retriever = retriever,
            reranker = reranker,
            diversity = diversity,
        ).ask(case.question)
        sources = response.sources
        answer = response.text
    else:
        sources = retrieve_sources(
            retriever = retriever,
            question = case.question,
            reranker = reranker,
            diversity = diversity,
        )
        answer = ""

    metrics = measure(
        case = case,
        retrieved_groups = to_page_groups(sources),
        context = " ".join(source.text for source in sources),
        answer = answer,
        known_pages = known_pages,
        seconds = time.perf_counter() - started,
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
    parser.add_argument(
        "--filters",
        action = "store_true",
        help = "ограничивать поиск разметкой документов из поля tags у вопроса",
    )
    parser.add_argument(
        "--rerank",
        action = "store_true",
        help = "переставлять найденные фрагменты cross-encoder из RERANK_MODEL",
    )
    parser.add_argument(
        "--mmr",
        action = "store_true",
        help = "разбавлять выдачу непохожими фрагментами, вес из MMR_THRESHOLD",
    )
    parser.add_argument(
        "--mmr2",
        action = "store_true",
        help = "то же разнообразие, но отбором из оценённых реранкером кандидатов",
    )
    parser.add_argument(
        "--hybrid",
        action = "store_true",
        help = "искать векторно и по словам сразу, объединяя выдачи",
    )
    parser.add_argument("--name", default = "baseline", help = "имя снимка в docs/eval")

    arguments = parser.parse_args()

    if arguments.judge and arguments.mode != "full":
        parser.error("судья оценивает ответы, поэтому нужен режим full")

    # Обе техники решают одну задачу на разных шагах пайплайна. Включённые вместе,
    # они дважды жертвуют релевантностью ради разнообразия, и вклад каждой не разделить.
    if arguments.mmr and arguments.mmr2:
        parser.error("--mmr и --mmr2 — две реализации одного приёма, включается одна")

    return arguments


def enabled_modes(arguments: argparse.Namespace) -> List[str]:
    """Перечисляет техники поиска, включённые в прогоне.

    Идут в настройки снимка: без них прогоны с разными флагами неразличимы,
    и защита от затирания чужого снимка пропускает подмену.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Имена включённых техник в порядке применения.
    """
    return [
        name
        for name in ("filters", "rerank", "mmr", "mmr2", "hybrid")
        if getattr(arguments, name)
    ]


def main() -> None:
    """Прогоняет набор вопросов и сохраняет отчёт.

    Возвращает:
        None.
    """
    arguments = parse_arguments()
    config = AppConfig.from_env()
    configure_global_settings(config)

    try:
        diversity_threshold = resolve_mmr_threshold(config) if arguments.mmr2 else None
        index = open_index(
            config = config,
            load_documents = partial(load_documents, config = config),
            rebuild = False,
        )
    except (IndexSettingsChanged, DiversityWeightMissing) as mismatch:
        print(mismatch, file = sys.stderr)
        raise SystemExit(1)

    cases = load_cases(QUESTIONS_PATH)
    known_pages = load_known_pages(config)
    # Судья отвечает размеченным текстом, который разбирается вручную: схема ему не нужна.
    judge = (
        create_llm(config = config, model = config.judge_model, schema_constrained = False)
        if arguments.judge
        else None
    )
    reranker = (
        create_reranker(
            config = config,
            top_n = rerank_top_n(config = config, diversity = arguments.mmr2),
        )
        if arguments.rerank
        else None
    )
    diversity = (
        create_diversity_postprocessor(
            config = config,
            mmr_threshold = diversity_threshold,
        )
        if diversity_threshold is not None
        else None
    )
    lexical_index = create_lexical_index(index) if arguments.hybrid else None

    # Список техник собирается до прогона.
    modes = enabled_modes(arguments)

    runs = []
    for case in cases:
        print(f"[{case.number:>2}/{len(cases)}] {case.question[:60]}")
        filters = build_filters(case.tags) if arguments.filters else None
        runs.append(
            run_case(
                case = case,
                index = index,
                config = config,
                known_pages = known_pages,
                filters = filters,
                reranker = reranker,
                diversity = diversity,
                lexical_index = lexical_index,
                mmr_threshold = config.mmr_threshold if arguments.mmr else None,
                with_answer = arguments.mode == "full",
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
    try:
        save_report(
            runs = runs,
            summary = summary,
            config = config,
            name = arguments.name,
            modes = modes,
        )
    except SnapshotBelongsToOtherSettings as conflict:
        # Прогон уже сделан, терять его из-за занятого имени нельзя: показываем причину,
        # результаты выше в терминале остаются.
        print(conflict, file = sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

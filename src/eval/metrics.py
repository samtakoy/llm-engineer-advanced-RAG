"""Детерминированные метрики прогона: попадание в эталон и честность ссылок.

Модели здесь не участвуют, поэтому метрики считаются мгновенно и от прогона к прогону
не плавают. На них и сравниваются техники поиска.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from eval.cases import Case, Page
from eval.judge import Verdict, correctness, faithfulness

# Содержимое квадратных скобок в ответе: там лежат одна или несколько ссылок.
CITATION_BLOCK_PATTERN = re.compile(r"\[([^\[\]]+?)\]")

# Одна ссылка: «файл.pdf, стр. 8». Модель складывает несколько в одни скобки через «;»,
# поэтому разбирать надо каждую ссылку, а не содержимое скобок целиком.
CITATION_PATTERN = re.compile(r"\s*(?P<document>[^;]+?),\s*стр\.\s*(?P<page>\d+)\s*")

REFUSAL_MARKER = "не знаю"


@dataclass(frozen = True)
class Measure:
    """Итоговая метрика вместе с тем, из чего она сложилась.

    Одно число вида 0.800 не говорит, где искать причину. Счёт и номера вопросов,
    которые не зачлись, дают адрес: по нему открывают отчёт и смотрят.

    Атрибуты:
        value: значение метрики от нуля до единицы.
        scored: сколько засчиталось, может быть дробным для дробных оценок.
        total: сколько единиц участвовало в подсчёте.
        unit: что считалось — вопросы или ссылки.
        failed: номера вопросов, зачтённых не полностью.
    """

    value: float
    scored: float
    total: int
    unit: str
    failed: List[int]


@dataclass(frozen = True)
class CaseMetrics:
    """Метрики одного вопроса.

    Атрибуты:
        page_hit: True — выдача покрыла все группы эталонных страниц.
        page_hit_at_3: то же по первым трём фрагментам выдачи. Более жёсткая планка:
            попадание в топ-5 упирается в потолок раньше, чем поиск перестаёт улучшаться.
        covered_groups: сколько групп эталона покрыто.
        total_groups: сколько групп эталона всего.
        reciprocal_rank: единица, делённая на место первой эталонной страницы в выдаче.
        unique_pages: сколько разных страниц в выдаче, мера её избыточности.
        snippets_found: сколько эталонных подстрок дошло до модели в тексте выдачи.
        snippets_total: сколько эталонных подстрок у вопроса всего.
        citations: сколько ссылок модель поставила в ответе.
        invented_citations: ссылки на страницы, которых в корпусе нет.
        outside_context_citations: ссылки на существующие страницы, которых не было
            в переданном модели контексте.
        refused: True — модель ответила «не знаю», None — ответа не было.
    """

    page_hit: bool
    page_hit_at_3: bool
    covered_groups: int
    total_groups: int
    reciprocal_rank: float
    unique_pages: int
    snippets_found: int
    snippets_total: int
    citations: int
    invented_citations: int
    outside_context_citations: int
    refused: bool | None


def count_covered_groups(retrieved: Sequence[Page], expected_pages: Sequence[Sequence[Page]]) -> int:
    """Считает, сколько групп эталонных страниц закрыла выдача.

    Аргументы:
        retrieved: страницы, которые вернул поиск.
        expected_pages: группы эталонных страниц.

    Возвращает:
        Число групп, из которых нашлась хотя бы одна страница.
    """
    found = set(retrieved)

    return sum(1 for group in expected_pages if found & set(group))


def is_hit(retrieved: Sequence[Page], expected_pages: Sequence[Sequence[Page]]) -> bool:
    """Проверяет, покрыла ли выдача весь эталон.

    Аргументы:
        retrieved: страницы, которые вернул поиск.
        expected_pages: группы эталонных страниц.

    Возвращает:
        True, если из каждой группы нашлась хотя бы одна страница.
    """
    return count_covered_groups(retrieved, expected_pages) == len(expected_pages)


def reciprocal_rank(retrieved: Sequence[Page], expected_pages: Sequence[Sequence[Page]]) -> float:
    """Считает обратное место первой эталонной страницы в выдаче.

    Аргументы:
        retrieved: страницы в порядке, в котором их вернул поиск.
        expected_pages: группы эталонных страниц.

    Возвращает:
        1/место для первой эталонной страницы, ноль — если эталона в выдаче нет.
    """
    expected = {page for group in expected_pages for page in group}

    for place, page in enumerate(retrieved, start = 1):
        if page in expected:
            return 1 / place

    return 0.0


def find_citations(answer: str) -> List[Page]:
    """Вытаскивает из ответа ссылки на источники.

    Аргументы:
        answer: текст ответа модели.

    Возвращает:
        Страницы, на которые модель сослалась, в порядке появления. Несколько ссылок
        в одних скобках через «;» считаются по отдельности.
    """
    pages = []
    for block in CITATION_BLOCK_PATTERN.findall(answer):
        for reference in block.split(";"):
            citation = CITATION_PATTERN.fullmatch(reference)
            if citation:
                pages.append(Page(
                    document = citation.group("document").strip(),
                    number = int(citation.group("page")),
                ))

    return pages


def count_found_snippets(context: str, snippets: Sequence[str]) -> int:
    """Считает, сколько эталонных подстрок нашлось в тексте выдачи.

    Аргументы:
        context: склеенный текст всех найденных фрагментов.
        snippets: эталонные подстроки вопроса.

    Возвращает:
        Число найденных подстрок. Пробелы схлопываются: в разобранном PDF переносы
        строк стоят в неожиданных местах и мешают точному совпадению.
    """
    normalized = " ".join(context.split())

    return sum(1 for snippet in snippets if snippet in normalized)


def measure(
    case: Case,
    retrieved: Sequence[Page],
    context: str,
    answer: str,
    known_pages: Set[Page],
) -> CaseMetrics:
    """Считает метрики одного вопроса.

    Аргументы:
        case: контрольный вопрос с эталоном.
        retrieved: страницы, которые вернул поиск, в порядке выдачи.
        context: склеенный текст найденных фрагментов.
        answer: текст ответа модели, пустой в режиме проверки только поиска.
        known_pages: все страницы, лежащие в индексе.

    Возвращает:
        Метрики вопроса.
    """
    citations = find_citations(answer)
    pages_in_context = set(retrieved)

    return CaseMetrics(
        page_hit = is_hit(retrieved, case.expected_pages),
        page_hit_at_3 = is_hit(retrieved[:3], case.expected_pages),
        covered_groups = count_covered_groups(retrieved, case.expected_pages),
        total_groups = len(case.expected_pages),
        reciprocal_rank = reciprocal_rank(retrieved, case.expected_pages),
        unique_pages = len(pages_in_context),
        snippets_found = count_found_snippets(context, case.expected_snippets),
        snippets_total = len(case.expected_snippets),
        citations = len(citations),
        invented_citations = sum(1 for page in citations if page not in known_pages),
        outside_context_citations = sum(
            1 for page in citations if page in known_pages and page not in pages_in_context
        ),
        refused = REFUSAL_MARKER in answer.lower() if answer else None,
    )


def summarize(
    cases: Sequence[Case],
    measurements: Sequence[CaseMetrics],
    verdicts: Sequence[Verdict | None],
) -> Dict[str, Measure]:
    """Сводит метрики всех вопросов в итоговые.

    Аргументы:
        cases: контрольные вопросы.
        measurements: метрики по каждому вопросу в том же порядке.
        verdicts: разборы судьи по каждому вопросу, None — судья не вызывался.

    Возвращает:
        Итоговые метрики: попадание в эталонные страницы, средний обратный ранг,
        честность ссылок, отказы и оценки судьи. Уникальность страниц сюда не входит:
        она нужна для оценки эффекта MMR и видна по каждому вопросу в таблице.
    """
    answerable = [
        (case, metrics)
        for case, metrics in zip(cases, measurements)
        if not case.expected_refusal
    ]
    answered = [(case, metrics) for case, metrics in answerable if metrics.refused is not None]
    refusals = [
        (case, metrics)
        for case, metrics in zip(cases, measurements)
        if case.expected_refusal and metrics.refused is not None
    ]

    summary = {
        "page_hit_rate": collect(
            [(case.number, float(metrics.page_hit)) for case, metrics in answerable],
            unit = "вопросов",
        ),
        "page_hit_rate@3": collect(
            [(case.number, float(metrics.page_hit_at_3)) for case, metrics in answerable],
            unit = "вопросов",
        ),
        "mrr": collect(
            [(case.number, metrics.reciprocal_rank) for case, metrics in answerable],
            unit = "вопросов",
        ),
    }
    # Строже, чем page_hit_rate: страница может найтись, а нужный её фрагмент — нет.
    with_snippets = [
        (case.number, share(metrics.snippets_found, metrics.snippets_total))
        for case, metrics in answerable
        if metrics.snippets_total
    ]

    if with_snippets:
        summary["fact_hit_rate"] = collect(with_snippets, unit = "вопросов")

    citations = collect_citations(cases = cases, measurements = measurements)

    if citations is not None:
        summary["citation_honesty"] = citations
    # Считаем ответы, а не отказы: у всех метрик должно быть «больше — лучше»,
    # иначе список непрошедших вопросов перечисляет как раз прошедшие.
    if answered:
        summary["answered_rate"] = collect(
            [(case.number, float(not metrics.refused)) for case, metrics in answered],
            unit = "вопросов",
        )
    if refusals:
        summary["refusal_rate"] = collect(
            [(case.number, float(metrics.refused)) for case, metrics in refusals],
            unit = "вопросов",
        )

    summary.update(judge_summary(cases = cases, verdicts = verdicts))

    return summary


def collect(scores: Sequence[tuple], unit: str) -> Measure:
    """Сводит оценки по вопросам в одну метрику.

    Аргументы:
        scores: пары «номер вопроса, оценка от нуля до единицы».
        unit: что считается — вопросы или ссылки.

    Возвращает:
        Метрику со счётом и номерами вопросов, зачтённых не полностью.
    """
    scored = sum(score for _, score in scores)

    return Measure(
        value = share(scored, len(scores)),
        scored = scored,
        total = len(scores),
        unit = unit,
        failed = [number for number, score in scores if score < 1.0],
    )


def collect_citations(
    cases: Sequence[Case],
    measurements: Sequence[CaseMetrics],
) -> Measure | None:
    """Сводит честность ссылок по всем ответам.

    Считается по ссылкам, а не по вопросам: важно, какая доля всех проставленных
    ссылок указывает на существующую страницу, побывавшую в контексте.

    Аргументы:
        cases: контрольные вопросы.
        measurements: метрики по каждому вопросу.

    Возвращает:
        Метрику либо None, если ссылок не было вовсе.
    """
    total = sum(metrics.citations for metrics in measurements)

    if not total:
        return None

    dishonest = sum(
        metrics.invented_citations + metrics.outside_context_citations
        for metrics in measurements
    )

    return Measure(
        value = share(total - dishonest, total),
        scored = total - dishonest,
        total = total,
        unit = "ссылок",
        failed = [
            case.number
            for case, metrics in zip(cases, measurements)
            if metrics.invented_citations or metrics.outside_context_citations
        ],
    )


def judge_summary(
    cases: Sequence[Case],
    verdicts: Sequence[Verdict | None],
) -> Dict[str, Measure]:
    """Сводит оценки судьи по всем вопросам, где он вызывался.

    Аргументы:
        cases: контрольные вопросы.
        verdicts: разборы судьи, None — судья не вызывался.

    Возвращает:
        Средние faithfulness и correctness плюс долю разобранных ответов судьи:
        если судья часто отвечает неразбираемо, остальные его числа ничего не стоят.
    """
    judged = [(case, verdict) for case, verdict in zip(cases, verdicts) if verdict is not None]

    if not judged:
        return {}

    summary = {
        "judge_parsed": collect(
            [(case.number, float(not verdict.error)) for case, verdict in judged],
            unit = "вопросов",
        ),
    }

    for name, score_of in (("faithfulness", faithfulness), ("correctness", correctness)):
        scores = [
            (case.number, score_of(verdict))
            for case, verdict in judged
            if score_of(verdict) is not None
        ]
        if scores:
            summary[name] = collect(scores, unit = "вопросов")

    return summary


def share(value: float, total: int) -> float:
    """Делит с защитой от пустого набора.

    Аргументы:
        value: числитель.
        total: знаменатель.

    Возвращает:
        Частное, ноль при нулевом знаменателе.
    """
    return value / total if total else 0.0

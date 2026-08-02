"""Проверка метрик прогона контрольных вопросов."""
import json

import pytest

from eval.cases import QUESTIONS_PATH, Case, Page, load_cases, parse_page
from eval.judge import correctness, faithfulness, parse_verdict
from eval.report import SnapshotBelongsToOtherSettings, check_snapshot_free
from eval.metrics import (
    count_covered_groups,
    count_formatted_citations,
    count_malformed_citations,
    find_citations,
    is_hit,
    measure,
    reciprocal_rank,
    summarize,
)

FIRST = Page(document = "отчёт.pdf", number = 1)
SECOND = Page(document = "отчёт.pdf", number = 2)
THIRD = Page(document = "другой.pdf", number = 3)


def make_case(expected_pages: list, expected_refusal: bool) -> Case:
    """Собирает контрольный вопрос для проверки.

    Аргументы:
        expected_pages: группы эталонных страниц.
        expected_refusal: ожидается ли отказ.

    Возвращает:
        Вопрос с заполненными обязательными полями.
    """
    return Case(
        number = 1,
        kind = "факт",
        question = "вопрос",
        expected_answer = "эталон",
        expected_pages = expected_pages,
        expected_snippets = [],
        expected_refusal = expected_refusal,
        tags = [],
        checks = "",
        distractor = "",
    )


def test_single_group_needs_any_page() -> None:
    """Внутри группы достаточно одной страницы из перечисленных."""
    assert is_hit([SECOND], [[FIRST, SECOND]])


def test_several_groups_need_every_group() -> None:
    """Вопрос с агрегацией требует страницы из каждой группы."""
    groups = [[FIRST], [THIRD]]

    assert not is_hit([FIRST], groups)
    assert is_hit([FIRST, THIRD], groups)


def test_covered_groups_counts_partial_coverage() -> None:
    """Частичное покрытие видно по числу закрытых групп."""
    assert count_covered_groups([FIRST], [[FIRST], [THIRD]]) == 1


def test_reciprocal_rank_uses_first_match() -> None:
    """Обратный ранг считается по первой эталонной странице в выдаче."""
    assert reciprocal_rank([THIRD, SECOND, FIRST], [[FIRST]]) == 1 / 3


def test_reciprocal_rank_is_zero_without_match() -> None:
    """Эталона нет в выдаче — обратный ранг нулевой."""
    assert reciprocal_rank([THIRD], [[FIRST]]) == 0.0


def test_find_citations_reads_reference_format() -> None:
    """Ссылка разбирается в том виде, в каком её требует промпт."""
    answer = "Выручка выросла [отчёт.pdf, стр. 8] и EBITDA тоже [другой.pdf, стр. 3]."

    assert find_citations(answer) == [Page("отчёт.pdf", 8), Page("другой.pdf", 3)]


def test_find_citations_splits_combined_reference() -> None:
    """Модель складывает несколько ссылок в одни скобки — считаем их по отдельности."""
    answer = "Проекты [отчёт.pdf, стр. 35; отчёт.pdf, стр. 14] развиваются."

    assert find_citations(answer) == [Page("отчёт.pdf", 35), Page("отчёт.pdf", 14)]


def test_citation_without_page_marker_is_malformed() -> None:
    """Слабая модель пишет «[отчёт.pdf, 10]» без «стр.»: формат нарушен, но страница
    из ссылки понятна и на выдуманность проверяется наравне с остальными."""
    answer = "EBITDA составила 118,5 млн евро [отчёт.pdf, 10]."

    assert find_citations(answer) == [Page("отчёт.pdf", 10)]
    assert count_formatted_citations(answer) == 0
    assert count_malformed_citations(answer) == 1


def test_correct_citation_is_not_malformed() -> None:
    """Ссылка в требуемом формате в нарушения не попадает."""
    answer = "Выручка выросла [отчёт.pdf, стр. 8]."

    assert count_malformed_citations(answer) == 0


def test_find_citations_ignores_plain_text() -> None:
    """Текст без ссылок не даёт ложных срабатываний."""
    assert find_citations("Выручка выросла на 14% в 2025 году.") == []


def test_invented_citation_is_counted() -> None:
    """Ссылка на страницу, которой нет в корпусе, — выдуманная."""
    metrics = measure(
        case = make_case(expected_pages = [[FIRST]], expected_refusal = False),
        retrieved = [FIRST],
        context = "",
        answer = "Факт [отчёт.pdf, стр. 99].",
        known_pages = {FIRST, SECOND},
        seconds = 0.0,
    )

    assert metrics.invented_citations == 1
    assert metrics.outside_context_citations == 0


def test_citation_outside_context_is_counted() -> None:
    """Страница существует, но в контекст не попадала — тоже промах со ссылкой."""
    metrics = measure(
        case = make_case(expected_pages = [[FIRST]], expected_refusal = False),
        retrieved = [FIRST],
        context = "",
        answer = "Факт [отчёт.pdf, стр. 2].",
        known_pages = {FIRST, SECOND},
        seconds = 0.0,
    )

    assert metrics.invented_citations == 0
    assert metrics.outside_context_citations == 1


def test_refusal_is_undefined_without_answer() -> None:
    """В режиме проверки только поиска отказ не измеряется."""
    metrics = measure(
        case = make_case(expected_pages = [], expected_refusal = True),
        retrieved = [FIRST],
        context = "",
        answer = "",
        known_pages = {FIRST},
        seconds = 0.0,
    )

    assert metrics.refused is None


def test_refusal_is_detected_in_answer() -> None:
    """Отказ опознаётся по формулировке из промпта."""
    metrics = measure(
        case = make_case(expected_pages = [], expected_refusal = True),
        retrieved = [FIRST],
        context = "",
        answer = "Я не знаю",
        known_pages = {FIRST},
        seconds = 0.0,
    )

    assert metrics.refused


def test_summary_excludes_refusal_case_from_hit_rate() -> None:
    """Вопрос-провокация не участвует в доле попаданий: эталонных страниц у него нет."""
    cases = [
        make_case(expected_pages = [[FIRST]], expected_refusal = False),
        make_case(expected_pages = [], expected_refusal = True),
    ]
    measurements = [
        measure(case = cases[0], retrieved = [THIRD], context = "", answer = "", known_pages = {FIRST}, seconds = 0.0),
        measure(case = cases[1], retrieved = [THIRD], context = "", answer = "", known_pages = {FIRST}, seconds = 0.0),
    ]

    assert summarize(cases, measurements, [None, None])["page_hit_rate"].value == 0.0


def test_summary_names_the_questions_that_failed() -> None:
    """Метрика несёт не только долю, но и адрес: по каким вопросам смотреть."""
    cases = [
        make_case(expected_pages = [[FIRST]], expected_refusal = False),
        make_case(expected_pages = [[THIRD]], expected_refusal = False),
    ]
    cases = [cases[0], Case(**{**cases[1].__dict__, "number": 2})]
    measurements = [
        measure(case = cases[0], retrieved = [FIRST], context = "", answer = "", known_pages = {FIRST}, seconds = 0.0),
        measure(case = cases[1], retrieved = [FIRST], context = "", answer = "", known_pages = {FIRST}, seconds = 0.0),
    ]

    page_hits = summarize(cases, measurements, [None, None])["page_hit_rate"]

    assert page_hits.value == 0.5
    assert page_hits.scored == 1
    assert page_hits.total == 2
    assert page_hits.failed == [2]


def test_snippets_catch_right_page_wrong_fragment() -> None:
    """Страница нашлась, а нужный её фрагмент — нет: по страницам зачёт, по факту провал."""
    case = Case(**{
        **make_case(expected_pages = [[FIRST]], expected_refusal = False).__dict__,
        "expected_snippets": ["50,3"],
    })
    metrics = measure(
        case = case,
        retrieved = [FIRST],
        context = "другой фрагмент той же страницы, таблицы в нём нет",
        answer = "",
        known_pages = {FIRST},
        seconds = 0.0,
    )

    assert metrics.page_hit
    assert metrics.snippets_found == 0
    assert metrics.snippets_total == 1


def test_snippets_ignore_line_breaks() -> None:
    """Переносы строк из разобранного PDF не мешают совпадению."""
    case = Case(**{
        **make_case(expected_pages = [[FIRST]], expected_refusal = False).__dict__,
        "expected_snippets": ["359 443"],
    })
    metrics = measure(
        case = case,
        retrieved = [FIRST],
        context = "визиты\n359\n443 штуки",
        answer = "",
        known_pages = {FIRST},
        seconds = 0.0,
    )

    assert metrics.snippets_found == 1


def judged(claims: list, expected_facts: list) -> str:
    """Собирает ответ судьи в том виде, в каком его возвращает модель.

    Аргументы:
        claims: пары «текст, подтверждено» по утверждениям ответа.
        expected_facts: пары «текст, найдено» по фактам эталона.

    Возвращает:
        Строку json.
    """
    return json.dumps({
        "claims": [{"text": text, "supported": holds} for text, holds in claims],
        "expected_facts": [{"text": text, "found": holds} for text, holds in expected_facts],
        "reasoning": "разбор",
    })


def test_scores_are_fractions_not_flags() -> None:
    """Три факта эталона из четырёх дают 0.75, а не «неверно»."""
    verdict = parse_verdict(judged(
        claims = [("процентный риск", True), ("кредитный риск", True), ("валютный риск", True)],
        expected_facts = [
            ("валютный риск", True),
            ("процентный риск", True),
            ("кредитный риск", True),
            ("риск ликвидности", False),
        ],
    ))

    assert faithfulness(verdict) == 1.0
    assert correctness(verdict) == 0.75


def test_unsupported_claim_lowers_faithfulness() -> None:
    """Утверждение, которого нет в контексте, снижает faithfulness."""
    verdict = parse_verdict(judged(
        claims = [("выручка 253,7", True), ("выручка 300,0", False)],
        expected_facts = [("выручка 253,7", True)],
    ))

    assert faithfulness(verdict) == 0.5


def test_refusal_invents_nothing() -> None:
    """Отказ отвечать не приводит фактов, значит и не выдумывает: faithfulness единица."""
    verdict = parse_verdict(judged(
        claims = [],
        expected_facts = [("27 коек", False), ("4 655 кв. м", False)],
    ))

    assert faithfulness(verdict) == 1.0
    assert correctness(verdict) == 0.0


def test_parse_verdict_strips_code_fence() -> None:
    """Модель часто оборачивает json в блок кода — обёртка снимается."""
    verdict = parse_verdict("```json\n" + judged(claims = [("факт", True)], expected_facts = []) + "\n```")

    assert not verdict.error
    assert faithfulness(verdict) == 1.0


def test_broken_response_is_not_scored() -> None:
    """Неразобранный ответ судьи не превращается ни в ноль, ни в единицу."""
    verdict = parse_verdict("судья ответил прозой")

    assert "не разобран" in verdict.error
    assert faithfulness(verdict) is None
    assert correctness(verdict) is None


def test_judge_scores_reach_the_summary() -> None:
    """Оценки судьи попадают в итоговые метрики, иначе прогоны не сравнить."""
    case = make_case(expected_pages = [[FIRST]], expected_refusal = False)
    metrics = measure(
        case = case,
        retrieved = [FIRST],
        context = "",
        answer = "",
        known_pages = {FIRST},
        seconds = 0.0,
    )
    verdict = parse_verdict(judged(
        claims = [("факт", True), ("выдумка", False)],
        expected_facts = [("факт", True)],
    ))

    summary = summarize([case], [metrics], [verdict])

    assert summary["faithfulness"].value == 0.5
    assert summary["correctness"].value == 1.0
    assert summary["judge_parsed"].value == 1.0


def test_questions_file_loads() -> None:
    """Набор вопросов читается, эталоны расшифрованы в имена файлов."""
    cases = load_cases(QUESTIONS_PATH)

    assert cases
    assert all(case.question for case in cases)
    assert all(
        page.document.endswith(".pdf")
        for case in cases
        for group in case.expected_pages
        for page in group
    )


def test_parse_page_expands_alias() -> None:
    """Сокращение документа разворачивается в имя файла."""
    page = parse_page(reference = "FY2025:10", documents = {"FY2025": "отчёт.pdf"})

    assert page == Page(document = "отчёт.pdf", number = 10)


def test_snapshot_with_other_settings_is_protected(tmp_path) -> None:
    """Снимок другой модели не затирается: восстановить его нельзя, снимался минутами."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"settings": {"llm_model": "большая"}}), encoding = "utf-8")

    with pytest.raises(SnapshotBelongsToOtherSettings) as conflict:
        check_snapshot_free(path = path, settings = {"llm_model": "слабая"})

    assert "большая" in str(conflict.value)
    assert "слабая" in str(conflict.value)


def test_snapshot_with_same_settings_is_overwritten() -> None:
    """Пересъёмка теми же настройками — обычное обновление, не конфликт."""
    check_snapshot_free(path = QUESTIONS_PATH.parent / "нет-такого.json", settings = {"a": 1})

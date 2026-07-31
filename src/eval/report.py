"""Вывод результатов прогона: таблица в терминал, отчёт в markdown и json."""
import json
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

from eval.cases import Case
from eval.judge import Verdict, correctness, faithfulness
from eval.metrics import Measure
from eval.results import CaseRun, to_pages
from rag_assistant.config import PROJECT_ROOT, AppConfig
from rag_assistant.prompts import QA_TEMPLATE_TEXT

REPORTS_DIR = PROJECT_ROOT / "docs" / "eval"


def print_table(runs: List[CaseRun]) -> None:
    """Печатает результаты по каждому вопросу одной строкой.

    Аргументы:
        runs: результаты прогона.

    Возвращает:
        None.
    """
    header = f"\n{'#':>2}  {'тип':<13} {'стр.':<5} {'факт':<6} {'RR':<5} {'разн.':<6} {'ссылки':<8} "
    print(header + f"{'судья':<12} вопрос")
    for run in runs:
        metrics = run.metrics
        if run.case.expected_refusal:
            hit = "—"
        else:
            hit = "да" if metrics.page_hit else f"{metrics.covered_groups}/{metrics.total_groups}"
        # В таблице показывается честность: сколько разобранных ссылок ведёт
        # на страницу, которая была в контексте.
        bad = metrics.invented_citations + metrics.outside_context_citations
        citations = f"{metrics.citations - bad}/{metrics.citations}" if metrics.citations else "—"
        facts = f"{metrics.snippets_found}/{metrics.snippets_total}" if metrics.snippets_total else "—"
        print(
            f"{run.case.number:>2}  {run.case.kind:<13} {hit:<5} {facts:<6} "
            f"{metrics.reciprocal_rank:<5.2f} {metrics.unique_pages:<6} {citations:<8} "
            f"{format_verdict_scores(run.verdict):<12} {run.case.question[:38]}"
        )


def format_score(score: float | None) -> str:
    """Печатает оценку судьи или прочерк, если она не измерена.

    Аргументы:
        score: оценка от нуля до единицы либо None.

    Возвращает:
        Оценку с двумя знаками после запятой либо прочерк.
    """
    return f"{score:.2f}" if score is not None else "—"


def format_verdict_scores(verdict: Verdict | None) -> str:
    """Сжимает оценки судьи в одну ячейку таблицы.

    Аргументы:
        verdict: разбор судьи либо None, если судья не вызывался.

    Возвращает:
        Строку вида «1.00/0.75»: faithfulness и correctness.
    """
    if verdict is None:
        return "—"
    if verdict.error:
        return "сбой"

    return f"{format_score(faithfulness(verdict))}/{format_score(correctness(verdict))}"


def print_summary(summary: Dict[str, Measure], runs: List[CaseRun]) -> None:
    """Печатает итоговые метрики со счётом и адресом проблемы.

    Аргументы:
        summary: итоговые метрики прогона.
        runs: результаты прогона, нужны для строки диагностики.

    Возвращает:
        None.
    """
    print("\nИтого:")
    for name, measure in summary.items():
        print(f"  {name:<18} {format_value(measure):<5}   {format_count(measure):<22}{format_failed(measure)}")

    print(f"\nДиагностика:\n  {format_distinct_pages(runs)}")


def format_value(measure: Measure) -> str:
    """Печатает значение метрики.

    Аргументы:
        measure: итоговая метрика.

    Возвращает:
        Число либо прочерк, если считать было не по чему: нулевое значение при
        нулевом счёте означало бы провал, которого не было.
    """
    return f"{measure.value:.3f}" if measure.total else "—"


def format_count(measure: Measure) -> str:
    """Печатает счёт, из которого сложилась метрика.

    Аргументы:
        measure: итоговая метрика.

    Возвращает:
        Строку вида «8 из 10 вопросов».
    """
    return f"{round(measure.scored, 2):g} из {measure.total} {measure.unit}"


def format_failed(measure: Measure) -> str:
    """Печатает номера вопросов, которые зачлись не полностью.

    Аргументы:
        measure: итоговая метрика.

    Возвращает:
        Строку вида «не зачлось: 7, 9» либо пустую строку.
    """
    if not measure.failed:
        return ""

    return "не зачлось: " + ", ".join(str(number) for number in measure.failed)


def format_distinct_pages(runs: List[CaseRun]) -> str:
    """Печатает, по скольким разным страницам разошлась выдача.

    Метрика диагностическая: сама по себе решения не подсказывает, но по ней будет
    виден эффект MMR, когда он появится в поиске.

    Аргументы:
        runs: результаты прогона.

    Возвращает:
        Строку со средним числом страниц и номерами вопросов с повтором страницы.
    """
    if not runs:
        return "страниц в выдаче: нет данных"

    pages = [run.metrics.unique_pages for run in runs]
    repeated = [
        run.case.number
        for run in runs
        if run.metrics.unique_pages < len(to_pages(run.sources))
    ]
    average = sum(pages) / len(pages)
    tail = f", повтор страницы в вопросах: {', '.join(str(number) for number in repeated)}"

    return f"разных страниц в выдаче: {average:.2f} в среднем{tail if repeated else ''}"


def format_expected(case: Case) -> str:
    """Описывает эталонные страницы вопроса.

    Аргументы:
        case: контрольный вопрос.

    Возвращает:
        Группы эталонных страниц строкой либо пометку об ожидаемом отказе.
    """
    if case.expected_refusal:
        return "ожидается отказ «Я не знаю»"

    return " и ".join(
        "(" + " или ".join(str(page) for page in group) + ")"
        for group in case.expected_pages
    )


def render_report(runs: List[CaseRun], summary: Dict[str, Measure], config: AppConfig) -> str:
    """Собирает отчёт о прогоне в markdown.

    Аргументы:
        runs: результаты прогона.
        summary: итоговые метрики.
        config: конфигурация приложения.

    Возвращает:
        Текст отчёта.
    """
    lines = [
        "# Прогон контрольных вопросов",
        "",
        " · ".join(
            f"{name}: `{value}`" if isinstance(value, str) else f"{name}: {value}"
            for name, value in describe_settings(config = config, runs = runs).items()
        ),
        "",
        "## Итого",
        "",
        "| Метрика | Значение | Счёт | Не зачлось |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {name} | {format_value(measure)} | {format_count(measure)} | "
        f"{', '.join(str(number) for number in measure.failed) or '—'} |"
        for name, measure in summary.items()
    )
    lines.extend(["", format_distinct_pages(runs)])

    for run in runs:
        lines.extend(render_case(run))

    return "\n".join(lines) + "\n"


def render_case(run: CaseRun) -> List[str]:
    """Собирает раздел отчёта по одному вопросу.

    Аргументы:
        run: результат прогона вопроса.

    Возвращает:
        Строки раздела.
    """
    metrics = run.metrics
    lines = [
        "",
        f"## {run.case.number}. {run.case.question}",
        "",
        f"- тип: {run.case.kind}",
        f"- эталон: {format_expected(run.case)}",
        "- нашлось:",
    ]
    lines.extend(
        f"  {place}. {page}" for place, page in enumerate(to_pages(run.sources), start = 1)
    )

    if not run.case.expected_refusal:
        lines.append(
            "- попадание: "
            + ("да" if metrics.page_hit else f"{metrics.covered_groups} из {metrics.total_groups} групп")
        )
    if run.answer:
        lines.extend([
            "",
            "Ответ:",
            "",
            "```",
            run.answer,
            "```",
            "",
            f"- ссылок: {metrics.citations}, из них в формате: {metrics.formatted_citations}, "
            f"с нарушением формата: {metrics.malformed_citations}, "
            f"выдуманных: {metrics.invented_citations}, "
            f"вне контекста: {metrics.outside_context_citations}",
        ])
    if run.verdict is not None:
        lines.extend(render_verdict(run.verdict))

    return lines


def render_verdict(verdict: Verdict) -> List[str]:
    """Собирает разбор судьи для отчёта.

    Перечисляются только несошедшиеся пункты: подтвердившиеся видны по числам,
    а разбираться нужно с расхождениями.

    Аргументы:
        verdict: разбор ответа судьёй.

    Возвращает:
        Строки раздела.
    """
    if verdict.error:
        return ["", f"- судья: {verdict.error}"]

    supported = sum(1 for claim in verdict.claims if claim.holds)
    found = sum(1 for fact in verdict.expected_facts if fact.holds)
    lines = [
        "",
        f"- судья: faithfulness {format_score(faithfulness(verdict))} "
        f"({supported} из {len(verdict.claims)} утверждений ответа подтверждены контекстом), "
        f"correctness {format_score(correctness(verdict))} "
        f"({found} из {len(verdict.expected_facts)} фактов эталона попали в ответ)",
    ]
    lines.extend(
        f"  - не подтверждено контекстом: «{claim.text}»"
        for claim in verdict.claims
        if not claim.holds
    )
    lines.extend(
        f"  - нет в ответе: «{fact.text}»"
        for fact in verdict.expected_facts
        if not fact.holds
    )

    if verdict.reasoning:
        lines.append(f"  - {verdict.reasoning}")

    return lines


def describe_settings(config: AppConfig, runs: List[CaseRun]) -> dict:
    """Собирает настройки, при которых снят прогон.

    Без них два снимка невозможно сравнить: непонятно, что именно менялось.
    Модель судьи записывается, только если судья вызывался.

    Аргументы:
        config: конфигурация приложения.
        runs: результаты прогона.

    Возвращает:
        Настройки прогона.
    """
    questions = "\n".join(run.case.question for run in runs)
    settings = {
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        # Отпечаток набора вопросов: сменили формулировки — снимок несравним с прежним,
        # хотя все остальные настройки те же.
        "questions": f"{len(runs)} шт., {sha1(questions.encode()).hexdigest()[:8]}",
        # Промпт — тоже часть проверяемой системы: поменяли правила, снимок несравним.
        "prompt": sha1(QA_TEMPLATE_TEXT.encode()).hexdigest()[:8],
    }

    if any(run.verdict is not None for run in runs):
        settings["judge_model"] = config.judge_model

    return settings


class SnapshotBelongsToOtherSettings(RuntimeError):
    """Снимок с этим именем снят при других настройках."""


def check_snapshot_free(path: Path, settings: dict) -> None:
    """Не даёт затереть снимок, снятый при других настройках.

    Пересъёмка с теми же настройками разрешена — это обычное обновление. А вот прогон
    на другой модели должен лечь рядом, а не поверх: снимок стоит минут, восстановить
    его нельзя, и сравнивать после затирания будет не с чем.

    Аргументы:
        path: путь к json прошлого снимка.
        settings: настройки текущего прогона.

    Возвращает:
        None.

    Исключения:
        SnapshotBelongsToOtherSettings: снимок занят другими настройками.
    """
    if not path.exists():
        return

    stored = json.loads(path.read_text(encoding = "utf-8")).get("settings", {})
    changed = {
        key: (stored.get(key), value)
        for key, value in settings.items()
        if stored.get(key) != value
    }

    if not changed:
        return

    details = "; ".join(
        f"{key}: в снимке {was}, сейчас {now}" for key, (was, now) in sorted(changed.items())
    )
    raise SnapshotBelongsToOtherSettings(
        f"Снимок «{path.stem}» снят при других настройках ({details}). "
        f"Возьмите другое имя: --name <имя>",
    )


def verdict_payload(verdict: Verdict | None) -> dict | None:
    """Готовит разбор судьи к записи в json.

    Аргументы:
        verdict: разбор ответа судьёй либо None, если судья не вызывался.

    Возвращает:
        Разбор вместе с посчитанными по нему оценками либо None.
    """
    if verdict is None:
        return None

    return {
        **asdict(verdict),
        "faithfulness": faithfulness(verdict),
        "correctness": correctness(verdict),
    }


def save_report(
    runs: List[CaseRun],
    summary: Dict[str, Measure],
    config: AppConfig,
    name: str,
) -> None:
    """Сохраняет отчёт в markdown и json.

    Аргументы:
        runs: результаты прогона.
        summary: итоговые метрики.
        config: конфигурация приложения.
        name: имя снимка, из него получаются имена файлов.

    Возвращает:
        None.

    Исключения:
        SnapshotBelongsToOtherSettings: снимок с этим именем снят при других настройках.
    """
    REPORTS_DIR.mkdir(parents = True, exist_ok = True)
    settings = describe_settings(config = config, runs = runs)
    check_snapshot_free(path = REPORTS_DIR / f"{name}.json", settings = settings)

    payload = {
        "settings": settings,
        "summary": {name: asdict(measure) for name, measure in summary.items()},
        "cases": [
            {
                "number": run.case.number,
                "question": run.case.question,
                "retrieved": [str(page) for page in to_pages(run.sources)],
                "answer": run.answer,
                "metrics": asdict(run.metrics),
                "verdict": verdict_payload(run.verdict),
            }
            for run in runs
        ],
    }

    (REPORTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )
    (REPORTS_DIR / f"{name}.md").write_text(
        render_report(runs = runs, summary = summary, config = config),
        encoding = "utf-8",
    )
    print(f"\nОтчёт: {REPORTS_DIR / f'{name}.md'}")

"""Результат прогона одного вопроса — общая форма для прогона и для отчёта."""
from dataclasses import dataclass
from typing import List

from eval.cases import Case, Page
from eval.judge import Verdict
from eval.metrics import CaseMetrics
from rag_assistant.engine import Source


@dataclass(frozen = True)
class CaseRun:
    """Результат прогона одного вопроса.

    Атрибуты:
        case: контрольный вопрос.
        sources: фрагменты, которые вернул поиск, в порядке выдачи.
        answer: ответ ассистента, пустой в режиме проверки только поиска.
        metrics: детерминированные метрики вопроса.
        verdict: оценка судьи либо None, если судья не запускался.
    """

    case: Case
    sources: List[Source]
    answer: str
    metrics: CaseMetrics
    verdict: Verdict | None


def to_page_groups(sources: List[Source]) -> List[List[Page]]:
    """Переводит фрагменты в страницы, не смешивая фрагменты между собой.

    Аргументы:
        sources: фрагменты, которые вернул поиск.

    Возвращает:
        По списку страниц на каждый фрагмент, у которого номер страницы известен,
        в порядке выдачи. Фрагмент со сшитой таблицей отдаёт все свои страницы:
        эталон вопроса может стоять на второй.
    """
    return [
        [
            Page(document = source.file_name, number = number)
            for number in page_range(first = source.page_label, last = source.page_end)
        ]
        for source in sources
        if source.page_label and source.page_label.isdigit()
    ]


def to_pages(sources: List[Source]) -> List[Page]:
    """Переводит фрагменты в страницы, сохраняя порядок выдачи.

    Аргументы:
        sources: фрагменты, которые вернул поиск.

    Возвращает:
        Страницы всех фрагментов одним списком. Фрагмент со сшитой таблицей отдаёт
        все свои страницы, поэтому список бывает длиннее выдачи.
    """
    return [page for group in to_page_groups(sources) for page in group]


def page_range(first: str, last: str | None) -> List[int]:
    """Перечисляет страницы фрагмента.

    Аргументы:
        first: номер первой страницы фрагмента.
        last: номер последней страницы либо None, если фрагмент на одной странице.

    Возвращает:
        Номера страниц от первой до последней включительно.
    """
    if not last or not last.isdigit() or int(last) <= int(first):
        return [int(first)]

    return list(range(int(first), int(last) + 1))

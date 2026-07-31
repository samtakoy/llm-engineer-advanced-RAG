"""Контрольные вопросы: чтение questions.yaml в датаклассы."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


@dataclass(frozen = True)
class Page:
    """Страница документа.

    Атрибуты:
        document: имя файла, как оно лежит в метаданных узла.
        number: номер страницы.
    """

    document: str
    number: int

    def __str__(self) -> str:
        """Возвращает страницу в виде «файл, стр. 8»."""
        return f"{self.document}, стр. {self.number}"


@dataclass(frozen = True)
class Case:
    """Контрольный вопрос вместе с эталоном.

    Атрибуты:
        number: номер вопроса в наборе.
        kind: тип вопроса — факт, ловушка, агрегация, провокация и так далее.
        question: текст вопроса.
        expected_answer: эталонный ответ, по нему судья оценивает правильность.
        expected_pages: группы эталонных страниц. Внутри группы достаточно любой
            страницы, покрыты должны быть все группы.
        expected_snippets: дословные куски, которые обязаны оказаться в тексте выдачи.
            Страница может найтись, а нужный её фрагмент — нет.
        expected_refusal: True — правильный ответ «Я не знаю».
        checks: что вопрос проверяет, для отчёта.
        distractor: чем вопрос сбивает систему, для отчёта.
    """

    number: int
    kind: str
    question: str
    expected_answer: str
    expected_pages: List[List[Page]]
    expected_snippets: List[str]
    expected_refusal: bool
    checks: str
    distractor: str


def parse_page(reference: str, documents: Dict[str, str]) -> Page:
    """Разбирает ссылку вида «FY2025:10» в страницу документа.

    Аргументы:
        reference: ссылка «сокращение:номер».
        documents: расшифровка сокращений в имена файлов.

    Возвращает:
        Страницу с полным именем файла.

    Исключения:
        KeyError: сокращение не расшифровано в questions.yaml.
    """
    alias, _, number = reference.partition(":")

    return Page(document = documents[alias], number = int(number))


def load_cases(path: Path) -> List[Case]:
    """Читает набор контрольных вопросов.

    Аргументы:
        path: путь к questions.yaml.

    Возвращает:
        Вопросы в порядке, заданном в файле.
    """
    content = yaml.safe_load(path.read_text(encoding = "utf-8"))
    documents = content["documents"]

    return [
        Case(
            number = case["number"],
            kind = case["kind"],
            question = case["question"],
            expected_answer = case["expected_answer"],
            expected_pages = [
                [parse_page(reference = reference, documents = documents) for reference in group]
                for group in case["expected_pages"]
            ],
            expected_snippets = case["expected_snippets"],
            expected_refusal = case.get("expected_refusal", False),
            checks = case["checks"],
            distractor = case["distractor"],
        )
        for case in content["cases"]
    ]

"""Оценка ответа моделью: то, чего не видно по страницам.

Страницы показывают, что нужный фрагмент нашёлся. Они не показывают случай «нашёл верную
страницу, но ответил неверно» — его и ловит судья.

Судья не называет балл сам: его просят разложить тексты на отдельные утверждения и
ответить по каждому «да» или «нет». Балл получается делением. Так каждое суждение остаётся
проверяемым, а итог — дробным, и «три риска из четырёх» перестаёт быть равносильно нулю.

Судит та же локальная модель, что отвечает: это слабее независимого судьи, зато
воспроизводимо и бесплатно, а при сравнении двух прогонов систематический сдвиг судьи
одинаков и сокращается.
"""
import json
from dataclasses import dataclass
from typing import List, Sequence

from llama_index.core.llms import LLM

from eval.cases import Case

JUDGE_PROMPT = """Ты — проверяющий. Твоя задача — разобрать ответ ассистента на части и оценить каждую.
Опирайся только на приведённые ниже тексты, свои знания о предмете не используй.

Вопрос: {question}

Фрагменты, которые ассистент получил в качестве контекста:
---------------------
{context}
---------------------

Ответ ассистента:
---------------------
{answer}
---------------------

Эталонный ответ:
---------------------
{expected_answer}
---------------------

Сделай две вещи.

1. claims — разбей ответ ассистента на отдельные утверждения, по одному факту в каждом.
   Для каждого проставь supported: true, если утверждение целиком следует из фрагментов
   выше; false, если во фрагментах его нет или числа расходятся.
   Если ассистент отказался отвечать и фактов не привёл, верни пустой список.
   Формулируй каждое утверждение коротко и самостоятельно, не повторяя в каждом
   общую часть предложения.

2. expected_facts — разбей эталонный ответ на отдельные факты, по одному в каждом.
   Для каждого проставь found: true, если этот факт есть в ответе ассистента.
   Различие в формулировках и единицах измерения допустимо, различие в числах — нет.

Верни строго один объект JSON без пояснений вокруг него:
{{"claims": [{{"text": "...", "supported": true}}],
 "expected_facts": [{{"text": "...", "found": true}}],
 "reasoning": "одно предложение о главном расхождении, если оно есть"}}"""


@dataclass(frozen = True)
class Claim:
    """Отдельное утверждение с вердиктом судьи.

    Атрибуты:
        text: само утверждение.
        holds: для утверждения ответа — следует ли оно из контекста;
            для факта эталона — нашёлся ли он в ответе.
    """

    text: str
    holds: bool


@dataclass(frozen = True)
class Verdict:
    """Разбор ответа судьёй.

    Атрибуты:
        claims: утверждения ответа, проверенные по контексту.
        expected_facts: факты эталона, проверенные по ответу.
        reasoning: обоснование одной фразой, без него оценка непроверяема.
        error: описание проблемы разбора, пустая строка — разбор удался.
    """

    claims: List[Claim]
    expected_facts: List[Claim]
    reasoning: str
    error: str


def faithfulness(verdict: Verdict) -> float | None:
    """Считает долю утверждений ответа, которые следуют из контекста.

    Аргументы:
        verdict: разбор ответа судьёй.

    Возвращает:
        Долю подтверждённых утверждений. Единица, если ассистент фактов не приводил:
        отказ отвечать ничего не выдумывает. None, если разбор не удался.
    """
    if verdict.error:
        return None
    if not verdict.claims:
        return 1.0

    return sum(1 for claim in verdict.claims if claim.holds) / len(verdict.claims)


def correctness(verdict: Verdict) -> float | None:
    """Считает долю фактов эталона, попавших в ответ.

    Аргументы:
        verdict: разбор ответа судьёй.

    Возвращает:
        Долю найденных фактов либо None, если разбор не удался или эталон
        не разложен на факты — тогда мерить нечего.
    """
    if verdict.error or not verdict.expected_facts:
        return None

    return sum(1 for fact in verdict.expected_facts if fact.holds) / len(verdict.expected_facts)


def judge_answer(llm: LLM, case: Case, answer: str, contexts: Sequence[str]) -> Verdict:
    """Просит модель разобрать ответ ассистента на утверждения.

    Аргументы:
        llm: модель-судья.
        case: контрольный вопрос с эталонным ответом.
        answer: ответ ассистента.
        contexts: тексты фрагментов, переданных ассистенту.

    Возвращает:
        Разбор ответа судьёй.
    """
    prompt = JUDGE_PROMPT.format(
        question = case.question,
        expected_answer = case.expected_answer,
        context = "\n\n---\n\n".join(contexts),
        answer = answer,
    )

    return parse_verdict(str(llm.complete(prompt)))


def parse_verdict(response: str) -> Verdict:
    """Разбирает ответ судьи.

    Аргументы:
        response: текст, который вернула модель-судья.

    Возвращает:
        Разбор судьи. Если ответ неразбираем, возвращается пустой разбор с описанием
        проблемы: такой случай не должен молча становиться нулём или единицей.
    """
    text = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

    try:
        content = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return Verdict(
            claims = [],
            expected_facts = [],
            reasoning = "",
            error = f"ответ судьи не разобран: {error}",
        )

    return Verdict(
        claims = parse_claims(items = content.get("claims", []), verdict_key = "supported"),
        expected_facts = parse_claims(items = content.get("expected_facts", []), verdict_key = "found"),
        reasoning = str(content.get("reasoning", "")),
        error = "",
    )


def parse_claims(items: list, verdict_key: str) -> List[Claim]:
    """Собирает список утверждений из разобранного json.

    Аргументы:
        items: элементы списка, как их вернул судья.
        verdict_key: имя поля с вердиктом — supported для ответа, found для эталона.

    Возвращает:
        Утверждения, пропуская элементы неожиданной формы.
    """
    return [
        Claim(text = str(item.get("text", "")), holds = bool(item.get(verdict_key)))
        for item in items
        if isinstance(item, dict)
    ]

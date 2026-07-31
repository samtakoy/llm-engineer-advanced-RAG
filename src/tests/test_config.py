"""Проверка чтения конфигурации из окружения."""
import pytest

from rag_assistant.config import AppConfig


def test_judge_model_falls_back_to_answering_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Судья по умолчанию берёт ту же модель, что отвечает на вопросы."""
    monkeypatch.setenv("LOCAL_MODEL", "модель-которая-отвечает")
    monkeypatch.delenv("JUDGE_MODEL", raising = False)

    config = AppConfig.from_env()

    assert config.judge_model == "модель-которая-отвечает"


def test_judge_model_can_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Заданная отдельно модель судьи не совпадает с отвечающей."""
    monkeypatch.setenv("LOCAL_MODEL", "модель-которая-отвечает")
    monkeypatch.setenv("JUDGE_MODEL", "модель-которая-судит")

    config = AppConfig.from_env()

    assert config.llm_model == "модель-которая-отвечает"
    assert config.judge_model == "модель-которая-судит"


def test_empty_judge_model_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустая переменная равносильна незаданной: пустое имя модели сервер не примет."""
    monkeypatch.setenv("LOCAL_MODEL", "модель-которая-отвечает")
    monkeypatch.setenv("JUDGE_MODEL", "")

    assert AppConfig.from_env().judge_model == "модель-которая-отвечает"

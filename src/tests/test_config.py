"""Проверка чтения конфигурации из окружения."""
import pytest

from rag_assistant.config import AppConfig


def test_judge_model_is_set_apart_from_answering_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Судья настраивается отдельно: слабую модель проверяют сильным судьёй."""
    monkeypatch.setenv("LOCAL_MODEL", "модель-которая-отвечает")
    monkeypatch.setenv("JUDGE_MODEL", "модель-которая-судит")

    config = AppConfig.from_env()

    assert config.llm_model == "модель-которая-отвечает"
    assert config.judge_model == "модель-которая-судит"

"""Подпись индекса: настройки, при смене которых готовые векторы становятся негодными.

Подпись пишется в метаданные коллекции при сборке и сверяется при открытии. Без неё
смена модели эмбеддингов или размера чанка ничем себя не проявляет: если размерность
вектора совпала, приложение поднимется на старых векторах и будет молча отвечать мимо.
"""
from typing import Dict

from chromadb.api.models.Collection import Collection

from rag_assistant.config import AppConfig


class IndexSettingsChanged(RuntimeError):
    """Настройки нарезки или эмбеддинга разошлись с теми, на которых построен индекс."""


def build_signature(config: AppConfig) -> Dict[str, str]:
    """Собирает подпись текущих настроек.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Настройки строками — ChromaDB хранит метаданные коллекции только так.
    """
    return {
        "embedding_model": config.embedding_model,
        "chunk_size": str(config.chunk_size),
        "chunk_overlap": str(config.chunk_overlap),
    }


def verify_signature(collection: Collection, config: AppConfig) -> None:
    """Сверяет текущие настройки с подписью готового индекса.

    Аргументы:
        collection: коллекция готового индекса.
        config: конфигурация приложения.

    Возвращает:
        None.

    Исключения:
        IndexSettingsChanged: хотя бы одна настройка разошлась с подписью.
    """
    stored = collection.metadata or {}
    changed = {
        key: (stored.get(key), value)
        for key, value in build_signature(config).items()
        if stored.get(key) != value
    }

    if not changed:
        return

    details = "; ".join(
        f"{key}: индекс построен на «{was if was is not None else 'настройке без подписи'}», "
        f"в конфигурации «{now}»"
        for key, (was, now) in sorted(changed.items())
    )
    raise IndexSettingsChanged(
        f"Индекс не соответствует настройкам ({details}). "
        f"Перестроить: uv run main.py --reindex",
    )

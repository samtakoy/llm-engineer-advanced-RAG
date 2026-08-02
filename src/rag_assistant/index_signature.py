"""Подпись индекса: настройки, при смене которых готовые векторы становятся негодными.

Подпись пишется в метаданные коллекции при сборке и сверяется при открытии. Без неё
смена модели эмбеддингов или размера чанка ничем себя не проявляет: если размерность
вектора совпала, приложение поднимется на старых векторах и будет молча отвечать мимо.
"""
from typing import Dict, Mapping, Tuple

from chromadb.api.models.Collection import Collection

from rag_assistant.config import AppConfig
from rag_assistant.ingest.loader import DOCUMENT_METADATA


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
        "document_metadata": format_document_metadata(),
    }


def format_document_metadata() -> str:
    """Собирает разметку документов в строку для подписи.

    Метаданные лежат в Chroma рядом с векторами: правка разметки без пересборки
    оставила бы фильтр работать по прежним значениям.

    Возвращает:
        Разметку вида «файл.pdf: year=2024; …».
    """
    return "; ".join(
        f"{file_name}: " + ", ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        for file_name, fields in sorted(DOCUMENT_METADATA.items())
    )


def find_signature_changes(
    stored: Mapping[str, str],
    config: AppConfig,
) -> Dict[str, Tuple[str | None, str | None]]:
    """Ищет настройки, разошедшиеся с сохранённой подписью.

    Аргументы:
        stored: подпись, записанная при сборке.
        config: конфигурация приложения.

    Возвращает:
        Расхождения по имени настройки: «что записано при сборке, что в конфигурации».
        Пусто — настройки совпали. None на месте записанного значения означает, что
        настройки в подписи не было вовсе, None на месте текущего — что настройка
        из подписи ушла, а индекс собран ещё с ней.
    """
    signature = build_signature(config)

    return {
        key: (stored.get(key), signature.get(key))
        for key in set(stored) | set(signature)
        if stored.get(key) != signature.get(key)
    }


def describe_signature_changes(changed: Mapping[str, Tuple[str | None, str | None]]) -> str:
    """Описывает расхождения настроек для сообщения пользователю.

    Аргументы:
        changed: расхождения, найденные find_signature_changes.

    Возвращает:
        Перечисление вида «chunk_size: собрано на «480», в конфигурации «512»».
    """
    return "; ".join(
        f"{key}: собрано на «{was if was is not None else 'настройке без подписи'}», "
        f"в конфигурации «{now if now is not None else 'настройки больше нет'}»"
        for key, (was, now) in sorted(changed.items())
    )


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
    changed = find_signature_changes(stored = collection.metadata or {}, config = config)

    if not changed:
        return

    raise IndexSettingsChanged(
        f"Индекс не соответствует настройкам ({describe_signature_changes(changed)}). "
        f"Перестроить: uv run main.py reindex",
    )

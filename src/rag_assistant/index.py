"""Векторный индекс: ChromaDB как хранилище, VectorStoreIndex как логический слой."""
from typing import List

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_assistant.config import AppConfig


def build_index(config: AppConfig, documents: List[Document]) -> VectorStoreIndex:
    """Строит индекс с нуля: документы режутся на узлы и векторизуются.

    Аргументы:
        config: конфигурация приложения.
        documents: документы для индексации.

    Возвращает:
        Индекс поверх коллекции ChromaDB.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))
    collection = client.get_or_create_collection(name = config.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection = collection)

    return VectorStoreIndex.from_documents(
        documents,
        storage_context = StorageContext.from_defaults(vector_store = vector_store),
        show_progress = True,
    )


def load_index(config: AppConfig) -> VectorStoreIndex:
    """Подключается к уже построенному индексу без повторной векторизации.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Индекс поверх существующей коллекции ChromaDB.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))
    collection = client.get_or_create_collection(name = config.chroma_collection)

    return VectorStoreIndex.from_vector_store(
        vector_store = ChromaVectorStore(chroma_collection = collection),
    )


def open_index(config: AppConfig, documents: List[Document], rebuild: bool) -> VectorStoreIndex:
    """Возвращает готовый к запросам индекс, строя его только при необходимости.

    Аргументы:
        config: конфигурация приложения.
        documents: документы для индексации, если индекс придётся строить.
        rebuild: True — удалить коллекцию и проиндексировать заново.

    Возвращает:
        Индекс, готовый отдавать движок запросов.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))

    if rebuild and config.chroma_collection in [c.name for c in client.list_collections()]:
        client.delete_collection(name = config.chroma_collection)

    if client.get_or_create_collection(name = config.chroma_collection).count() > 0:
        return load_index(config)

    return build_index(config = config, documents = documents)

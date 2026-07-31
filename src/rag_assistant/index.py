"""Векторный индекс: ChromaDB как хранилище, VectorStoreIndex как логический слой."""
from typing import Callable, List

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_assistant.config import AppConfig
from rag_assistant.index_signature import build_signature, verify_signature
from rag_assistant.models import create_node_parser

# Векторы нормализованы, поэтому косинус и заданная по умолчанию l2 дают один и тот же
# порядок выдачи. Косинусное расстояние выбрано ради границ [0, 2]: на нём осмысленно
# ставить порог отсечения. Метрика задаётся только в момент создания коллекции.
COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


class DocstoreMissing(RuntimeError):
    """Векторы коллекции есть, а сохранённых связей между узлами рядом нет."""


def find_collection(client: ClientAPI, name: str) -> Collection | None:
    """Находит коллекцию по имени.

    Аргументы:
        client: клиент ChromaDB.
        name: имя коллекции.

    Возвращает:
        Коллекцию либо None, если её ещё нет.
    """
    if name not in [collection.name for collection in client.list_collections()]:
        return None

    return client.get_collection(name = name)


def build_index(
    config: AppConfig,
    client: ClientAPI,
    documents: List[Document],
) -> VectorStoreIndex:
    """Строит индекс с нуля: документы режутся на узлы и векторизуются.

    Аргументы:
        config: конфигурация приложения.
        client: клиент ChromaDB.
        documents: документы для индексации.

    Возвращает:
        Индекс поверх новой коллекции ChromaDB.
    """
    nodes = create_node_parser(config).get_nodes_from_documents(documents, show_progress = True)
    collection = client.create_collection(
        name = config.chroma_collection,
        configuration = COLLECTION_CONFIGURATION,
        metadata = build_signature(config),
    )
    storage_context = StorageContext.from_defaults(
        vector_store = ChromaVectorStore(chroma_collection = collection),
        docstore = SimpleDocumentStore(),
    )

    # В докстор кладутся все узлы, включая родительские: по ним поиск потом
    # собирает крупный фрагмент. Векторизуются только листья — родитель попал бы
    # в выдачу наравне с потомками и занял бы место своей же копией.
    storage_context.docstore.add_documents(nodes)
    index = VectorStoreIndex(
        nodes = get_leaf_nodes(nodes),
        storage_context = storage_context,
        show_progress = True,
    )
    storage_context.docstore.persist(persist_path = str(config.docstore_path))

    return index


def load_index(config: AppConfig, collection: Collection) -> VectorStoreIndex:
    """Подключается к уже построенному индексу без повторной векторизации.

    Аргументы:
        config: конфигурация приложения.
        collection: коллекция с готовыми векторами.

    Возвращает:
        Индекс поверх существующей коллекции ChromaDB и сохранённого докстора.

    Исключения:
        DocstoreMissing: коллекция есть, а файла со связями узлов рядом нет.
    """
    if not config.docstore_path.exists():
        raise DocstoreMissing(
            f"Векторы есть, а связи узлов ({config.docstore_path}) не найдены. "
            f"Перестроить: uv run main.py reindex",
        )

    # Пустой список узлов означает «ничего не добавлять»: векторы уже в коллекции.
    # Готового конструктора нет — from_vector_store выбрасывает переданный
    # storage_context и собирает свой, без докстора.
    return VectorStoreIndex(
        nodes = [],
        storage_context = StorageContext.from_defaults(
            vector_store = ChromaVectorStore(chroma_collection = collection),
            docstore = SimpleDocumentStore.from_persist_path(str(config.docstore_path)),
        ),
    )


def open_index(
    config: AppConfig,
    load_documents: Callable[[], List[Document]],
    rebuild: bool,
) -> VectorStoreIndex:
    """Возвращает готовый к запросам индекс, строя его только при необходимости.

    Аргументы:
        config: конфигурация приложения.
        load_documents: чтение документов, вызывается только если индекс строится.
            Разбор PDF занимает секунды и на готовом индексе не нужен.
        rebuild: True — удалить коллекцию и проиндексировать заново.

    Возвращает:
        Индекс, готовый отдавать движок запросов.

    Исключения:
        IndexSettingsChanged: готовый индекс построен на других настройках.
        DocstoreMissing: векторы есть, а связей между узлами нет.
    """
    client = chromadb.PersistentClient(path = str(config.chroma_dir))
    collection = find_collection(client = client, name = config.chroma_collection)

    # Пустая коллекция остаётся от прерванной сборки: метрика и подпись настроек
    # задаются при создании, поэтому её проще пересоздать, чем достраивать.
    if collection is not None and (rebuild or collection.count() == 0):
        client.delete_collection(name = config.chroma_collection)
        config.docstore_path.unlink(missing_ok = True)
        collection = None

    if collection is not None:
        verify_signature(collection = collection, config = config)
        return load_index(config = config, collection = collection)

    return build_index(config = config, client = client, documents = load_documents())

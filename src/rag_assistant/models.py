"""Модели LlamaIndex: LLM, эмбеддер, сплиттер и их регистрация в Settings."""
import torch
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from rag_assistant.config import AppConfig


def select_device() -> str:
    """Выбирает устройство для модели эмбеддингов.

    Возвращает:
        "mps" на Apple Silicon, "cuda" на NVIDIA, иначе "cpu".
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_llm(config: AppConfig) -> OpenAILike:
    """Создаёт клиент к локальному OpenAI-совместимому серверу.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Клиент OpenAILike, настроенный на локальную модель.
    """
    additional_kwargs = {}
    if config.llm_reasoning_effort:
        additional_kwargs["reasoning_effort"] = config.llm_reasoning_effort

    return OpenAILike(
        model = config.llm_model,
        api_base = config.llm_base_url,
        api_key = config.llm_api_key,
        temperature = config.llm_temperature,
        max_tokens = config.llm_max_tokens,
        context_window = config.llm_context_window,
        timeout = config.llm_timeout_seconds,
        is_chat_model = True,
        additional_kwargs = additional_kwargs,
    )


def create_embedding_model(config: AppConfig) -> HuggingFaceEmbedding:
    """Создаёт модель эмбеддингов.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Модель HuggingFaceEmbedding. Моделям семейства e5 добавляются
        обязательные префиксы "query: " и "passage: ".
    """
    is_e5_family = "e5" in config.embedding_model.lower()

    return HuggingFaceEmbedding(
        model_name = config.embedding_model,
        device = select_device(),
        normalize = True,
        query_instruction = "query: " if is_e5_family else None,
        text_instruction = "passage: " if is_e5_family else None,
    )


def configure_global_settings(config: AppConfig) -> None:
    """Прописывает модели и сплиттер в глобальный Settings LlamaIndex.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        None.
    """
    Settings.llm = create_llm(config)
    Settings.embed_model = create_embedding_model(config)
    Settings.node_parser = SentenceSplitter(
        chunk_size = config.chunk_size,
        chunk_overlap = config.chunk_overlap,
    )

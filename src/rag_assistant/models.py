"""Модели LlamaIndex: LLM, эмбеддер, сплиттер и их регистрация в Settings."""
from typing import Callable, List

import torch
from llama_index.core import Settings
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from transformers import AutoTokenizer

from rag_assistant.config import AppConfig

# Имена уровней нарезки: порядок в HierarchicalNodeParser задаёт вложенность,
# первый уровень режет документ, каждый следующий — узлы предыдущего.
PARENT_LEVEL = "parent"
LEAF_LEVEL = "leaf"


class TextTooLongForModel(RuntimeError):
    """Узел не помещается в окно модели: лишнее было бы молча отброшено."""


def model_window(model) -> int:
    """Определяет, сколько токенов модель физически способна прочитать.

    Аргументы:
        model: SentenceTransformer или CrossEncoder.

    Возвращает:
        Размер окна модели в токенах.
    """
    return model[0].auto_model.config.max_position_embeddings


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


def create_llm(config: AppConfig, model: str) -> OpenAILike:
    """Создаёт клиент к локальному OpenAI-совместимому серверу.

    Аргументы:
        config: конфигурация приложения.
        model: идентификатор модели на сервере. Задаётся явно, потому что
            отвечать и судить могут разные модели.

    Возвращает:
        Клиент OpenAILike, настроенный на локальную модель.
    """
    additional_kwargs = {}
    if config.llm_reasoning_effort:
        additional_kwargs["reasoning_effort"] = config.llm_reasoning_effort

    return OpenAILike(
        model = model,
        api_base = config.llm_base_url,
        api_key = config.llm_api_key,
        temperature = config.llm_temperature,
        max_tokens = config.llm_max_tokens,
        context_window = config.llm_context_window,
        timeout = config.llm_timeout_seconds,
        is_chat_model = True,
        # Без этого флага LlamaIndex не кладёт в запрос список tools, но всё
        # равно требует вызова инструмента, и сервер отвечает ошибкой 400.
        # Ломается всё, что просит структурированный ответ.
        is_function_calling_model = True,
        additional_kwargs = additional_kwargs,
    )


def create_embedding_model(config: AppConfig) -> HuggingFaceEmbedding:
    """Создаёт модель эмбеддингов.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Модель HuggingFaceEmbedding. Моделям семейства e5 добавляются
        обязательные префиксы "query:" и "passage:".

    Исключения:
        TextTooLongForModel: листовой чанк не помещается в окно модели.
    """
    is_e5_family = "e5" in config.embedding_model.lower()

    embedding_model = HuggingFaceEmbedding(
        model_name = config.embedding_model,
        device = select_device(),
        normalize = True,
        # Пробел между префиксом и текстом ставит сам llama-index. Свой пробел
        # дал бы "query:  вопрос" вместо ожидаемого моделью "query: вопрос".
        query_instruction = "query:" if is_e5_family else None,
        text_instruction = "passage:" if is_e5_family else None,
    )
    # Векторизуются листья, поэтому в окно должен помещаться именно лист. Всё, что
    # за границей окна, модель отбрасывает без предупреждения: текст лежал бы
    # в индексе, но для поиска не существовал.
    limit = embedding_model._model.max_seq_length

    if config.chunk_size > limit:
        raise TextTooLongForModel(
            f"CHUNK_SIZE={config.chunk_size} больше окна модели {config.embedding_model} "
            f"({limit} токенов). Хвост чанка в вектор не попадёт — уменьшите CHUNK_SIZE.",
        )

    return embedding_model


def create_reranker(config: AppConfig) -> SentenceTransformerRerank | None:
    """Создаёт cross-encoder, переставляющий найденные фрагменты по близости к вопросу.

    Эмбеддер считает вектор вопроса и вектор фрагмента порознь, поэтому меряет
    близость приблизительно. Cross-encoder читает пару «вопрос — фрагмент» целиком
    и оценивает точнее, но за это платится вызовом модели на каждый фрагмент —
    отсюда порядок работы: поиск приносит кандидатов, реранкер оставляет лучших.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Реранкер либо None, если модель не задана: тогда порядок остаётся за поиском.

    Исключения:
        TextTooLongForModel: родительский узел не помещается в окно модели.
    """
    if not config.rerank_model:
        return None

    reranker = SentenceTransformerRerank(
        model = config.rerank_model,
        top_n = config.top_k,
        device = select_device(),
        keep_retrieval_score = True,
    )
    # Реранкер оценивает то, что вернул поиск, а поиск отдаёт склеенные родительские
    # узлы — значит мерить нужно по их размеру, а не по размеру листа.
    window = model_window(reranker._model)

    if config.parent_chunk_size > window:
        raise TextTooLongForModel(
            f"PARENT_CHUNK_SIZE={config.parent_chunk_size} больше окна модели "
            f"{config.rerank_model} ({window} токенов). Конец узла до оценки не дойдёт — "
            f"возьмите модель с большим окном или уменьшите PARENT_CHUNK_SIZE.",
        )

    # Обёртка llama-index создаёт модель с жёстким пределом в 512 токенов и параметра
    # не даёт, поэтому предел поднимается до размера родительского узла: на 512 токенах
    # хвост узла молча не дошёл бы до оценки.
    reranker._model.max_seq_length = config.parent_chunk_size

    return reranker


def create_embedding_tokenizer(config: AppConfig) -> Callable[[str], List[int]]:
    """Создаёт счётчик токенов той модели, которая будет векторизовать текст.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Функцию, считающую токены так же, как их считает эмбеддер.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)

    # Токенизатор здесь только считает, а не готовит вход модели: предупреждение
    # о превышении длины выдал бы сам сплиттер, которому эту длину и мерить.
    tokenizer.model_max_length = int(1e9)

    return tokenizer.encode


def create_node_parser(config: AppConfig) -> HierarchicalNodeParser:
    """Создаёт сплиттер документов на двухуровневые узлы.

    Векторизуются только листья, поэтому их размер ограничен окном эмбеддера.
    Родительский узел крупнее и нужен на выдаче: когда в неё попадает несколько
    листьев одного родителя, поиск отдаёт модели родителя целиком.

    Размер чанка меряется токенизатором эмбеддера, а не заданным по умолчанию
    токенизатором OpenAI: у них разное дробление русского текста, и на настройках
    по умолчанию чанк не помещался в окно эмбеддера, а лишнее молча отбрасывалось.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Сплиттер, дающий родительские узлы и листья со связями между ними.
    """
    tokenizer = create_embedding_tokenizer(config)
    levels = {
        PARENT_LEVEL: config.parent_chunk_size,
        LEAF_LEVEL: config.chunk_size,
    }

    return HierarchicalNodeParser.from_defaults(
        node_parser_ids = list(levels),
        node_parser_map = {
            level: SentenceSplitter(
                chunk_size = chunk_size,
                chunk_overlap = config.chunk_overlap,
                tokenizer = tokenizer,
            )
            for level, chunk_size in levels.items()
        },
    )


def configure_global_settings(config: AppConfig) -> None:
    """Прописывает модели и сплиттер в глобальный Settings LlamaIndex.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        None.
    """
    Settings.llm = create_llm(config = config, model = config.llm_model)
    Settings.embed_model = create_embedding_model(config)
    Settings.node_parser = create_node_parser(config)

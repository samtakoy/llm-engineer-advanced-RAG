"""Модели LlamaIndex: LLM, эмбеддер, сплиттер и их регистрация в Settings."""
import sys
from typing import Any, Callable, Dict, List, Type

import torch
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from rag_assistant.config import AppConfig
from rag_assistant.ingest.chained_parser import ChainedNodeParser
from rag_assistant.ingest.heading_merge import HeadingMergeParser
from rag_assistant.ingest.table_splitter import TableRowSplitter

# До скольких символов ужимается сообщение об ошибке разбора.
ERROR_MESSAGE_LIMIT = 200


class TextTooLongForModel(RuntimeError):
    """Узел не помещается в окно модели: лишнее было бы молча отброшено."""


class SchemaConstrainedLLM(OpenAILike):
    """Клиент, требующий структурированный ответ схемой, а не вызовом инструмента.

    Атрибуты:
        failed_responses: сколько ответов сервера не удалось разобрать по схеме.
    """

    failed_responses: int = Field(
        default = 0,
        description = "Сколько ответов сервера не удалось разобрать по схеме.",
    )

    def _should_use_structure_outputs(self) -> bool:
        """Сообщает, что сервер умеет ограничивать генерацию схемой.

        Возвращает:
            True всегда: класс и создаётся для серверов, которые это умеют.
        """
        return True

    async def astructured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Запрашивает структурированный ответ и считает неудачные разборы.

        Аргументы:
            output_cls: класс, описывающий ожидаемый ответ.
            prompt: шаблон запроса.
            llm_kwargs: аргументы, уходящие в запрос к серверу.
            prompt_args: значения для подстановки в шаблон.

        Возвращает:
            Разобранный ответ модели.

        Исключения:
            ValueError, TypeError, AttributeError: ответ сервера не разобрался
                по схеме. Те же типы, что ловит извлечение троек llama-index.
        """
        try:
            return await super().astructured_predict(
                output_cls = output_cls,
                prompt = prompt,
                llm_kwargs = llm_kwargs,
                **prompt_args,
            )
        except (ValueError, TypeError, AttributeError) as error:
            self.failed_responses += 1
            message = " ".join(str(error).split())[:ERROR_MESSAGE_LIMIT]
            print(
                f"Ответ не разобран по схеме: {message or type(error).__name__}",
                file = sys.stderr,
            )
            raise


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


def create_llm(config: AppConfig, model: str, schema_constrained: bool) -> OpenAILike:
    """Создаёт клиент к локальному OpenAI-совместимому серверу.

    Аргументы:
        config: конфигурация приложения.
        model: идентификатор модели на сервере. Задаётся явно, потому что
            отвечать и судить могут разные модели.
        schema_constrained: True — просить структурированный ответ схемой,
            чтобы сервер ограничил генерацию грамматикой. Нужно там, где ответ
            модели разбирается программой, а не читается человеком.

    Возвращает:
        Клиент OpenAILike, настроенный на локальную модель.
    """
    additional_kwargs = {}
    if config.llm_reasoning_effort:
        additional_kwargs["reasoning_effort"] = config.llm_reasoning_effort

    client_class = SchemaConstrainedLLM if schema_constrained else OpenAILike

    return client_class(
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
        TextTooLongForModel: чанк не помещается в окно модели.
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
    # В окно должен помещаться весь чанк. Всё, что за границей окна, модель
    # отбрасывает без предупреждения: текст лежал бы в индексе, но для поиска
    # не существовал.
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
    """
    if not config.rerank_model:
        return None

    return SentenceTransformerRerank(
        model = config.rerank_model,
        top_n = config.top_k,
        device = select_device(),
        keep_retrieval_score = True,
        # Класс из llama-index-core зашивает предел в 512 токенов и наружу его
        # не отдаёт; у этой же обёртки отдельной интеграцией настройки CrossEncoder
        # открыты. None означает «взять предел, объявленный самой моделью» — без
        # этого ключа интеграция подставит те же 512, и хвост длинного узла молча
        # не дошёл бы до оценки.
        cross_encoder_kwargs = {"max_length": None},
    )


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


def create_node_parser(config: AppConfig) -> NodeParser:
    """Создаёт сплиттер документов на узлы одного уровня.

    Каждый узел векторизуется, поэтому его размер ограничен окном эмбеддера.

    Размер чанка меряется токенизатором эмбеддера, а не заданным по умолчанию
    токенизатором OpenAI: у них разное дробление русского текста, и на настройках
    по умолчанию чанк не помещался в окно эмбеддера, а лишнее молча отбрасывалось.

    Границу ставит заголовок раздела, а узел, в котором остался один заголовок,
    приклеивается к следующему. Таблица, не влезающая в окно, режется по своим
    строкам с повтором шапки — иначе её дорезал бы предел длины, по предложениям
    и посреди строк, оставляя числа без имён колонок. Предел длины стоит последним
    и добирает то, что структурой не разрезалось.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        Парсер, дающий плоский список узлов в пределах окна эмбеддера.
    """
    return ChainedNodeParser(
        parsers = [
            MarkdownNodeParser(),
            HeadingMergeParser(),
            TableRowSplitter.create(
                chunk_size = config.chunk_size,
                tokenizer = create_embedding_tokenizer(config),
            ),
            SentenceSplitter(
                chunk_size = config.chunk_size,
                chunk_overlap = config.chunk_overlap,
                tokenizer = create_embedding_tokenizer(config),
            ),
        ],
    )


def configure_global_settings(config: AppConfig) -> None:
    """Прописывает модели и сплиттер в глобальный Settings LlamaIndex.

    Аргументы:
        config: конфигурация приложения.

    Возвращает:
        None.
    """
    # Ответы читает человек, схемой их не ограничивают.
    Settings.llm = create_llm(
        config = config,
        model = config.llm_model,
        schema_constrained = False,
    )
    Settings.embed_model = create_embedding_model(config)
    Settings.node_parser = create_node_parser(config)

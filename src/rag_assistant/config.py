"""Конфигурация приложения: единственное место, где читается окружение."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen = True)
class AppConfig:
    """Настройки пайплайна.

    Атрибуты:
        llm_base_url: адрес OpenAI-совместимого сервера.
        llm_api_key: ключ сервера.
        llm_model: идентификатор модели, которая отвечает на вопросы.
        judge_model: идентификатор модели, которая оценивает ответы при прогоне
            контрольных вопросов. По умолчанию та же, что отвечает.
        llm_temperature: температура генерации.
        llm_max_tokens: лимит токенов ответа.
        llm_context_window: размер контекстного окна модели.
        llm_timeout_seconds: таймаут запроса к модели.
        llm_reasoning_effort: уровень reasoning (low, medium, high) или None.
        embedding_model: имя модели эмбеддингов из HuggingFace.
        documents_dir: папка с исходными документами.
        parsed_dir: папка, в которую выгружается нарезка для просмотра.
        chroma_dir: папка, в которой ChromaDB хранит векторы.
        chroma_collection: имя коллекции векторов.
        chunk_size: размер чанка в токенах эмбеддера, вместе с метаданными.
            Должен помещаться в окно эмбеддера, иначе хвост чанка не векторизуется.
        chunk_overlap: перекрытие соседних чанков в тех же токенах.
        top_k: сколько фрагментов забирает поиск.
    """

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    judge_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_context_window: int
    llm_timeout_seconds: float
    llm_reasoning_effort: str | None

    embedding_model: str

    documents_dir: Path
    parsed_dir: Path
    chroma_dir: Path
    chroma_collection: str

    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Собирает конфигурацию из переменных окружения и файла .env.

        Возвращает:
            AppConfig со значениями по умолчанию там, где переменная не задана.
        """

        local_model = os.getenv("LOCAL_MODEL", "google/gemma-4-e2b")
        # local_model = os.getenv("LOCAL_MODEL", "google/gemma-4-26b-a4b-qat")

        return cls(
            llm_base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"),
            llm_api_key = os.getenv("LOCAL_API_KEY", "lm-studio"),
            llm_model = local_model,
            judge_model = os.getenv("JUDGE_MODEL", "google/gemma-4-26b-a4b-qat") or local_model,
            llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1")),
            llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096")),
            llm_context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "32768")),
            llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "600")),
            llm_reasoning_effort = os.getenv("LOCAL_REASONING_EFFORT") or None,

            embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),

            documents_dir = PROJECT_ROOT / os.getenv("DOCUMENTS_DIR", "documents"),
            parsed_dir = PROJECT_ROOT / os.getenv("PARSED_DIR", "parsed"),
            chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma"),
            chroma_collection = os.getenv("CHROMA_COLLECTION", "documents"),

            # Окно multilingual-e5-small — 512 токенов, остаток оставлен под
            # служебные токены и префикс "passage:", который эмбеддер ставит сам.
            chunk_size = int(os.getenv("CHUNK_SIZE", "480")),
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "96")),
            top_k = int(os.getenv("TOP_K", "5")),
        )

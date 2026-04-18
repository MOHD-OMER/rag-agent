"""
Centralized configuration using pydantic-settings.
All values can be overridden via environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(BASE_DIR / "agent_memory.db")


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field("groq", description="'groq' or 'ollama'")
    groq_api_key: str = Field("", description="Groq API key")
    groq_model: str = Field("llama-3.1-8b-instant")
    ollama_base_url: str = Field("http://localhost:11434")
    ollama_model: str = Field("llama3.2")

    # Search
    tavily_api_key: str = Field("", description="Tavily search API key")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2")

    # Vectors & Storage
    chroma_persist_dir: str = Field("./chroma_db")
    chroma_collection_name: str = Field("rag_documents")
    agent_checkpoint_db: str = Field(DEFAULT_DB_PATH)

    # Chunking
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)

    # Retrieval
    top_k_retrieval: int = Field(6)
    bm25_weight: float = Field(0.4)
    dense_weight: float = Field(0.6)

    # Memory
    memory_window_size: int = Field(10)

    # API
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)
    allowed_origins: str = Field("http://localhost:8501,http://localhost:7860")

    # Logging
    log_level: str = Field("INFO")

    model_config = {"env_file": ["../.env", ".env"], "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()

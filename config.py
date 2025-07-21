import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    MISTRAL = "mistral"


class VectorStore(str, Enum):
    QDRANT = "qdrant"
    LANCEDB = "lancedb"
    FAISS = "faiss"

class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0", env="GHOST_TRACER_API_HOST")
    api_port: int = Field(default=8000, env="GHOST_TRACER_API_PORT")
    debug: bool = Field(default=False, env="GHOST_TRACER_DEBUG")
    
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="ghost_tracer", env="POSTGRES_DB")
    postgres_user: str = Field(default="ghost_tracer", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_log_topic: str = Field(default="ghost_tracer-logs", env="KAFKA_LOG_TOPIC")
    kafka_consumer_group: str = Field(default="ghost_tracer-consumers", env="KAFKA_CONSUMER_GROUP")
    
    @property
    def kafka_servers_list(self) -> List[str]:
        if isinstance(self.kafka_bootstrap_servers, str):
            return [s.strip() for s in self.kafka_bootstrap_servers.split(',')]
        return self.kafka_bootstrap_servers
    
    vector_store_type: VectorStore = Field(default=VectorStore.QDRANT, env="VECTOR_STORE_TYPE")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    lancedb_path: str = Field(default="./data/lancedb", env="LANCEDB_PATH")
    
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    ollama_host: str = Field(default="localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    ollama_model: str = Field(default="mistral:7b", env="OLLAMA_MODEL")
    
    hf_model: str = Field(default="microsoft/DialoGPT-medium", env="HF_MODEL")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    
    max_log_batch_size: int = Field(default=100, env="MAX_LOG_BATCH_SIZE")
    log_processing_interval: int = Field(default=5, env="LOG_PROCESSING_INTERVAL")
    trace_correlation_window: int = Field(default=300, env="TRACE_CORRELATION_WINDOW")
    
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    summarization_chunk_size: int = Field(default=1000, env="SUMMARIZATION_CHUNK_SIZE")
    
    data_retention_days: int = Field(default=30, env="DATA_RETENTION_DAYS")
    max_vector_collection_size: int = Field(default=1000000, env="MAX_VECTOR_COLLECTION_SIZE")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


def get_settings() -> Settings:
    return settings 
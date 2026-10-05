from pydantic_settings import BaseSettings, SettingsConfigDict


"""
.env
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=mistral
COLLECTION_NAME=docs
"""


class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    ollama_host: str
    embedding_model: str
    llm_model: str
    collection_name: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()

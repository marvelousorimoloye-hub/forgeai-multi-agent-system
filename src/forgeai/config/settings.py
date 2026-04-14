from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file at the very beginning
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # LLM Settings
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 8192

    # API Keys
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Vector DB
    chroma_persist_directory: str = "data/vector_store"

    # Context Engineering
    compression_ratio_target: float = 0.15

    # Monitoring
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "forgeai-research"
    langsmith_tracing: bool = True


# Global settings instance
settings = Settings()
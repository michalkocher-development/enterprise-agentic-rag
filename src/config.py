"""Moduł konfiguracji aplikacji — zarządzanie zmiennymi środowiskowymi i profilami sprzętowymi."""

from functools import lru_cache
from typing import Optional
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Główne ustawienia projektu AI Engineering Lab."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Gemini
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")

    # LangSmith Observability
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="ai-engineering-lab", alias="LANGCHAIN_PROJECT"
    )

    # Hardware & PyTorch Settings
    device: str = Field(default="cuda", alias="DEVICE")
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base", alias="RERANKER_MODEL_NAME"
    )
    reranker_fp16: bool = Field(default=True, alias="RERANKER_FP16")

    # Local AWS / LocalStack
    aws_access_key_id: str = Field(default="test", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="test", alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="eu-central-1", alias="AWS_DEFAULT_REGION")
    aws_endpoint_url: str = Field(
        default="http://localhost:4566", alias="AWS_ENDPOINT_URL"
    )
    s3_bucket_name: str = Field(
        default="ai-engineering-artifacts", alias="S3_BUCKET_NAME"
    )

    def get_resolved_device(self) -> str:
        """Zwraca aktywne urządzenie obliczeniowe PyTorch z bezpiecznym fallbackiem."""
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Zwraca cache'owaną instancję konfiguracji aplikacji."""
    return AppSettings()

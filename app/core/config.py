"""
Centralized application configuration.

Loaded once at import time and shared across the app via the `settings`
singleton. Values are pulled from environment variables / .env file so the
same image can be promoted across dev -> staging -> prod without rebuilding.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    service_name: str = "llmops-platform"

    # LLM Provider
    openai_api_key: str = ""
    openai_base_url: str | None = None  # set to use Groq/OpenRouter/Ollama instead of OpenAI
    default_model: str = "gpt-4o-mini"
    model_timeout_seconds: int = 30

    # Postgres
    database_url: str = "postgresql+asyncpg://llmops:llmops_pass@localhost:5432/llmops"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "llmops-platform"

    # Cost tracking
    cost_per_1k_prompt_tokens: float = 0.00015
    cost_per_1k_completion_tokens: float = 0.0006


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class ConfigParse:
    """Configuration parsed from environment variables (see .env.example)."""

    def __init__(self) -> None:
        self.embedding_provider: Literal["local", "openai"] = os.getenv("EMBEDDING_PROVIDER", "local")
        if self.embedding_provider not in ("local", "openai"):
            raise ValueError(
                "EMBEDDING_PROVIDER must be 'local' or 'openai', "
                f"got: {self.embedding_provider!r}"
            )
        self.embedding_model_name: str = self._required("EMBEDDING_MODEL_NAME")
        self.embedding_base_url: str | None = os.getenv("EMBEDDING_BASE_URL")
        self.embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY")

        self.db_path: str = os.getenv("DB_PATH", "db")

        self.llm_model: str = self._required("LLM_MODEL")
        self.llm_base_url: str | None = os.getenv("LLM_BASE_URL")
        self.llm_api_key: str | None = os.getenv("LLM_API_KEY")

    @staticmethod
    def _required(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise ValueError(f"{name} is required but not set")
        return value

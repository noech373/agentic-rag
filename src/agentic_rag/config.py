"""Application settings, read from environment variables and the `.env` file.

Each group reads the variables sharing its prefix (e.g. `OPENAI_API_KEY`). The names match
what the SDKs and the Postgres Docker image expect, so a single `.env` configures everything.
"""

from functools import lru_cache

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvSettings(BaseSettings):
    # extra="ignore": the shared .env also holds variables meant for the other groups.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class OpenAISettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr
    chat_model: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


class PostgresSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str
    password: SecretStr
    db: str


class LangfuseSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    public_key: str | None = None
    secret_key: SecretStr | None = None
    base_url: str = "https://cloud.langfuse.com"

    @property
    def enabled(self) -> bool:
        """Tracing is optional: it is on only when both keys are set."""
        return self.public_key is not None and self.secret_key is not None


class Settings(BaseModel):
    openai: OpenAISettings
    postgres: PostgresSettings
    langfuse: LangfuseSettings


@lru_cache
def get_settings() -> Settings:
    """Load and validate all settings once. Fails at startup if a required value is missing."""
    # Required fields come from the environment, which pyright cannot see: it reports them as
    # missing constructor arguments, hence the targeted ignores.
    return Settings(
        openai=OpenAISettings(),  # pyright: ignore[reportCallIssue]
        postgres=PostgresSettings(),  # pyright: ignore[reportCallIssue]
        langfuse=LangfuseSettings(),
    )

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.config import LangfuseSettings, OpenAISettings

_SETTINGS_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests must not depend on the developer's shell variables or `.env` file."""
    monkeypatch.chdir(tmp_path)  # no .env file in this empty directory
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_reads_prefixed_env_vars_and_applies_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "some-chat-model")

    settings = OpenAISettings()  # pyright: ignore[reportCallIssue]  (values come from env)

    assert settings.api_key.get_secret_value() == "sk-test"
    assert settings.chat_model == "some-chat-model"
    assert settings.embedding_model == "text-embedding-3-small"


def test_missing_required_value_fails_fast() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        OpenAISettings()  # pyright: ignore[reportCallIssue]  (missing on purpose)


def test_langfuse_tracing_is_disabled_without_keys() -> None:
    assert not LangfuseSettings().enabled


def test_langfuse_tracing_is_disabled_with_the_empty_keys_of_the_example_env(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("LANGFUSE_PUBLIC_KEY=\nLANGFUSE_SECRET_KEY=\n")

    assert not LangfuseSettings().enabled


def test_langfuse_tracing_is_enabled_with_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    assert LangfuseSettings().enabled

"""`cp .env.example .env` must leave a fresh clone in offline mode, and the template must document every variable read.

Only `.env.example` is read here (never a real `.env`). It is parsed with python-dotenv, the same parser `config.py` uses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

from deepdive.common import config, embeddings

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
SRC = ROOT / "src" / "deepdive"


def apply_env_file(path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    """Make the process environment look like a shell that just loaded `path` on a machine with nothing else set."""
    for name in ("OPENAI_API_KEY", "DEEPDIVE_OFFLINE"):  # conftest sets DEEPDIVE_OFFLINE=1 for every test; undo that here
        monkeypatch.delenv(name, raising=False)
    values = dotenv_values(path)
    for name, value in values.items():
        if value is not None:  # a bare `NAME` line sets nothing, as in load_dotenv
            monkeypatch.setenv(name, value)
    return values


def test_copying_env_example_to_env_stays_offline(monkeypatch):
    apply_env_file(ENV_EXAMPLE, monkeypatch)
    assert config.is_offline(), "`cp .env.example .env` must not switch the playground to live mode"
    assert config.mode_label() == "offline"
    assert embeddings._offline(), "Studio's embedding index must be offline too"


def test_the_old_placeholder_key_would_have_gone_live(monkeypatch, tmp_path):
    """Guards the check itself: the template this replaced (a fake non-empty key) is detected as live."""
    old = tmp_path / "old.env"
    old.write_text("OPENAI_API_KEY=your-openai-api-key-here\nOPENAI_MODEL=gpt-6-luna\n")
    apply_env_file(old, monkeypatch)
    assert not config.is_offline()


def test_no_api_key_in_the_template_carries_a_value():
    secrets = {k: v for k, v in dotenv_values(ENV_EXAMPLE).items() if k.endswith("_API_KEY") and v}
    assert secrets == {}, f"leave API keys empty in the template: {sorted(secrets)}"
    assert "OPENAI_API_KEY" in dotenv_values(ENV_EXAMPLE), "keep an explicit, empty OPENAI_API_KEY= line"


def test_the_template_says_how_to_stay_offline():
    text = ENV_EXAMPLE.read_text()
    assert "Leave empty for offline mode" in text


def test_every_environment_variable_the_code_reads_is_documented_in_the_template():
    read = set()
    for path in SRC.rglob("*.py"):
        read |= set(re.findall(r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z0-9_]+)["']""", path.read_text()))
        read |= set(re.findall(r"""os\.environ\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\]""", path.read_text()))
    assert {"OPENAI_API_KEY", "DEEPDIVE_OFFLINE", "EMBEDDING_MODEL", "DECISIONS_TIMEOUT", "DEEPDIVE_DATA_DIR"} <= read, read
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", ENV_EXAMPLE.read_text(), flags=re.MULTILINE))
    assert read - documented == set(), f"read by src/ but missing from .env.example: {sorted(read - documented)}"


def test_documented_optional_settings_are_commented_out_so_the_defaults_apply():
    active = set(dotenv_values(ENV_EXAMPLE))
    for name in ("DECISIONS_TIMEOUT", "EMBEDDING_MODEL", "DEEPDIVE_DATA_DIR", "DEEPDIVE_OFFLINE", "OPENAI_MODEL_FAST"):
        assert name not in active, f"{name} would override its default as soon as the file is copied"

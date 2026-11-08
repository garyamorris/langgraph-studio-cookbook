"""Environment-driven configuration shared by every example.

Nothing here talks to the network. Values are read lazily (on each call) so tests and
Studio runs can flip environment variables without re-importing modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present. Existing environment variables win (override=False),
# so `export OPENAI_API_KEY=...` in your shell always takes precedence.
load_dotenv()

_TRUTHY = {"1", "true", "yes", "on"}

# <repo>/src/deepdive/common/config.py -> parents[3] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Directory holding the bundled datasets (override with DEEPDIVE_DATA_DIR)."""
    return Path(os.getenv("DEEPDIVE_DATA_DIR") or REPO_ROOT / "data")


def is_offline() -> bool:
    """True when examples should use deterministic stand-in models instead of OpenAI.

    Offline is on when ``DEEPDIVE_OFFLINE`` is truthy, or when no ``OPENAI_API_KEY`` is
    set (so a fresh clone still runs end to end).
    """
    if os.getenv("DEEPDIVE_OFFLINE", "").strip().lower() in _TRUTHY:
        return True
    return not os.getenv("OPENAI_API_KEY", "").strip()


def mode_label() -> str:
    """'offline' or 'live'; examples stamp this into their output so it is obvious in Studio."""
    return "offline" if is_offline() else "live"


def openai_model() -> str:
    """Default chat model for all examples."""
    return os.getenv("OPENAI_MODEL", "gpt-6-luna")


def fast_model() -> str:
    """Cheaper model for high-volume nodes; falls back to the default model."""
    return os.getenv("OPENAI_MODEL_FAST") or openai_model()


def decisions_model() -> str:
    """Model for client.decisions.create (the endpoint currently supports only gpt-6-luna)."""
    return os.getenv("DECISIONS_MODEL", "gpt-6-luna")

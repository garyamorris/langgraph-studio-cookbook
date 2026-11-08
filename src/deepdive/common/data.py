"""Loaders for the bundled datasets in ``data/``.

All loaders are lazy and cached, and they return *copies*, so nodes may mutate what they get.
Do not call these at import time: LangGraph's dev server forbids blocking file I/O on its event
loop while graphs are being loaded; sync nodes run in worker threads where this is fine.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from deepdive.common.config import data_dir


@lru_cache(maxsize=None)
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir() / name)


def read_csv(name: str) -> pd.DataFrame:
    """Read ``data/<name>`` (e.g. ``"sales.csv"``) as a DataFrame (a fresh copy each call)."""
    return _csv(name).copy()


@lru_cache(maxsize=None)
def _jsonl(name: str) -> tuple[dict[str, Any], ...]:
    with (data_dir() / name).open(encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def read_jsonl(name: str) -> list[dict[str, Any]]:
    """Read ``data/<name>`` (JSON lines) as a list of dicts."""
    return [dict(r) for r in _jsonl(name)]


def db_connect(*, read_only: bool = True) -> sqlite3.Connection:
    """Open ``data/store.sqlite`` (the Lumen Home shop). Read-only by default."""
    path = (data_dir() / "store.sqlite").resolve()
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def db_schema() -> str:
    """Human/LLM-readable schema of the shop database (CREATE statements)."""
    con = db_connect()
    try:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return "\n".join(r[0] for r in rows)
    finally:
        con.close()


@dataclass(frozen=True)
class Doc:
    """One markdown knowledge-base article."""

    id: str  # file stem, e.g. "returns-and-refunds"
    title: str
    text: str  # body without the title line
    corpus: str  # "langgraph" or "support"
    audience: str = "public"  # "internal" for articles titled "(internal)": never show these to customers


@lru_cache(maxsize=None)
def _kb(corpus: str) -> tuple[Doc, ...]:
    folder: Path = data_dir() / "kb" / corpus
    docs = []
    for path in sorted(folder.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        title, _, body = raw.partition("\n")
        clean_title = title.lstrip("# ").strip()
        audience = "internal" if "(internal)" in clean_title.lower() else "public"
        docs.append(Doc(id=path.stem, title=clean_title, text=body.strip(), corpus=corpus, audience=audience))
    return tuple(docs)


def kb_docs(corpus: str) -> list[Doc]:
    """All articles of a knowledge base: ``"langgraph"`` (handbook) or ``"support"`` (Lumen Home policies)."""
    return list(_kb(corpus))

"""Embeddings for the LangGraph Store's semantic search (long-term memory).

``embed`` is referenced from ``langgraph.json`` (``store.index.embed``), so Studio's built-in store supports
``store.search(namespace, query="...")`` out of the box. Items must have a ``text`` field (``fields: ["text"]``).

* live    (OPENAI_API_KEY set, not offline): OpenAI ``text-embedding-3-small`` reduced to ``DIMS`` dimensions.
* offline: a deterministic hashed bag-of-words vector. It ranks by *lexical* overlap (not true semantics), which is
  enough to exercise the whole memory flow for free. The dimension is identical in both modes, so one
  ``langgraph.json`` works for both.

Keep this module free of heavy imports: the dev server loads it by file path.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

DIMS = 256
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "my", "i", "we", "you", "for", "with", "that", "this", "at", "be", "are"}


def _offline() -> bool:
    if os.getenv("DEEPDIVE_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return not os.getenv("OPENAI_API_KEY", "").strip()


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks if t not in _STOP]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # unigrams + bigrams


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec[h % DIMS] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts to ``DIMS``-dimensional vectors (live OpenAI, or hashed when offline)."""
    if _offline():
        return [_hash_embed(t) for t in texts]
    import openai

    resp = openai.OpenAI().embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), input=texts, dimensions=DIMS
    )
    return [d.embedding for d in resp.data]


def make_store():
    """An ``InMemoryStore`` with the same semantic index Studio uses - for tests and scripts."""
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore(index={"dims": DIMS, "embed": embed, "fields": ["text"]})

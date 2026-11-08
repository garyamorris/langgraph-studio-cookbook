"""A tiny dependency-free BM25 retriever over the markdown knowledge bases.

Why not embeddings? So every RAG example runs offline, deterministically and for free. The
retriever is behind a two-method interface (``search``), so swapping in an embeddings-backed vector
store later changes only ``get_retriever``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from deepdive.common.data import Doc, kb_docs

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "has", "have", "how", "i", "if", "in",
    "is", "it", "my", "of", "on", "or", "so", "that", "the", "their", "this", "to", "was", "what", "when", "with", "you",
    "your", "we", "our", "any", "not", "no", "will", "which", "does", "should",
    # conversational filler that is rare in documents but carries no topic ("how long does X take?")
    "long", "take", "takes", "get", "got", "much", "many", "need", "want", "please", "help", "know", "like", "would",
    "could", "also", "just", "still", "been", "me", "us", "am", "im", "i'm", "there", "about", "into", "than", "then",
}


def _stem(tok: str) -> str:
    """Strip a plural 's' so 'refunds' matches 'refund' (deliberately minimal)."""
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")) else tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9\-']*", text.lower())
    return [_stem(t) for t in toks if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float


class BM25Retriever:
    """Okapi BM25 over whole articles (they are short enough to be a single chunk)."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        # Titles are repeated so a title match counts more than a body match.
        self._tfs = [Counter(tokenize(f"{d.title} {d.title} {d.text}")) for d in docs]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokenize(query)
        scored: list[Hit] = []
        for doc, tf, ln in zip(self.docs, self._tfs, self._lens):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self._avg or 1)))
            if s > 0:
                scored.append(Hit(doc, s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


@lru_cache(maxsize=None)
def get_retriever(corpus: str, audience: str = "public") -> BM25Retriever:
    """Retriever for ``"langgraph"`` (the LangGraph handbook) or ``"support"`` (Lumen Home policies).

    ``audience="public"`` (default) excludes internal-only articles such as the refund approval matrix, so a
    customer-facing node can never quote them. Internal tooling (e.g. refund approval) passes ``audience="all"``.
    """
    docs = [d for d in kb_docs(corpus) if audience == "all" or d.audience == audience]
    return BM25Retriever(docs)

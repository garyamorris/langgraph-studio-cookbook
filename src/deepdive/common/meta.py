"""Metadata every example module declares as ``META``.

``scripts/gen_docs.py`` reads this (plus the module docstring and the compiled graph) to
generate ``docs/CATALOG.md`` and one page per example, so docs can never drift from code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal[
    "Core workflows",
    "Agents & tools",
    "Multi-agent",
    "RAG & knowledge",
    "Human-in-the-loop & persistence",
    "Memory & streaming",
    "Decisions API",
    "Runtime & resilience",
]

CATEGORY_ORDER: list[str] = [
    "Core workflows",
    "Agents & tools",
    "Multi-agent",
    "RAG & knowledge",
    "Memory & streaming",
    "Human-in-the-loop & persistence",
    "Runtime & resilience",
    "Decisions API",
]


@dataclass(frozen=True)
class ExampleMeta:
    """Describes one example for the generated catalog."""

    id: str  # must equal the key in langgraph.json and the module file name
    title: str  # short human title
    category: Category
    problem: str  # ONE sentence: the real-world problem this graph solves
    features: list[str] = field(default_factory=list)  # LangGraph features it demonstrates
    models: list[str] = field(default_factory=lambda: ["OpenAI chat"])
    datasets: list[str] = field(default_factory=list)  # files under data/
    difficulty: Literal["starter", "intermediate", "advanced"] = "intermediate"

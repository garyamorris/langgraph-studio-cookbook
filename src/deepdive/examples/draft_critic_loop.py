"""Launch announcement writer with a generate-and-critique loop that always terminates.

PROBLEM
    Marketing must publish a product-launch announcement under hard constraints: at most 120 words, three specific
    facts mentioned word for word (price, warranty, return window), and none of the claims legal has banned ("best",
    "guaranteed", "#1", "revolutionary" ...). It also has to read well. A model's first draft often breaks a rule,
    and asking a model to check its own work is unreliable for rules that are easy to count: models miscount words and
    overlook a banned word they just wrote.

DESIGN
    Two critics with different jobs, run side by side on every draft:

      * ``check_rules``    deterministic code: word count, required phrases present verbatim (case-insensitive),
                           banned phrases absent. It cannot be talked out of a verdict. Matching required facts as
                           exact phrases is strict on purpose: "two-year warranty" does not satisfy "24-month
                           warranty", and the revision prompt names the phrase that is missing.
      * ``critique_draft`` an LLM editor that scores tone and clarity (1-10) and gives concrete edits. It is told NOT to
                           judge the rules above, because code already does.

    ``judge`` joins them, records the attempt, and decides: approved (no rule violations AND editor score >= 7), out of
    attempts (``iteration >= max_iterations``), or revise. The revision prompt contains the rule failures spelled out
    ("Too long: 143 words, the limit is 120") plus the editor's notes, so the generator fixes what is actually wrong.

    Two decisions keep the loop honest. (1) The cap is a counter in state with an input (``max_iterations``, default
    3), because LangGraph's own recursion limit is 10007 steps and would let a stubborn critic run up a large bill
    first. (2) When the cap is reached the graph returns the BEST attempt, ranked by fewest rule violations and then by
    editor score, not simply the last one: revisions can make a draft worse. The result is labelled
    ``status="best_effort"`` with the unresolved problems listed, so nobody mistakes it for an approved draft.
    Every attempt stays in an append-only ``history`` so you can see how the draft evolved.

GRAPH
    START -> init -> write_draft --+--> check_rules ------+--> judge --(passed, or out of attempts)--> finalize -> END
                         ^         +--> critique_draft ---+      |
                         |                                      |
                         +----------------(revise)--------------+

LANGGRAPH FEATURES
    * a cycle (``judge`` -> ``write_draft``) guarded by an explicit counter in state
    * ``add_conditional_edges`` for the loop / stop decision
    * parallel nodes joined with ``add_edge([a, b], "judge")`` (``judge`` waits for both critics)
    * an append-only ``operator.add`` reducer for the attempt history, reset with ``Overwrite([])`` at the start of a run
    * separate ``input_schema`` / ``output_schema``

MODELS AND DATA
    OpenAI chat model (plain text for the writer, structured output for the editor). No dataset is read: the default
    facts are real facts from the bundled data ($149 is the Lumen Therm price in ``sales.csv`` after the May cut;
    the 24-month thermostat warranty and 30-day returns are in ``kb/support``).
    Offline, the stand-in writer echoes the prompt and the stand-in editor gives a fixed score, so you will see the
    loop run to its cap and return a ``best_effort`` result: that is the cap behaviour working, not a failure.

TRY IT IN STUDIO
    Input ``{"product": "Lumen Therm", "required_facts": ["$149", "24-month warranty", "30-day returns"]}`` (add
    ``"max_iterations": 2`` or ``"max_words": 40`` to make it harder). Open the ``history`` field of the output to read
    every attempt with its rule violations and score. To watch the cap, give it a contradiction: use
    ``"required_facts": ["$149", "best-in-class"]``. "best" is a banned claim, so no draft can satisfy both rules; the
    loop stops after ``max_iterations`` attempts and returns the best one as ``status="best_effort"``, with ``unmet``
    saying what is still wrong.

EXTEND IT
    Add a ``banned_phrases`` input that extends the built-in list. Replace the editor score with a Decisions API
    ``score`` question so the pass threshold works on a calibrated probability. Add ``interrupt()`` when the loop ends
    as ``best_effort`` so a person can choose between attempts.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm

META = ExampleMeta(
    id="draft_critic_loop",
    title="Draft, critique, revise (bounded)",
    category="Core workflows",
    problem="Produce a launch announcement that satisfies hard rules (length, required facts, banned claims) and a "
    "quality bar, revising until it passes or an explicit attempt cap returns the best draft, flagged as best-effort.",
    features=["cycle with explicit cap", "conditional edges", "parallel critics + join edge", "append-only reducer", "input/output schemas"],
    models=["OpenAI chat"],
    datasets=[],
    difficulty="starter",
)

EXAMPLE_INPUT = {"product": "Lumen Therm", "required_facts": ["$149", "24-month warranty", "30-day returns"]}

DEFAULT_MAX_WORDS = 120
DEFAULT_MAX_ITERATIONS = 3
MAX_ITERATIONS_LIMIT = 10
PASS_SCORE = 7  # editor score (1-10) needed, in addition to zero rule violations

# label -> pattern. Word boundaries so "bestow" or "guaranteeing a refund policy" style text is judged by the pattern, not substring luck.
BANNED_CLAIMS: dict[str, re.Pattern[str]] = {
    "best": re.compile(r"\bbest\b", re.IGNORECASE),
    "guaranteed": re.compile(r"\bguarantee[ds]?\b|\bguaranteeing\b", re.IGNORECASE),
    "#1 / number one": re.compile(r"#\s?1\b|\bnumber[- ]one\b", re.IGNORECASE),
    "revolutionary": re.compile(r"\brevolution\w*", re.IGNORECASE),
    "unbeatable": re.compile(r"\bunbeatable\b", re.IGNORECASE),
    "world's first": re.compile(r"\bworld['’]?s[- ]first\b|\bworld[- ]first\b", re.IGNORECASE),
    "risk-free": re.compile(r"\brisk[- ]free\b", re.IGNORECASE),
    "never fails": re.compile(r"\bnever fails?\b", re.IGNORECASE),
}


# --------------------------------------------------------------------------- schemas
class Critique(BaseModel):
    """The editor's verdict on tone and clarity."""

    score: int = Field(description="1 to 10. 7 or more means ready to publish as far as tone and clarity go")
    feedback: list[str] = Field(description="Up to 3 concrete edits, each one sentence. Empty if the score is 9 or 10")


class DraftIn(TypedDict, total=False):
    product: str
    required_facts: list[str]  # phrases that must appear word for word, e.g. "$149"
    max_words: int  # default 120
    max_iterations: int  # default 3, at most 10


class DraftOut(TypedDict):
    status: Literal["approved", "best_effort"]
    draft: str
    word_count: int
    iterations: int
    unmet: list[str]
    history: list[dict]
    mode: str


class DraftState(DraftIn, total=False):
    iteration: int  # drafts written so far: THE loop counter
    draft: str
    violations: list[dict]
    critique: dict
    passed: bool
    history: Annotated[list[dict], operator.add]  # append-only: one record per attempt
    status: str
    word_count: int
    iterations: int
    unmet: list[str]
    mode: str


# --------------------------------------------------------------------------- deterministic checker (pure, unit-tested)
def word_count(text: str) -> int:
    return len(text.split())


def check_draft(draft: str, required_facts: list[str], max_words: int) -> list[dict]:
    """Every rule the draft breaks, as ``{"rule": ..., "message": ...}`` dicts (empty list = compliant)."""
    if not draft.strip():
        return [{"rule": "empty", "message": "The draft is empty."}]
    problems = []
    n = word_count(draft)
    if n > max_words:
        problems.append({"rule": "length", "message": f"Too long: {n} words; the limit is {max_words}. Cut at least {n - max_words} words."})
    lowered = draft.lower()
    for fact in required_facts:
        if fact.lower() not in lowered:
            problems.append({"rule": "missing_fact", "message": f'Missing required fact: "{fact}" must appear exactly as written.'})
    for label, pattern in BANNED_CLAIMS.items():
        if (m := pattern.search(draft)) is not None:
            problems.append({"rule": "banned_claim", "message": f'Banned claim: "{m.group(0)}" ({label}). Remove it or state a checkable fact instead.'})
    return problems


def attempt_rank(attempt: dict) -> tuple[int, int]:
    """Higher is better: fewest rule violations first, then the editor's score."""
    return (-len(attempt["violations"]), attempt["score"])


def pick_best(history: list[dict]) -> dict:
    """The best attempt; on a tie the later one wins (it has seen more feedback)."""
    best = history[0]
    for attempt in history[1:]:
        if attempt_rank(attempt) >= attempt_rank(best):
            best = attempt
    return best


# --------------------------------------------------------------------------- nodes
def init(state: DraftState) -> dict:
    """Validate and normalise the input; start the counter and the history."""
    product = (state.get("product") or "").strip()
    if not product:
        raise ValueError("Provide a product, e.g. 'Lumen Therm'.")
    facts = [f.strip() for f in state.get("required_facts") or [] if isinstance(f, str) and f.strip()]
    max_words = DEFAULT_MAX_WORDS if state.get("max_words") is None else state["max_words"]
    if not isinstance(max_words, int) or isinstance(max_words, bool) or not 10 <= max_words <= 400:
        raise ValueError(f"max_words must be a whole number between 10 and 400; got {max_words!r}.")
    cap = DEFAULT_MAX_ITERATIONS if state.get("max_iterations") is None else state["max_iterations"]
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= MAX_ITERATIONS_LIMIT:
        raise ValueError(f"max_iterations must be a whole number between 1 and {MAX_ITERATIONS_LIMIT}; got {cap!r}.")
    return {
        "product": product,
        "required_facts": facts,
        "max_words": max_words,
        "max_iterations": cap,
        "iteration": 0,
        "history": Overwrite([]),  # fresh run: forget attempts left by a previous run on this thread
    }


WRITER_PROMPT = (
    "You write short product-launch announcements for Lumen Home, a smart-home hardware company. Hard rules: at most "
    "{max_words} words; include each required fact exactly as written; never use superlatives or unprovable claims "
    "(banned: {banned}). Tone: confident, factual, warm; say what "
    "the product does for the customer. Plain text only: no title, no hashtags, no quotation marks around the text. "
    "Output only the announcement."
)


def write_draft(state: DraftState) -> dict:
    """Generator. First pass writes from the brief; later passes revise the previous draft against the listed problems."""
    n = state["iteration"] + 1
    facts = "\n".join(f"- {f}" for f in state["required_facts"]) or "- (none)"
    brief = f"Product: {state['product']}\nRequired facts (use word for word):\n{facts}"
    if n == 1:
        user = f"{brief}\n\nWrite the announcement."
    else:
        problems = [v["message"] for v in state["violations"]] + [f"Editor: {note}" for note in state["critique"]["feedback"]]
        if state["critique"]["score"] < PASS_SCORE and not state["critique"]["feedback"]:
            problems.append(f"Editor: the draft scored {state['critique']['score']}/10 on tone and clarity; make it more concrete and less promotional.")
        user = (
            f"{brief}\n\nRevise the draft below. Fix every problem listed, keep what already works, and stay within "
            f"{state['max_words']} words.\n\nPROBLEMS TO FIX\n" + "\n".join(f"- {p}" for p in problems)
            + f"\n\nCURRENT DRAFT\n{state['draft']}"
        )
    msg = get_llm().invoke([SystemMessage(WRITER_PROMPT.format(max_words=state["max_words"], banned=", ".join(BANNED_CLAIMS))), HumanMessage(user)])
    return {"draft": str(msg.text).strip().strip('"'), "iteration": n}


def check_rules(state: DraftState) -> dict:
    """Critic 1 (code): the rules that can be counted."""
    return {"violations": check_draft(state["draft"], state["required_facts"], state["max_words"])}


EDITOR_PROMPT = (
    "You are the copy editor for Lumen Home launch announcements. Score the draft from 1 to 10 for tone and clarity: "
    "is it confident and factual rather than hype, is it easy to read, does it tell the customer a concrete benefit? "
    "7 or more means ready to publish. Give up to 3 specific edits (for example 'Replace the opening question with "
    "a statement of what the thermostat does'). Do NOT comment on word count, required facts or banned words: "
    "automated checks handle those."
)


def critique_draft(state: DraftState) -> dict:
    """Critic 2 (LLM): tone and clarity only."""
    c = structured_llm(Critique, "fast").invoke(
        [SystemMessage(EDITOR_PROMPT), HumanMessage(f"Product: {state['product']}\n\nDRAFT\n{state['draft']}")]
    )
    return {"critique": {"score": max(1, min(10, int(c.score))), "feedback": [f.strip() for f in c.feedback if f.strip()][:3]}}


def judge(state: DraftState) -> dict:
    """Join: record this attempt (append-only), then decide whether it passes."""
    v, c = state["violations"], state["critique"]
    passed = not v and c["score"] >= PASS_SCORE
    attempt = {
        "attempt": state["iteration"],
        "draft": state["draft"],
        "word_count": word_count(state["draft"]),
        "violations": v,
        "score": c["score"],
        "feedback": c["feedback"],
        "passed": passed,
    }
    return {"history": [attempt], "passed": passed}


def route(state: DraftState) -> Literal["finalize", "write_draft"]:
    """Stop when approved or out of attempts; otherwise go round again."""
    if state["passed"] or state["iteration"] >= state["max_iterations"]:
        return "finalize"
    return "write_draft"


def finalize(state: DraftState) -> dict:
    best = pick_best(state["history"])
    unmet = [] if best["passed"] else [x["message"] for x in best["violations"]]
    if not best["passed"] and best["score"] < PASS_SCORE:
        unmet.append(f"Editor score {best['score']}/10 is below the bar of {PASS_SCORE}.")
    if not best["passed"]:
        unmet += [f"Editor: {note}" for note in best["feedback"]]
    return {
        "status": "approved" if best["passed"] else "best_effort",
        "draft": best["draft"],
        "word_count": best["word_count"],
        "iterations": state["iteration"],
        "unmet": unmet,
        "mode": mode_label(),
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(DraftState, input_schema=DraftIn, output_schema=DraftOut)
    g.add_node("init", init)
    g.add_node("write_draft", write_draft)
    g.add_node("check_rules", check_rules)
    g.add_node("critique_draft", critique_draft)
    g.add_node("judge", judge)
    g.add_node("finalize", finalize)
    g.add_edge(START, "init")
    g.add_edge("init", "write_draft")
    g.add_edge("write_draft", "check_rules")  # two critics run in parallel on the same draft
    g.add_edge("write_draft", "critique_draft")
    g.add_edge(["check_rules", "critique_draft"], "judge")  # join: judge runs once both have finished
    g.add_conditional_edges("judge", route, ["finalize", "write_draft"])
    g.add_edge("finalize", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

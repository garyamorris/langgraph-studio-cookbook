"""Live customer-voice digest: progress, partial results and token streaming.

PROBLEM
    The weekly customer-voice digest is slow: load the reviews, run three analysis stages, then an LLM writes the
    summary. Behind a bare spinner users assume it has hung, refresh, and resubmit - doubling the cost and the wait.
    The fix is not a faster graph but a graph that *talks while it works*: a progress bar that only moves forward,
    the rating histogram on screen the moment it exists, and the digest typing itself out token by token.

DESIGN
    Streaming is a contract between the graph and a front-end, so it is designed, not sprinkled:

      * Every stage reports through ``get_stream_writer()`` with two typed events:
        ``{"type": "progress", "stage", "pct", "message"}`` and ``{"type": "partial", "name", "data"}``. ``pct`` only
        ever increases and the last event of a successful run is 100, so a UI can render it without special cases.
      * Which parts are deterministic code and which are model: ratings and risk flags are plain code (a safety
        signal must not depend on a model's mood). Themes are proposed by a model but *verified* by code: it must cite
        review ids, code drops ids it did not load, recomputes counts and average ratings itself, and falls back to a
        keyword tally if nothing survives. Only the final write-up is free text, and only that text is streamed to
        the user.
      * The analysis stages live in a **subgraph** so the front-end can show "analysis: step 2 of 3" and Studio can
        drill into it. Streaming with ``subgraphs=True`` labels every event with the subgraph it came from.
      * Internal model calls emit tokens too. ``stream_mode="messages"`` reports *every* chat-model call made inside
        the graph (checked offline with a plain call in a non-UI node; the theme classifier's structured-output call is
        expected to stream its raw JSON the same way live, which I could not check offline), so a naive UI could print
        classifier JSON into the digest. ``iter_ui_events`` below keeps only tokens from the ``write_digest`` node. (Alternative on the producer side: tag the internal call
        with ``.with_config(tags=["nostream"])`` and LangGraph never emits its tokens; checked offline. The consumer
        filter is used here because the UI owner should not need to edit every node.)
      * The module ships the front-end half: ``iter_ui_events`` turns the raw multi-mode stream into small
        normalised events, and ``python -m deepdive.examples.live_digest_stream`` is a terminal "UI" built on it.

GRAPH
    START -> load -> [ analysis: ratings -> theme_finder -> risk_scan ] -> write_digest -> finish -> END
                      (a subgraph; its three steps stream under namespace "analysis:<id>")

STREAM MODES (``graph.stream(input, config, stream_mode=...)``)
    +--------------+---------------------------------+-----------------------------+------------------------------+
    | mode         | yields                          | use it for                  | in this example              |
    +--------------+---------------------------------+-----------------------------+------------------------------+
    | values       | the full state after each step  | "what does the state look   | not used (it would resend    |
    |              |                                 | like now" (can be large)    | all the reviews each step)   |
    | updates      | {node: what that node returned} | per-stage "done" ticks      | ``stage_done`` events        |
    | messages     | (AIMessageChunk, metadata) per  | the typing effect; metadata | ``token`` events, filtered   |
    |              | token of EVERY model call       | has ``langgraph_node``      | to ``write_digest``          |
    | custom       | whatever a node passes to       | progress bars, partial      | ``progress`` and ``partial`` |
    |              | ``get_stream_writer()``         | results, domain events      | events                       |
    | debug        | checkpoints plus task start /   | tracing and post-mortems    | not used (``tasks`` and      |
    |              | finish with full config         |                             | ``checkpoints`` are subsets) |
    +--------------+---------------------------------+-----------------------------+------------------------------+
    | subgraphs=   | prefixes every item with a      | knowing WHERE an event came | ``path`` / ``ns`` on every   |
    | True         | namespace tuple                 | from; without it subgraph   | normalised event             |
    +--------------+---------------------------------+-----------------------------+------------------------------+
    Verified shape in langgraph 1.2.20: with a LIST of modes and ``subgraphs=True`` every item is a 3-tuple
    ``(namespace, mode, payload)``. ``namespace`` is ``()`` for the parent graph and ``("analysis:<task id>",)`` inside
    the subgraph (one entry per nesting level). The namespace is present even with no checkpointer. With a single
    mode and no subgraphs you get the bare payload; with ``version="v2"`` you get dicts ``{"type", "ns", "data"}``
    instead (this module uses the default tuples).
    What you see in Studio: the run view lights up nodes as they finish, the ``analysis`` node expands into its three
    steps, and the final state appears in the state panel. Studio is not a stream-event inspector (how it renders
    custom events depends on its version - not verified here); to see the exact events a front-end receives, run the
    command under TRY IT IN STUDIO.

LANGGRAPH FEATURES
    * ``get_stream_writer()`` custom events - progress and partial results are domain events no built-in mode has
    * ``stream_mode=["updates", "messages", "custom"]`` together, and ``subgraphs=True`` for namespaces
    * a compiled subgraph added directly as a node (shared state keys), so its inner steps are streamable
    * token streaming from a plain ``llm.invoke`` inside a node: LangGraph streams under the hood when a
      ``messages`` consumer is attached, so the node needs no streaming code
    * separate ``input_schema`` / ``output_schema`` and a structured-output call whose result is verified by code

MODELS AND DATA
    OpenAI chat: a ``fast`` model for the theme classifier (structured output), the default model for the digest.
    Data: ``product_reviews.csv`` (320 reviews of the eight Lumen Home products; newest ``max_reviews`` are used).
    Simulated: nothing about the streaming is simulated, but offline mode streams a canned reply word by word and
    returns instantly, so ``pace_s`` (sleep per stage) exists purely to make the progress visible in a demo.

TRY IT IN STUDIO
    Pick a ready-made case from the "Ready-made case" dropdown and press Submit (a blank form on a new thread runs the
    first one). Watch ``load``, then the ``analysis`` subgraph (expand it for its three steps), ``write_digest`` and
    ``finish``; open ``risk_scan`` to see which reviews were flagged. ``pace_s`` makes each stage take visibly long:
      * Plug 4-pack (newest 40, 0.5 s per stage, about 3 s in all): the standard demo; the burning plug review R-0315
        is flagged.
      * Lock (all 22): three flags, a jammed lock and two lock-outs. Cam Pro (all 57): the largest run, with a
        privacy flag on R-0317.
      * Hub (newest 30): a counterfeit listing is flagged. Bell (newest 5, the minimum): the shortest run, and no risk
        language, so ``risk_scan`` reports nothing flagged. These four add no pause; give them a ``pace_s`` to slow any of them.
    Or paste your own input, for example ``{"product": "Lumen Plug 4-pack", "max_reviews": 40, "pace_s": 0.5}``; try
    ``"Nope"`` for the clean error. To see the event stream itself, run in a terminal (add ``--pace 0.3`` to slow it
    down)::

        .venv/bin/python -m deepdive.examples.live_digest_stream "Lumen Plug 4-pack"

EXTEND IT
    Handle the resubmit itself: a run submitted while another is active on the same thread can be rejected, queued or
    allowed to interrupt the old one (the server's multitask options) - decide which fits your UI. Add a cancel check
    between stages so an abandoned run stops spending model calls. Emit the digest's sections as separate ``partial``
    events by splitting the write-up into one model call per section.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import read_csv
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="live_digest_stream",
    title="Live customer-voice digest (streaming)",
    category="Memory & streaming",
    problem="Stop users resubmitting a slow report by streaming typed progress, partial results and the digest's "
    "tokens, with subgraph-aware events a front-end can consume directly.",
    features=[
        "get_stream_writer custom events",
        "multi-mode streaming (updates/messages/custom)",
        "subgraph streaming with namespaces",
        "token streaming with node filtering",
        "subgraph as a node",
    ],
    models=["OpenAI chat"],
    datasets=["product_reviews.csv"],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"product": "Lumen Plug 4-pack", "max_reviews": 40}

# Ready-made cases for Studio's dropdown (see deepdive.common.scenarios). Every product is in data/product_reviews.csv.
# The counts and the risk flags in the labels come from the deterministic code (``scan_risks``, the newest-N window),
# never from the model, and a test recomputes them. ``pace_s`` None means no pause. The first case is EXAMPLE_INPUT plus a
# half-second pause per stage, so the first thing a person sees is a progress bar they can watch. Only this case pauses:
# every pause is real sleep, also in the tests that run each case.
CASES: dict[str, dict] = {
    "Lumen Plug 4-pack, newest 40, 0.5 s per stage -> watch the stages tick; burning plug R-0315 flagged": {
        "product": "Lumen Plug 4-pack", "max_reviews": 40, "pace_s": 0.5},
    "Lumen Lock, all 22 reviews -> risk_scan flags 3: jammed lock R-0319 and lock-outs R-0167, R-0262": {
        "product": "Lumen Lock", "max_reviews": None, "pace_s": None},
    "Lumen Cam Pro, newest 200 (all 57) -> the largest run; R-0317 flagged for privacy": {
        "product": "Lumen Cam Pro", "max_reviews": 200, "pace_s": None},
    "Lumen Hub, newest 30 -> counterfeit review R-0320 (12th newest) is flagged; no demo pause": {
        "product": "Lumen Hub", "max_reviews": 30, "pace_s": None},
    "Lumen Bell, newest 5 (the minimum) -> the shortest run; risk_scan flags nothing": {
        "product": "Lumen Bell", "max_reviews": 5, "pace_s": None},
}
SCENARIO = scenario_field(CASES)

DEFAULT_MAX_REVIEWS = 40
MAX_REVIEWS_LIMIT = 200
MIN_REVIEWS = 5
MAX_PACE_S = 2.0
MIN_THEME_SUPPORT = 2  # a "theme" mentioned by one review is an anecdote
MAX_THEMES = 5

TOKEN_NODE = "write_digest"  # the only node whose model tokens a UI should show
STREAM_MODES = ["updates", "messages", "custom"]


# --------------------------------------------------------------------------- schemas
class DigestIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made case; replaces the three fields below
    product: str  # exact product name, e.g. "Lumen Plug 4-pack" (case-insensitive)
    max_reviews: int  # newest N reviews to analyse (5-200, default 40)
    pace_s: float  # demo knob: sleep this long inside each stage (0-2 s, default 0)


class DigestOut(TypedDict):
    product: str
    n_reviews: int
    rating_stats: dict
    themes: list[dict]
    risks: list[dict]
    digest: str
    digest_cites_all_risks: bool
    mode: str


class DigestState(DigestIn, total=False):
    reviews: list[dict]
    rating_stats: dict
    themes: list[dict]
    risks: list[dict]
    digest: str
    n_reviews: int
    digest_cites_all_risks: bool
    mode: str


class AnalysisState(TypedDict, total=False):
    """What the analysis subgraph reads and writes. Keys are shared with ``DigestState`` so it can be a plain node."""

    reviews: list[dict]
    pace_s: float
    rating_stats: dict
    themes: list[dict]
    risks: list[dict]


class Theme(BaseModel):
    """One recurring topic in the reviews, as proposed by the model."""

    name: str = Field(description="2-4 word label, for example 'App crashes' or 'Easy setup'")
    summary: str = Field(description="One sentence on what customers say, using only the reviews provided")
    review_ids: list[str] = Field(description="Ids of the reviews that mention this topic, copied exactly (for example R-0202)")


class ThemeReport(BaseModel):
    themes: list[Theme] = Field(description="3 to 5 themes, most frequently mentioned first")


# --------------------------------------------------------------------------- streaming helpers
def progress(stage: str, pct: int, message: str) -> None:
    """Emit a ``progress`` event on the ``custom`` stream (a no-op when nobody is listening)."""
    get_stream_writer()({"type": "progress", "stage": stage, "pct": pct, "message": message})


def partial(name: str, data: Any) -> None:
    """Emit a ``partial`` result as soon as it exists, so the UI can render it before the run ends."""
    get_stream_writer()({"type": "partial", "name": name, "data": data})


def _pace(state: dict) -> None:
    """Demo knob: make a stage take visibly long. Zero (the default) means no delay."""
    try:
        delay = float(state.get("pace_s") or 0.0)
    except (TypeError, ValueError):
        delay = 0.0
    if delay > 0:
        time.sleep(min(delay, MAX_PACE_S))


# --------------------------------------------------------------------------- nodes: load
def load(state: DigestState) -> dict:
    """Resolve a picked ready-made case, validate the product and load its newest reviews as plain dicts (JSON-safe state)."""
    progress("load", 5, "Loading reviews")
    picked = expand(
        state,
        CASES,
        has_input=lambda s: bool((s.get("product") or "").strip())
        or s.get("max_reviews") is not None
        or s.get("pace_s") is not None,
    )
    state = {**state, **picked}  # a picked case replaces all three input fields (None means "default")
    _pace(state)
    df = read_csv("product_reviews.csv")
    products = sorted(df["product"].unique())
    wanted = (state.get("product") or "").strip().lower()
    match = [p for p in products if p.lower() == wanted]
    if not match:
        raise ValueError(f"Unknown product {state.get('product')!r}; choose one of: {', '.join(products)}")
    try:
        limit = int(state.get("max_reviews") or DEFAULT_MAX_REVIEWS)
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_REVIEWS
    limit = max(MIN_REVIEWS, min(limit, MAX_REVIEWS_LIMIT))
    rows = df[df["product"] == match[0]].sort_values(["date", "review_id"], ascending=False).head(limit)
    reviews = [
        {
            "review_id": str(r.review_id),
            "rating": int(r.rating),
            "title": str(r.title),
            "text": str(r.text),
            "date": str(r.date),
            "verified": bool(r.verified_purchase),
            "helpful_votes": int(r.helpful_votes),
        }
        for r in rows.itertuples()
    ]
    progress("load", 15, f"Loaded {len(reviews)} reviews ({reviews[-1]['date']} to {reviews[0]['date']})")
    return {**picked, "product": match[0], "reviews": reviews}


# --------------------------------------------------------------------------- nodes: analysis subgraph
def ratings_step(state: AnalysisState) -> dict:
    """Deterministic rating statistics. Emits the histogram as a partial result the moment it is ready."""
    progress("ratings", 20, "Computing rating statistics")
    _pace(state)
    reviews = state["reviews"]  # newest first
    ratings = [r["rating"] for r in reviews]
    n = len(ratings)
    half = n // 2
    recent, older = ratings[:half], ratings[half:]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0  # noqa: E731
    histogram = [{"rating": k, "count": ratings.count(k)} for k in range(1, 6)]
    stats = {
        "n": n,
        "mean": mean(ratings),
        "histogram": histogram,
        "share_4_or_5": round(sum(1 for x in ratings if x >= 4) / n, 3),
        "verified_share": round(sum(1 for r in reviews if r["verified"]) / n, 3),
        "recent_mean": mean(recent),
        "older_mean": mean(older),
        "trend": round(mean(recent) - mean(older), 2),  # positive: newer reviews are happier
    }
    partial("rating_histogram", {"mean": stats["mean"], "n": n, "histogram": histogram})
    progress("ratings", 30, f"Mean rating {stats['mean']} across {n} reviews")
    return {"rating_stats": stats}


# Keyword baseline: always computed, used as the model's hint and as the fallback if the model's themes are unusable.
KEYWORD_THEMES: list[tuple[str, str]] = [
    ("Connectivity and dropouts", r"offline|disconnect|lost connection|buffer|hub range|arrive \d+ seconds late"),
    ("App quality", r"app crashes|app is slow|scheduling is confusing|ignored my away mode|app is clear"),
    ("Setup and pairing", r"setup took|pairs with|installed it|easy to place|needed three resets|wiring guide"),
    ("Build and fit", r"bulky|adhesive|ghosting|flicker|chime|keypad|feels solid|rock solid|looks great"),
    ("Battery and cold", r"battery|cold"),
    ("Value and price", r"cheap|pricey|price|subscription|value|paid for itself|heating bill"),
    ("Alerts and detection", r"alerts?|tamper|motion|night mode|picture quality|two-way audio"),
]


def _theme_record(name: str, summary: str, ids: list[str], by_id: dict[str, dict], source: str) -> dict:
    """Counts, averages and sentiment are computed here from the ratings - never taken from the model."""
    ratings = [by_id[i]["rating"] for i in ids]
    avg = round(sum(ratings) / len(ratings), 2)
    sentiment = "praise" if avg >= 4.0 else "complaint" if avg <= 3.0 else "mixed"
    return {"name": name, "summary": summary, "n": len(ids), "avg_rating": avg, "sentiment": sentiment,
            "review_ids": ids[:8], "source": source}


def keyword_themes(reviews: list[dict]) -> list[dict]:
    by_id = {r["review_id"]: r for r in reviews}
    out = []
    for name, pattern in KEYWORD_THEMES:
        ids = [r["review_id"] for r in reviews if re.search(pattern, f"{r['title']} {r['text']}", re.I)]
        if len(ids) >= MIN_THEME_SUPPORT:
            out.append(_theme_record(name, f"{len(ids)} reviews mention this topic (keyword match).", ids, by_id, "keywords"))
    return sorted(out, key=lambda t: -t["n"])[:MAX_THEMES]


def verify_themes(report: ThemeReport, reviews: list[dict]) -> list[dict]:
    """Keep only model themes that cite loaded reviews; recompute everything numeric from the data."""
    by_id = {r["review_id"]: r for r in reviews}
    out = []
    for t in report.themes:
        ids = list(dict.fromkeys(i.strip() for i in t.review_ids if i.strip() in by_id))  # valid, de-duplicated
        if len(ids) >= MIN_THEME_SUPPORT:
            out.append(_theme_record(t.name.strip(), t.summary.strip(), ids, by_id, "model"))
    return sorted(out, key=lambda t: -t["n"])[:MAX_THEMES]


THEME_PROMPT = (
    "You analyse customer reviews of one smart-home product. Identify 3 to 5 recurring themes - things several "
    "different reviews mention - and for each give a short label, a one-sentence summary that only uses what the "
    "reviews say, and the ids of the reviews that mention it, copied exactly as written. Include both praise and "
    "complaints. Do not invent review ids, do not count reviews yourself, and do not make a theme out of a single review."
)


def themes_step(state: AnalysisState) -> dict:
    """Model proposes themes, code verifies them (valid ids, recomputed counts), keywords are the fallback."""
    progress("theme_finder", 35, "Finding recurring themes")
    _pace(state)
    reviews = state["reviews"]
    listing = "\n".join(f"{r['review_id']} | {r['rating']} stars | {r['title']} | {r['text']}" for r in reviews)
    report = structured_llm(ThemeReport, "fast").invoke([SystemMessage(THEME_PROMPT), HumanMessage(listing)])
    found = verify_themes(report, reviews)
    source_note = "model themes verified against the reviews"
    if not found:
        found = keyword_themes(reviews)
        source_note = "keyword fallback (no verifiable model themes)"
    partial("themes", found)
    progress("theme_finder", 55, f"{len(found)} themes - {source_note}")
    return {"themes": found}


RISK_PATTERNS: dict[str, str] = {
    "fire_or_injury": r"\b(fire|burn\w*|scorch\w*|smok\w*|spark\w*|overheat\w*|melt\w*|injur\w*|shock\w*|hot to the touch|became very hot)\b",
    "legal": r"\b(lawyer|lawsuit|sue|legal|legally|illegal|solicitor|chargeback|counterfeit)\b",
    "privacy": r"\b(privacy|another household'?s|someone else'?s|strangers?)\b",
    "lockout": r"\b(locked out|couldn't get out|can't get out)\b",
}


def scan_risks(reviews: list[dict]) -> list[dict]:
    """Deterministic risk scan: a safety or legal signal must never depend on a model's mood."""
    out = []
    for r in reviews:
        text = f"{r['title']} {r['text']}"
        cats = [name for name, pat in RISK_PATTERNS.items() if re.search(pat, text, re.I)]
        if cats:
            out.append({"review_id": r["review_id"], "categories": cats, "rating": r["rating"], "date": r["date"],
                        "snippet": r["text"][:140]})
    return sorted(out, key=lambda x: (x["rating"], x["review_id"]))


def risks_step(state: AnalysisState) -> dict:
    progress("risk_scan", 60, "Scanning for safety, legal and privacy language")
    _pace(state)
    found = scan_risks(state["reviews"])
    partial("risk_flags", found)
    progress("risk_scan", 70, f"{len(found)} review(s) flagged" if found else "No risk language found")
    return {"risks": found}


def build_analysis_graph():
    """The subgraph: three sequential steps, compiled without a checkpointer (it inherits the parent's)."""
    g = StateGraph(AnalysisState)
    # Node names must differ from state keys (a node called "themes" cannot coexist with a "themes" channel).
    g.add_node("ratings", ratings_step)
    g.add_node("theme_finder", themes_step)
    g.add_node("risk_scan", risks_step)
    g.add_edge(START, "ratings")
    g.add_edge("ratings", "theme_finder")
    g.add_edge("theme_finder", "risk_scan")
    g.add_edge("risk_scan", END)
    return g.compile()


# --------------------------------------------------------------------------- nodes: write-up and finish
DIGEST_PROMPT = (
    "You write the weekly customer-voice digest for the Lumen Home product team. Use ONLY the facts provided: do "
    "not add numbers, review ids or quotes that are not in them. Format: one headline sentence; then three short "
    "labelled sections - 'What customers like', 'What hurts', 'Risks to escalate'. If risk flags are provided, the "
    "Risks section must list every flagged review id with a few words on why, before anything else; if there are "
    "none, say so in one line. Plain text, under 180 words, no markdown tables."
)


def _facts_text(state: DigestState) -> str:
    s = state["rating_stats"]
    lines = [
        f"Product: {state['product']}",
        f"Reviews analysed: {s['n']} (newest first). Mean rating {s['mean']}; share of 4-5 stars {s['share_4_or_5']:.0%}; "
        f"newer half {s['recent_mean']} vs older half {s['older_mean']} (trend {s['trend']:+}).",
        "Histogram: " + ", ".join(f"{h['rating']} stars: {h['count']}" for h in s["histogram"]),
        "Themes:",
    ]
    lines += [f"- {t['name']} ({t['sentiment']}, {t['n']} reviews, avg {t['avg_rating']}): {t['summary']} ids {', '.join(t['review_ids'])}"
              for t in state["themes"]] or ["- none found"]
    lines.append("Risk flags:")
    lines += [f"- {r['review_id']} ({', '.join(r['categories'])}, {r['rating']} stars): {r['snippet']}" for r in state["risks"]] or ["- none"]
    return "\n".join(lines)


def write_digest(state: DigestState) -> dict:
    """The only free-text step - and the only node whose tokens the UI shows. A plain ``invoke`` streams under the hood."""
    progress("write_digest", 75, "Writing the digest")
    _pace(state)
    reply = get_llm().invoke([SystemMessage(DIGEST_PROMPT), HumanMessage(_facts_text(state))])
    progress("write_digest", 95, "Digest written")
    return {"digest": str(reply.text)}


def finish(state: DigestState) -> dict:
    """Deterministic last check: does the digest mention every flagged review id? Then close the progress bar."""
    digest = state.get("digest", "")
    cites_all = all(r["review_id"] in digest for r in state["risks"])
    progress("finish", 100, "Done")
    return {"n_reviews": len(state["reviews"]), "digest_cites_all_risks": cites_all, "mode": mode_label()}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(DigestState, input_schema=DigestIn, output_schema=DigestOut)
    g.add_node("load", load)
    g.add_node("analysis", build_analysis_graph())  # shared keys with DigestState -> add the compiled graph directly
    g.add_node("write_digest", write_digest)
    g.add_node("finish", finish)
    g.add_edge(START, "load")
    g.add_edge("load", "analysis")
    g.add_edge("analysis", "write_digest")
    g.add_edge("write_digest", "finish")
    g.add_edge("finish", END)
    return g.compile(**compile_kwargs)


graph = build_graph()


# --------------------------------------------------------------------------- the front-end half
def normalise_stream(raw: Iterable[tuple[tuple[str, ...], str, Any]]) -> Iterator[dict]:
    """Turn raw ``(namespace, mode, payload)`` items into the small events a UI consumes.

    Pure function (no graph needed), so it is unit-tested with hand-made items. Every event carries ``ns`` (raw
    namespace) and ``path`` (the same with task ids stripped, e.g. ``("analysis",)``; ``()`` for the parent graph).

      {"kind": "progress",   "stage", "pct", "message"}   from custom events of type "progress"
      {"kind": "partial",    "name", "data"}              from custom events of type "partial"
      {"kind": "stage_done", "node", "keys"}              from "updates": a node finished and wrote these state keys
      {"kind": "token",      "node", "text"}              from "messages", only for the top-level ``write_digest`` node
    """
    for ns, mode, payload in raw:
        ns = tuple(ns)
        base = {"ns": ns, "path": tuple(seg.split(":", 1)[0] for seg in ns)}
        if mode == "custom":
            if isinstance(payload, dict) and payload.get("type") == "progress":
                yield {"kind": "progress", **base, "stage": payload["stage"], "pct": payload["pct"], "message": payload["message"]}
            elif isinstance(payload, dict) and payload.get("type") == "partial":
                yield {"kind": "partial", **base, "name": payload["name"], "data": payload["data"]}
        elif mode == "updates":
            for node, update in payload.items():
                if node.startswith("__"):  # __interrupt__ and friends are not stages
                    continue
                keys = sorted(update) if isinstance(update, dict) else []
                yield {"kind": "stage_done", **base, "node": node, "keys": keys}
        elif mode == "messages":
            chunk, metadata = payload
            text = str(chunk.text)
            # Every model call in the graph streams here; keep only the user-facing node (and not a same-named
            # node nested in a subgraph).
            if not ns and metadata.get("langgraph_node") == TOKEN_NODE and text:
                yield {"kind": "token", **base, "node": TOKEN_NODE, "text": text}


def iter_ui_events(graph: Any, input: dict, config: dict | None = None) -> Iterator[dict]:
    """Run ``graph`` and yield normalised UI events as they happen.

    One ``graph.stream`` call with three modes and ``subgraphs=True`` feeds ``normalise_stream``. A failed run ends
    with ``{"kind": "error", ...}`` instead of an exception, so a UI can show it rather than spin forever.
    """
    try:
        yield from normalise_stream(graph.stream(input, config, stream_mode=STREAM_MODES, subgraphs=True))
    except Exception as e:  # noqa: BLE001 - surfaced to the UI as an event
        yield {"kind": "error", "ns": (), "path": (), "error_type": type(e).__name__, "message": str(e)}


def _render(ev: dict, out: Any = None) -> None:
    """Terminal rendering of one UI event (the CLI's whole 'front-end')."""
    out = out or sys.stdout  # looked up at call time so redirected stdout (tests, pipes) works
    where = " > ".join(ev["path"]) + " > " if ev["path"] else ""
    kind = ev["kind"]
    if kind == "progress":
        print(f"[{ev['pct']:3d}%] {where}{ev['stage']:<13} {ev['message']}", file=out, flush=True)
    elif kind == "partial":
        print(f"       partial: {where}{ev['name']}", file=out, flush=True)
        if ev["name"] == "rating_histogram":
            peak = max((h["count"] for h in ev["data"]["histogram"]), default=1) or 1
            for h in ev["data"]["histogram"]:
                print(f"         {h['rating']} stars {'#' * round(20 * h['count'] / peak):<20} {h['count']}", file=out, flush=True)
        elif ev["name"] == "risk_flags":
            for r in ev["data"]:
                print(f"         {r['review_id']} {','.join(r['categories'])}: {r['snippet'][:70]}", file=out, flush=True)
        elif ev["name"] == "themes":
            for t in ev["data"]:
                print(f"         {t['name']} ({t['sentiment']}, {t['n']} reviews)", file=out, flush=True)
    elif kind == "stage_done":
        print(f"       done: {where}{ev['node']}", file=out, flush=True)
    elif kind == "token":
        print(ev["text"], end="", file=out, flush=True)
    elif kind == "error":
        print(f"\nERROR {ev['error_type']}: {ev['message']}", file=out, flush=True)


def main(argv: list[str] | None = None) -> int:
    """``python -m deepdive.examples.live_digest_stream "Lumen Plug 4-pack" [--max-reviews N] [--pace S]``"""
    ap = argparse.ArgumentParser(description="Stream a customer-voice digest to the terminal.")
    ap.add_argument("product", nargs="?", default=EXAMPLE_INPUT["product"])
    ap.add_argument("--max-reviews", type=int, default=EXAMPLE_INPUT["max_reviews"])
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to pause inside each stage (demo)")
    args = ap.parse_args(argv)
    print(f"Streaming digest for {args.product!r} ({mode_label()} mode)\n")
    failed = False
    in_digest = False
    for ev in iter_ui_events(graph, {"product": args.product, "max_reviews": args.max_reviews, "pace_s": args.pace}):
        if ev["kind"] == "token" and not in_digest:
            print("\n--- digest ---", flush=True)
            in_digest = True
        if ev["kind"] != "token" and in_digest:
            print("\n", flush=True)
            in_digest = False
        failed = failed or ev["kind"] == "error"
        _render(ev)
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

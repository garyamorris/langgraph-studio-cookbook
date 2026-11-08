"""Hierarchical research supervisor: two specialist teams and a fact-checked writer.

PROBLEM
    An analyst wants a one-page brief on one of the (fictional) listed companies. A single agent holding every tool
    does shallow work, lets narrative bleed into the numbers ("shares are volatile" with no volatility figure), and
    cannot show where any sentence came from. The brief is only useful if each claim can be traced to a fact.

DESIGN
    Three layers, each doing the job it is good at:

      * ``supervisor`` (model, structured output ``Decision``) reads the analyst's question and chooses the next
        worker plus a one-line instruction. It never touches data. Code, not the model, enforces the rules that must
        always hold: a worker runs at most once, the brief cannot be skipped, and a hop budget (``max_hops``) forces
        the writer to run when it is spent. (With once-only workers a runaway loop is already impossible; the budget
        also lets you ask for a quick brief: ``max_hops=1`` consults one team.)
      * Two **team subgraphs** with their own *private* state, wrapped by mapping functions so the parent never sees
        their internals:
          - ``market_team``: deterministic price analytics (5/20/60-day returns, 20-day realised volatility, drawdown
            from the 60-day closing high, price vs the 50-day average), then a model that writes risk notes from
            those numbers only - each note must cite the metric ids it relies on.
          - ``news_team``: fetch the ticker's recent headlines, classify each one (model, structured), tally the
            sentiment in code, then a model proposes catalysts, each citing headline ids.
        Numbers are code; interpretation is a model; every model sentence must point at ids code can check.
      * ``writer`` (model, structured ``BriefDraft``) turns the numbered fact list into bullets, each with ``fact_ids``.
        Code drops any bullet whose ids are missing or unknown, renders the page itself, and lists numbers in a bullet
        that do not appear in the cited facts. If nothing survives it prints the raw fact sheet instead of an
        unverifiable draft. The text is therefore never free-form model prose.

    Failure modes this structure prevents: a team being asked twice (double cost), the supervisor ending the run
    before a brief exists, an unknown ticker burning model calls (it is rejected before any model runs), and
    plausible-but-unsourced sentences reaching the page.

GRAPH
    START -> intake --(unknown ticker)--> END
                |
                v
            supervisor --Command(goto)--> market_team  [compute_metrics -> write_risk_notes]
              ^    |                      news_team    [fetch_headlines -> classify_sentiment -> tally_sentiment
              |    |                                    -> extract_catalysts]
              |    |                      writer       [drafts bullets with fact ids; code audits and renders]
              |    +--(writer already ran)--> END
              +-------- every worker reports back to the supervisor --------+
    market_team and news_team are subgraphs with private state; ``supervisor`` is the only node that decides routing.

LANGGRAPH FEATURES
    * ``Command(goto=..., update=...)`` from the supervisor: routing and bookkeeping in one step, no conditional edges
    * subgraphs with private state schemas, called from wrapper nodes that map state in and out (the parent keeps only
      each team's facts); streamable with ``subgraphs=True``, where each team's steps appear under its own namespace
    * a reducer (``operator.add``) on ``done`` and ``trail`` so each hop appends to the run's audit log; ``intake`` resets
      both with ``Overwrite([])``, because Studio reuses the thread and a leftover ``writer`` in ``done`` would end the next run at once
    * explicit loop bounds: once-only worker flags plus ``max_hops``, both enforced in code and tested
    * separate ``input_schema`` / ``output_schema`` and OpenAI-strict structured output on every model decision

MODELS AND DATA
    OpenAI chat (fast tier for the supervisor and classifier, default for notes, catalysts and the writer). Data:
    ``companies.csv``, ``prices.csv`` (160 trading days to 2026-10-30) and ``news_headlines.csv``. Everything is
    fictional: the output says "not investment advice" and offers no forecasts. The headline ``label`` column is NOT
    used by the graph (it is the model's answer key; see EXTEND IT). Offline, the placeholder model output cites ids that
    do not exist, so the checks drop it and the run shows the code-only parts: metrics, the headline list and the
    fact sheet (the supervisor's route is arbitrary but stable).

TRY IT IN STUDIO
    Input ``{"ticker": "ACMR"}``: watch ``supervisor`` hand off market_team, news_team, writer, then end; open the
    ``trail`` in the output to see each decision (and whether the model or a rule made it). Each team is invoked from a
    wrapper node, so the static graph shows one node per team; their inner steps appear in the run trace and in
    ``stream(..., subgraphs=True)`` under the namespaces ``market_team:<id>`` / ``news_team:<id>`` (not checked in Studio
    itself; the tests print them). Then try:
      * ``{"ticker": "CBLT", "question": "Is the recent price weakness explained by the news?"}``
      * ``{"ticker": "HELO", "max_hops": 1}`` - the budget is spent after one team, so the writer is forced and the
        brief says the other team was not consulted
      * ``{"ticker": "NOPE"}`` - a clean ``error`` and no model call
    Tickers: ACMR BORE CBLT DLPH EMBR FTHM GRNT HELO.

EXTEND IT
    Score the news classifier against the dataset's ``label`` column (the dataset marks ~15% of headlines as tricky).
    Run both teams in parallel with ``Send`` when the question needs both, keeping the supervisor for ambiguous
    questions. Add a ``reviewer`` worker that re-checks the draft's numbers with a second model before rendering.
"""

from __future__ import annotations

import math
import operator
import re
import statistics
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import read_csv
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import structured_llm

META = ExampleMeta(
    id="research_supervisor",
    title="Hierarchical research supervisor",
    category="Multi-agent",
    problem="Produce a one-page company brief in which numbers come from code, interpretation from specialist teams, "
    "and every sentence is traceable to a fact, with a supervisor that cannot loop or skip the brief.",
    features=[
        "hierarchical supervisor",
        "Command routing",
        "subgraphs with private state",
        "reducers (audit trail)",
        "loop bounds",
        "subgraph streaming",
        "structured output",
    ],
    models=["OpenAI chat"],
    datasets=["companies.csv", "prices.csv", "news_headlines.csv"],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"ticker": "ACMR", "question": "Give me a one-page brief: price action, risk and news."}

TEAMS = ("market_team", "news_team")
DEFAULT_QUESTION = "Give me a one-page brief covering price action, risk and news."
DEFAULT_MAX_HOPS = 5  # the normal path needs 3 routing decisions (market, news, writer)
MAX_HOPS_LIMIT = 8
MAX_HEADLINES = 12
MAX_INSTRUCTION_CHARS = 200
DISCLAIMER = "Fictional data for a demonstration. Not investment advice."
MIN_PRICE_ROWS = 61  # 60-day return needs 61 closes
TRADING_DAYS = 252


# --------------------------------------------------------------------------- schemas
class BriefIn(TypedDict, total=False):
    ticker: str  # ACMR BORE CBLT DLPH EMBR FTHM GRNT HELO
    question: str  # what the analyst wants to know; the supervisor uses it to pick teams
    max_hops: int  # supervisor routing budget before the writer is forced (default 5, max 8)


class BriefOut(TypedDict, total=False):
    ticker: str
    company: str
    brief: str  # markdown, rendered by code
    facts: list[dict]  # every fact the writer could cite: {id, kind, text, basis}
    route: list[str]  # where the supervisor sent the run, in order
    trail: list[dict]  # one record per supervisor decision (who decided, what was overridden)
    audit: dict  # which bullets were kept / dropped / have unverified numbers
    error: str
    disclaimer: str
    mode: str


class BriefState(BriefIn, total=False):
    profile: dict
    company: str
    instruction: str
    hops: int
    done: Annotated[list[str], operator.add]
    trail: Annotated[list[dict], operator.add]
    market: dict
    news: dict
    facts: list[dict]
    brief: str
    audit: dict
    route: list[str]
    error: str
    disclaimer: str
    mode: str


class Decision(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["market_team", "news_team", "writer", "FINISH"] = Field(
        description="market_team = price analytics and risk notes; news_team = headlines, sentiment, catalysts; "
        "writer = assemble the brief (needs at least one team report); FINISH only after the writer has run"
    )
    instruction: str = Field(description="One line (max 25 words) telling the chosen worker what to emphasise; empty for FINISH")


class RiskNote(BaseModel):
    text: str = Field(description="One sentence about price risk. Use numbers exactly as written in the facts.")
    basis: list[str] = Field(description="Ids of the metric facts this note relies on, for example ['M5', 'M6']")


class RiskNotes(BaseModel):
    notes: list[RiskNote] = Field(description="At most 3 notes")


class Labelled(BaseModel):
    headline_id: str = Field(description="The headline id exactly as given, for example H-0020")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="From a shareholder's point of view; 'mixed' when the headline contains both good and bad news"
    )


class SentimentBatch(BaseModel):
    items: list[Labelled] = Field(description="One entry per headline")


class Catalyst(BaseModel):
    text: str = Field(description="One sentence naming an event that could move the stock. No predictions.")
    headline_ids: list[str] = Field(description="Ids of the headlines that report it, for example ['H-0020']")


class CatalystList(BaseModel):
    catalysts: list[Catalyst] = Field(description="At most 3 catalysts")


class BriefBullet(BaseModel):
    section: Literal["price_and_risk", "news_and_sentiment", "catalysts"]
    text: str = Field(description="One sentence. Copy numbers exactly from the cited facts.")
    fact_ids: list[str] = Field(description="Ids of the facts that support the sentence, for example ['M2', 'N1']")


class BriefDraft(BaseModel):
    headline: str = Field(description="One sentence summarising the situation, using only the facts")
    headline_fact_ids: list[str] = Field(description="Ids of the facts behind the headline")
    bullets: list[BriefBullet] = Field(description="4 to 8 bullets")


# --------------------------------------------------------------------------- deterministic analytics
def price_metrics(ticker: str) -> dict:
    """Price analytics from closing prices, in plain Python (tests compare it with an independent pandas version).

    * ``ret_5d`` / ``ret_20d`` / ``ret_60d``: last close / close N trading days earlier - 1
    * ``vol_20d``: sample standard deviation of the last 20 daily simple returns, annualised with sqrt(252)
    * ``high_60d`` / ``drawdown_60d``: highest close of the last 60 days; last close / that high - 1 (never positive)
    * ``sma_50`` / ``vs_sma_50``: mean of the last 50 closes; last close / that mean - 1
    """
    df = read_csv("prices.csv")
    rows = df[df["ticker"] == ticker].sort_values("date")
    closes = [float(x) for x in rows["close"]]
    if len(closes) < MIN_PRICE_ROWS:
        return {}
    last = closes[-1]
    daily = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]
    high = max(closes[-60:])
    sma = sum(closes[-50:]) / 50
    return {
        "as_of": str(rows["date"].iloc[-1]),
        "last_close": last,
        "ret_5d": last / closes[-6] - 1,
        "ret_20d": last / closes[-21] - 1,
        "ret_60d": last / closes[-61] - 1,
        "vol_20d": statistics.stdev(daily) * math.sqrt(TRADING_DAYS),
        "high_60d": high,
        "drawdown_60d": last / high - 1,
        "sma_50": sma,
        "vs_sma_50": last / sma - 1,
    }


def recent_headlines(ticker: str, limit: int = MAX_HEADLINES) -> list[dict]:
    """The ticker's newest distinct headlines (the dataset repeats some wording on different days)."""
    df = read_csv("news_headlines.csv")
    rows = df[df["ticker"] == ticker].sort_values(["date", "headline_id"], ascending=False)
    seen: set[str] = set()
    out = []
    for r in rows.itertuples():
        if r.headline in seen:
            continue
        seen.add(r.headline)
        out.append({"headline_id": str(r.headline_id), "date": str(r.date), "headline": str(r.headline)})
        if len(out) == limit:
            break
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def metric_facts(m: dict, company: dict) -> list[dict]:
    def f(i: int, text: str) -> dict:
        return {"id": f"M{i}", "kind": "metric", "text": text, "basis": []}

    return [
        f(1, f"Last close {m['last_close']:.2f} on {m['as_of']}."),
        f(2, f"5-day return {_pct(m['ret_5d'])}."),
        f(3, f"20-day return {_pct(m['ret_20d'])}."),
        f(4, f"60-day return {_pct(m['ret_60d'])}."),
        f(5, f"20-day realised volatility (annualised, from daily closes) {m['vol_20d'] * 100:.1f}%."),
        f(6, f"Drawdown from the 60-day closing high ({m['high_60d']:.2f}): {m['drawdown_60d'] * 100:.1f}%."),
        f(7, f"Price vs 50-day average ({m['sma_50']:.2f}): {_pct(m['vs_sma_50'])}."),
        f(8, f"Sector {company['sector']}; beta {company['beta']:.2f}."),
    ]


def tally_labels(labels: dict[str, str], headline_ids: list[str]) -> dict:
    counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    unlabelled = 0
    for hid in headline_ids:
        s = labels.get(hid)
        if s in counts:
            counts[s] += 1
        else:
            unlabelled += 1
    labelled = sum(counts.values())
    net = (counts["positive"] - counts["negative"]) / labelled if labelled else None
    return {**counts, "unlabelled": unlabelled, "labelled": labelled, "net": net}


# --------------------------------------------------------------------------- team 1: market (private state)
class MarketState(TypedDict, total=False):
    """Private to the market team. The parent never sees ``metrics`` internals, only the facts the wrapper returns."""

    ticker: str
    profile: dict
    instruction: str
    metrics: dict
    status: str
    facts: list[dict]


def compute_metrics(state: MarketState) -> dict:
    m = price_metrics(state["ticker"])
    if not m:
        return {"metrics": {}, "status": "no_price_history", "facts": []}
    return {"metrics": m, "status": "ok", "facts": metric_facts(m, state["profile"])}


RISK_NOTES_PROMPT = (
    "You are a market-risk analyst on a research desk covering FICTIONAL companies. From ONLY the numbered metric "
    "facts you are given, write at most 3 short risk notes. Each note is one sentence, copies its numbers exactly "
    "from the facts, and lists in 'basis' the ids of the facts it relies on. Do not forecast, give price targets, or "
    "use buy/sell language. If nothing in the facts is notable, return fewer notes."
)


def write_risk_notes(state: MarketState) -> dict:
    if state.get("status") != "ok":
        return {}
    facts = state["facts"]
    listing = "\n".join(f"[{f['id']}] {f['text']}" for f in facts)
    hint = state.get("instruction") or "none"
    result = structured_llm(RiskNotes, "default").invoke(
        [SystemMessage(RISK_NOTES_PROMPT), HumanMessage(f"Supervisor emphasis (a hint only): {hint}\n\nFACTS\n{listing}")]
    )
    valid = {f["id"] for f in facts}
    notes = []
    for n in result.notes[:3]:
        basis = [b for b in dict.fromkeys(n.basis) if b in valid]
        if n.text.strip() and basis and len(basis) == len(set(n.basis)):  # every cited id must exist
            notes.append({"id": f"R{len(notes) + 1}", "kind": "note", "text": n.text.strip(), "basis": basis})
    return {"facts": facts + notes}


def build_market_team():
    g = StateGraph(MarketState)
    g.add_node("compute_metrics", compute_metrics)
    g.add_node("write_risk_notes", write_risk_notes)
    g.add_edge(START, "compute_metrics")
    g.add_edge("compute_metrics", "write_risk_notes")
    g.add_edge("write_risk_notes", END)
    return g.compile()


# --------------------------------------------------------------------------- team 2: news (private state)
class NewsState(TypedDict, total=False):
    """Private to the news team: raw headlines, per-headline labels and the tally never reach the parent."""

    ticker: str
    instruction: str
    headlines: list[dict]
    labels: dict
    tally: dict
    catalysts: list[dict]
    facts: list[dict]


def fetch_headlines(state: NewsState) -> dict:
    return {"headlines": recent_headlines(state["ticker"])}


CLASSIFY_PROMPT = (
    "You classify news headlines about one FICTIONAL listed company by their likely effect on a shareholder. Return "
    "one entry per headline, using the headline_id exactly as given. Use 'mixed' when a headline contains both good "
    "and bad news (for example a missed estimate alongside a large buyback), 'neutral' for scheduling or routine "
    "announcements. Judge only the words of the headline."
)


def classify_sentiment(state: NewsState) -> dict:
    heads = state["headlines"]
    if not heads:
        return {"labels": {}}
    listing = "\n".join(f"{h['headline_id']} | {h['date']} | {h['headline']}" for h in heads)
    result = structured_llm(SentimentBatch, "fast").invoke([SystemMessage(CLASSIFY_PROMPT), HumanMessage(listing)])
    known = {h["headline_id"] for h in heads}
    return {"labels": {i.headline_id: i.sentiment for i in result.items if i.headline_id in known}}


def tally_sentiment(state: NewsState) -> dict:
    """Counts and net sentiment are arithmetic, not model output; facts N1-N3 are built here."""
    heads = state["headlines"]
    if not heads:
        return {"tally": {}, "facts": [{"id": "N1", "kind": "metric", "text": "No headlines found for this ticker.", "basis": []}]}
    t = tally_labels(state["labels"], [h["headline_id"] for h in heads])
    net_text = (f"Net sentiment {t['net']:+.2f} (positive minus negative, divided by labelled headlines)."
                if t["net"] is not None else "Net sentiment not available: no headline was labelled.")
    newest = heads[0]
    facts = [
        {"id": "N1", "kind": "metric", "basis": [],
         "text": f"{len(heads)} distinct headlines from {heads[-1]['date']} to {newest['date']}: {t['positive']} positive, "
                 f"{t['negative']} negative, {t['neutral']} neutral, {t['mixed']} mixed ({t['unlabelled']} unlabelled)."},
        {"id": "N2", "kind": "metric", "basis": [],
         "text": net_text},
        {"id": "N3", "kind": "metric", "basis": [],
         "text": f"Latest headline ({newest['date']}, {newest['headline_id']}): {newest['headline']}"},
    ]
    return {"tally": t, "facts": facts}


CATALYST_PROMPT = (
    "You are a news analyst covering a FICTIONAL company. From ONLY the headlines given, name at most 3 catalysts: "
    "events that could plausibly move the share price (contracts, recalls, earnings surprises, regulation, "
    "buybacks). One sentence each, with the ids of the headlines that report it. Do not predict price moves. "
    "Return fewer if there are fewer."
)


def extract_catalysts(state: NewsState) -> dict:
    heads = state["headlines"]
    if not heads:
        return {"catalysts": []}
    labels = state.get("labels", {})
    listing = "\n".join(f"{h['headline_id']} | {h['date']} | {labels.get(h['headline_id'], 'unlabelled')} | {h['headline']}" for h in heads)
    hint = state.get("instruction") or "none"
    result = structured_llm(CatalystList, "default").invoke(
        [SystemMessage(CATALYST_PROMPT), HumanMessage(f"Supervisor emphasis (a hint only): {hint}\n\nHEADLINES\n{listing}")]
    )
    known = {h["headline_id"] for h in heads}
    facts = list(state["facts"])
    for c in result.catalysts[:3]:
        ids = list(dict.fromkeys(c.headline_ids))
        if c.text.strip() and ids and all(i in known for i in ids):  # every cited headline must exist
            facts.append({"id": f"C{sum(f['kind'] == 'catalyst' for f in facts) + 1}", "kind": "catalyst",
                          "text": f"{c.text.strip()} (headlines {', '.join(ids)})", "basis": ids})
    return {"catalysts": [f for f in facts if f["kind"] == "catalyst"], "facts": facts}


def build_news_team():
    g = StateGraph(NewsState)
    g.add_node("fetch_headlines", fetch_headlines)
    g.add_node("classify_sentiment", classify_sentiment)
    g.add_node("tally_sentiment", tally_sentiment)
    g.add_node("extract_catalysts", extract_catalysts)
    g.add_edge(START, "fetch_headlines")
    g.add_edge("fetch_headlines", "classify_sentiment")
    g.add_edge("classify_sentiment", "tally_sentiment")
    g.add_edge("tally_sentiment", "extract_catalysts")
    g.add_edge("extract_catalysts", END)
    return g.compile()


MARKET_TEAM = build_market_team()
NEWS_TEAM = build_news_team()


def run_market_team(state: BriefState) -> dict:
    """Wrapper: parent state in, team subgraph (private state) in the middle, only the facts back out."""
    out = MARKET_TEAM.invoke(
        {"ticker": state["profile"]["ticker"], "profile": state["profile"], "instruction": state.get("instruction", "")}
    )
    return {"market": {"status": out.get("status", "ok"), "facts": out.get("facts", [])}, "done": ["market_team"]}


def run_news_team(state: BriefState) -> dict:
    out = NEWS_TEAM.invoke({"ticker": state["profile"]["ticker"], "instruction": state.get("instruction", "")})
    return {"news": {"status": "ok" if out.get("headlines") else "no_headlines", "facts": out.get("facts", [])}, "done": ["news_team"]}


# --------------------------------------------------------------------------- intake and supervisor
def fresh_run() -> dict:
    """Everything a previous run on this thread may have left behind. Studio reuses the thread, so without this the
    supervisor would see ``writer`` in ``done`` and end at once with the OLD brief. ``Overwrite`` bypasses the
    ``operator.add`` reducers; the plain keys are simply written again."""
    return {"done": Overwrite([]), "trail": Overwrite([]), "market": {}, "news": {}, "facts": [], "brief": "",
            "audit": {}, "route": [], "error": "", "company": "", "instruction": ""}


def intake(state: BriefState) -> Command[Literal["supervisor", "__end__"]]:
    """Reject an unknown ticker before any model is called; otherwise set up the run."""
    companies = read_csv("companies.csv")
    ticker = (state.get("ticker") or "").strip().upper()
    row = companies[companies["ticker"] == ticker]
    if row.empty:
        known = ", ".join(companies["ticker"])
        return Command(
            goto=END,
            update={**fresh_run(), "ticker": ticker, "error": f"Unknown ticker {ticker!r}. Available tickers: {known}.",
                    "disclaimer": DISCLAIMER, "mode": mode_label()},
        )
    r = row.iloc[0]
    try:
        max_hops = int(state.get("max_hops") or DEFAULT_MAX_HOPS)
    except (TypeError, ValueError):
        max_hops = DEFAULT_MAX_HOPS
    prices = read_csv("prices.csv")
    profile = {"ticker": ticker, "name": str(r.company), "sector": str(r.sector), "beta": float(r.beta),
               "as_of": str(prices[prices["ticker"] == ticker]["date"].max())}
    return Command(
        goto="supervisor",
        update={**fresh_run(), "ticker": ticker, "profile": profile, "company": profile["name"], "hops": 0,
                "max_hops": max(1, min(max_hops, MAX_HOPS_LIMIT)),
                "question": (state.get("question") or "").strip() or DEFAULT_QUESTION},
    )


SUPERVISOR_PROMPT = (
    "You are the research supervisor on an equity research desk covering FICTIONAL companies. You never analyse "
    "anything yourself: you decide which worker acts next.\n"
    "Workers:\n"
    "- market_team: price analytics (returns, volatility, drawdown, trend) and risk notes.\n"
    "- news_team: recent headlines, sentiment tally and catalysts.\n"
    "- writer: assembles the final one-page brief from the team reports. Call it once at least one team has reported "
    "(usually both).\n"
    "Rules: call each worker at most once; never choose a worker listed as done; choose FINISH only after the writer "
    "has run. Skip a team only if the analyst's question clearly does not need it. Give a one-line instruction "
    "(max 25 words) saying what the chosen worker should emphasise."
)


def route_decision(asked: str, done: list[str], hops: int, max_hops: int) -> tuple[str, str]:
    """Apply the rules that must always hold to what the model asked for. Returns (target, why)."""
    pending = [t for t in TEAMS if t not in done]
    if "writer" in done:
        return "FINISH", "rule: brief already written"
    if hops >= max_hops:
        return "writer", "rule: hop budget spent, writer runs anyway"
    if asked == "FINISH":
        return "writer", "rule: cannot finish before the brief is written"
    if asked in done:
        return (pending[0] if pending else "writer"), f"rule: {asked} already ran, never twice"
    if asked == "writer" and len(pending) == len(TEAMS):
        return pending[0], "rule: writer needs at least one team report"
    return asked, "model"


def _clean_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:MAX_INSTRUCTION_CHARS]


def _status_line(state: BriefState) -> str:
    def line(name: str, key: str) -> str:
        if name not in state.get("done", []):
            return f"{name}: not run"
        data = state.get(key) or {}
        return f"{name}: reported ({len(data.get('facts', []))} facts, status {data.get('status')})"

    return "; ".join([line("market_team", "market"), line("news_team", "news"),
                      "writer: done" if "writer" in state.get("done", []) else "writer: not run"])


def supervisor(state: BriefState) -> Command[Literal["market_team", "news_team", "writer", "__end__"]]:
    done, hops, max_hops = state.get("done", []), state.get("hops", 0), state.get("max_hops", DEFAULT_MAX_HOPS)
    asked, instruction = "", ""
    if "writer" not in done and hops < max_hops:  # only spend a model call when a real choice remains
        human = (
            f"Analyst's question: {state['question']}\nCompany: {state['profile']['name']} ({state['profile']['ticker']})\n"
            f"Done so far: {', '.join(done) or 'nothing'}\nReports: {_status_line(state)}\n"
            f"Routing decisions used: {hops} of {max_hops}"
        )
        d = structured_llm(Decision, "fast").invoke([SystemMessage(SUPERVISOR_PROMPT), HumanMessage(human)])
        asked, instruction = d.next, _clean_instruction(d.instruction)
    target, why = route_decision(asked, done, hops, max_hops)
    record = {"hop": hops + 1, "asked": asked or None, "routed": target, "why": why,
              "instruction": instruction if target == asked else ""}
    update: dict = {"trail": [record], "instruction": record["instruction"]}
    if target == "FINISH":
        route = [r["routed"] for r in [*state.get("trail", []), record]]
        return Command(goto=END, update={**update, "route": route, "disclaimer": DISCLAIMER, "mode": mode_label()})
    return Command(goto=target, update={**update, "hops": hops + 1})


# --------------------------------------------------------------------------- writer
SECTION_TITLES = {
    "price_and_risk": "Price and risk",
    "news_and_sentiment": "News and sentiment",
    "catalysts": "Catalysts",
}

WRITER_PROMPT = (
    "You write a one-page brief for an equity analyst about a FICTIONAL company. You are given numbered facts. "
    "Produce a one-sentence headline and 4 to 8 bullets. EVERY bullet and the headline must list in fact_ids the ids "
    "of the facts that support it, and may use only those facts. Copy numbers exactly as written in the facts. Do not "
    "forecast, give targets, or use buy/sell language. Put price and risk statements in 'price_and_risk', sentiment "
    "tallies in 'news_and_sentiment' and events in 'catalysts'. If a section has no facts, write no bullet for it."
)

_NUM = re.compile(r"\d+(?:\.\d+)?")


def audit_draft(draft: BriefDraft, facts: list[dict]) -> dict:
    """Keep only bullets whose every cited id exists; report numbers that none of the cited facts contain."""
    by_id = {f["id"]: f for f in facts}
    kept, dropped, unverified = [], [], []
    for b in draft.bullets:
        ids = list(dict.fromkeys(b.fact_ids))
        if not b.text.strip():
            dropped.append({"text": b.text, "reason": "empty"})
        elif not ids:
            dropped.append({"text": b.text, "reason": "no fact ids"})
        elif any(i not in by_id for i in ids):
            dropped.append({"text": b.text, "reason": "unknown fact id: " + ", ".join(i for i in ids if i not in by_id)})
        else:
            cited = " ".join(by_id[i]["text"] for i in ids)
            missing = sorted(set(_NUM.findall(b.text)) - set(_NUM.findall(cited)))
            if missing:
                unverified.append({"text": b.text, "numbers": missing})
            kept.append({"section": b.section, "text": b.text.strip(), "fact_ids": ids})
    head_ids = list(dict.fromkeys(draft.headline_fact_ids))
    headline_ok = bool(draft.headline.strip()) and bool(head_ids) and all(i in by_id for i in head_ids)
    return {"kept": kept, "dropped": dropped, "unverified_numbers": unverified,
            "headline": draft.headline.strip() if headline_ok else "", "headline_fact_ids": head_ids if headline_ok else []}


def render_brief(state: BriefState, audit: dict, facts: list[dict]) -> str:
    c = state["profile"]
    lines = [f"# {c['name']} ({c['ticker']}) - research brief", "",
             f"Question: {state['question']}", f"Data as of {c['as_of']}. {DISCLAIMER}", ""]
    consulted = [t for t in TEAMS if t in state.get("done", [])]
    if audit["kept"]:
        if audit["headline"]:
            lines += [f"**{audit['headline']}** [{', '.join(audit['headline_fact_ids'])}]", ""]
        for key, title in SECTION_TITLES.items():
            items = [b for b in audit["kept"] if b["section"] == key]
            if items:
                lines += [f"## {title}"] + [f"- {b['text']} [{', '.join(b['fact_ids'])}]" for b in items] + [""]
    else:
        lines += ["## Fact sheet",
                  "No model-written claim could be tied to the facts, so these are the facts themselves.", ""]
        lines += [f"- {f['text']} [{f['id']}]" for f in facts] + [""]
    skipped = [t for t in TEAMS if t not in consulted]
    if skipped:
        lines += ["Not consulted for this brief: " + ", ".join(skipped) + "."]
    return "\n".join(lines).strip() + "\n"


def writer(state: BriefState) -> dict:
    facts = [*(state.get("market") or {}).get("facts", []), *(state.get("news") or {}).get("facts", [])]
    listing = "\n".join(f"[{f['id']}] ({f['kind']}) {f['text']}" for f in facts) or "(no facts available)"
    c = state["profile"]
    draft = structured_llm(BriefDraft, "default").invoke(
        [SystemMessage(WRITER_PROMPT),
         HumanMessage(f"Company: {c['name']} ({c['ticker']}), data as of {c['as_of']}\nAnalyst's question: {state['question']}\n\nFACTS\n{listing}")]
    )
    audit = audit_draft(draft, facts)
    fallback = not audit["kept"]
    return {
        "facts": facts,
        "brief": render_brief(state, audit, facts),
        "audit": {"bullets_kept": len(audit["kept"]), "dropped": audit["dropped"],
                  "unverified_numbers": audit["unverified_numbers"], "fact_sheet_fallback": fallback},
        "done": ["writer"],
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(BriefState, input_schema=BriefIn, output_schema=BriefOut)
    g.add_node("intake", intake)
    g.add_node("supervisor", supervisor)
    g.add_node("market_team", run_market_team)
    g.add_node("news_team", run_news_team)
    g.add_node("writer", writer)
    g.add_edge(START, "intake")
    for worker in ("market_team", "news_team", "writer"):
        g.add_edge(worker, "supervisor")  # every worker reports back; the supervisor decides what happens next
    return g.compile(**compile_kwargs)


graph = build_graph()

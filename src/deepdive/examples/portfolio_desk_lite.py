"""Portfolio desk (lite): a compact trading desk whose risk committee can send single names back.

THE MARKET IS FICTIONAL. The eight companies, their prices and their headlines are synthetic (see ``data/README.md``).
Nothing in this example is investment advice, and nothing it prints says anything about a real security.

PROBLEM
    A small PM wants a daily trade blotter across eight stocks. Doing it by hand is slow and inconsistent. Handing it
    to one big prompt is worse: a model that writes position sizes can write sizes that break the risk limits, and a
    model that writes the closing note can quote a figure that is not in the blotter. The stakes are the ones a real
    desk has: a blotter that breaches a limit must never go out, and a note must never invent a number. So one graph
    needs model judgement (what to look at, how to argue both sides, how deep to cut a position) AND code guarantees
    (limits, arithmetic, verification). This is the architecture of a 24-node hedge-fund graph cut down to 11 nodes
    that run offline on the bundled data.

DESIGN
    Who decides what:

      node            model judgement                       code (the guarantees)
      --------------  ------------------------------------  ---------------------------------------------------
      cio_plan        which 3-5 names, what gross budget    validates tickers against companies.csv, clamps the
                                                            budget, falls back to a rule if the model returns junk
      ticker_worker   (Decisions API) setup, direction,      every indicator: 5d and 20d return, 20d volatility,
                      conviction probabilities              RSI14, drawdown from the 60d high, z-score vs the
                                                            50d mean, 20d average volume; the setup gate
      bull_case,      the two arguments                     drops ideas about tickers that are not on the desk
      bear_case
      arbiter         (Decisions API) tilt, tail risk,      the tail-risk override; a refused tilt fails safe
                      rigor of each case
      desk            -                                     applies the tilt, sizes: conviction x side weight /
                                                            volatility, scaled to the gross budget
      risk_committee  -                                     every limit check, which names breach, the loop counter
      resize_worker   how deep to cut one position          bounds the cut, applies it, rounds down to $100
      execution_plan  -                                     shares, child orders, participation cap, cost in bps
      desk_note       the prose                             audits every number in the prose against the blotter

    Four decisions shape the graph.

    1. The limits are code, and the desk is deliberately not limit-aware. The desk sizes to the gross budget the way a
       PM would; the committee is the independent check (separation of duties). Concentrated books therefore breach
       naturally: if the only actionable names are three technology stocks, a NET_LONG tilt at a 45% budget puts all
       45% of NAV into one sector group against a 35% limit. The limits: one name <= 15% of NAV; one sector group
       <= 35% of NAV; gross <= the budget scaled by the tilt; no SHORT in a name that research marked AVOID (the desk
       never opens one, and the committee checks anyway, because limits must not depend on upstream code being
       right). About "sector group": every company in companies.csv sits in a different sector, so a literal
       one-sector cap could never bind. The cap is applied to groups of sectors instead: technology (industrial
       tech, semiconductors, software), cyclicals (energy, renewables, financials) and defensives (healthcare,
       consumer staples).

    2. Breaches go back SELECTIVELY. The committee names only the tickers involved in a breach (one oversized name, or
       every member of an over-full group) and sends exactly those to ``resize_worker`` with ``Send``. Names that are
       fine are not touched or recomputed. A resize worker is a model call: it sees the breach, REQUIRED_SCALE (the
       largest scale that cures every limit the name is in) and MIN_SCALE (the desk never cuts more than twice what
       the limit requires) and proposes a scale. Code bounds the proposal to [MIN_SCALE, 1.0]; a model that under-cuts
       just gets another round. A position the rules forbid (a SHORT in an AVOID name) is dropped by rule, without
       asking the model.

    3. The loop provably ends. ``revision_count`` goes up each time the committee dispatches resizes, and it only
       dispatches while ``revision_count < max_revisions``. At the cap the committee trims every breaching name itself
       with the same cure scales and sets ``cap_forced``. One trim always gives a clean book: each name is scaled by
       the minimum, over the breaches it is in, of limit/actual (0 for a forbidden short); that satisfies its own cap,
       its group's cap and the gross cap together, and scaling down never raises any other figure. So there are at
       most ``max_revisions`` resize rounds plus one trim (the input is clamped to 0..5). ``execution_plan`` re-runs
       the limit check on the final share counts and refuses to build a blotter that fails it.

    4. Calibrated probabilities where they are the right tool: two kinds of Decisions API gate, each ONE call with
       several questions. Per name: ``setup_supportive`` is a gate (below 40% a LONG or SHORT pick becomes AVOID),
       ``direction`` picks the side, ``conviction`` scales the size. For the debate: ``tilt`` picks the book's tilt,
       ``tail_risk_elevated`` at 70% or more moves it one notch toward DEFENSIVE (never the other way), and
       ``bull_rigor`` / ``bear_rigor`` grade the two arguments for you to read (they do not move money). A refused
       answer fails safe (AVOID, DEFENSIVE). If research finds nothing actionable, the debate and the arbiter are
       skipped. The desk note is written by a chat model, but code extracts every number from it and compares it with
       the blotter facts; an unsupported number replaces the note with a template and lists the offending numbers.

GRAPH
    START -> cio_plan
                |  Send x N  (one per chosen name)
                v
            ticker_worker x N     analytics + one Decisions call each; merge_by_ticker reducer
                |
                v
            dossier_review ------ nothing actionable ---------------------------------------+
                |  Command(goto=[...])                                                       |
          +-----+-----+                                                                      |
          v           v                                                                      |
      bull_case   bear_case     (in parallel)                                                |
          +-----+-----+                                                                      |
                v  join: wait for both                                                       |
             arbiter            one Decisions call: tilt, tail risk, rigor                   |
                |                                                                            |
                v                                                                            |
              desk <-------------------------------------------------------------------------+
                |
                v
        +-> risk_committee ---- no breach, or revision cap reached ----> execution_plan
        |       |  Send x k  (ONLY the breaching names)                         |
        |       v                                                               v
        +--- resize_worker x k                                         desk_note -> END

LANGGRAPH FEATURES
    * ``Send`` map-reduce from ``cio_plan``: one ``ticker_worker`` per chosen name, a number only known at run time.
      Results merge through a custom reducer, ``merge_by_ticker``: a list of keyed records, so state stays
      JSON-serialisable and the merge does not depend on which worker finishes first. The reducer never forgets a
      ticker, so ``cio_plan`` starts each run with ``Overwrite([])`` on every accumulating channel (Studio reuses the thread).
    * ``Command(goto=[...])`` in ``dossier_review``: start two branches at once, or skip them.
    * parallel branches with an explicit join, ``add_edge(["bull_case", "bear_case"], "arbiter")``.
    * ``Send`` from a conditional edge for selective re-dispatch, wired back into ``risk_committee``: a cycle that
      re-runs only the breaching names, bounded by a counter in state (DESIGN 3).
    * ``input_schema`` / ``output_schema``: Studio shows three input fields and a result without the internals.
    * ``RetryPolicy`` on the two nodes that call the Decisions API: a transient network error is retried in place
      instead of failing the whole run.

MODELS AND DATA
    OpenAI chat (structured output for the CIO, the two cases and the resize proposal; plain text for the desk note)
    and the OpenAI Decisions API (one call per ticker, one for the arbiter). Data: ``prices.csv`` (160 days of OHLCV,
    to 2026-10-30), ``news_headlines.csv`` (the sentiment labels are NOT shown to any model) and ``companies.csv``.
    Without an ``OPENAI_API_KEY`` the chat stand-in returns placeholder text and the Decisions stand-in returns
    arbitrary but stable answers, so the CIO always takes its deterministic fallback and the desk note is a
    placeholder. The result carries ``mode``: "live" or "offline".

TRY IT IN STUDIO
    Input (this is ``EXAMPLE_INPUT``; it names stocks with clear trends, because the setup gate rightly stands aside
    when the picture is mixed and a no-trade day shows none of the machinery):
        {"focus_hint": "Look at ACMR, BORE, GRNT, HELO and CBLT today", "nav_usd": 10000000, "max_revisions": 2}
    With an API key, one development run of this input went as follows (model output varies from run to run). The gate
    rated ACMR, HELO and CBLT as AVOID and BORE as SHORT (72%, setup 78%). The debate ran and the bear case won, so the
    tilt was DEFENSIVE. The desk sized BORE at 21% of NAV against the 15% limit; the committee sent ONLY BORE back, the
    PM model proposed a scale of 0.55 (inside the allowed band) and pass 2 was clean. Offline the same input trades only
    GRNT, so the numbered walkthrough below uses ``OFFLINE_WALKTHROUGH_INPUT`` instead, which the tests verify:
        {"focus_hint": "Look at GRNT, DLPH, ACMR and FTHM today", "nav_usd": 10000000, "max_revisions": 2}
    Open the run and read it step by step:
      * ``mandate`` (cio_plan): ``source`` says who chose the names. Offline the stand-in model returns placeholder
        tickers, so ``source`` is "fallback" and the names come from the hint (tickers, company names, sectors or
        group names all count); with a key the CIO model reads the hint and ``source`` is "model".
      * ``dossiers`` (parallel ticker_worker runs, one per focus name): the indicators, the headlines the model saw, and the
        Decisions probabilities: ``p_setup``, ``direction_probs``, ``conviction_probs``. ``gated`` is true when the
        setup gate overrode a LONG/SHORT pick.
      * ``verdict`` (arbiter): the tilt and its probabilities, ``tail_risk_p``, the rigor grade of each case, and who
        won the debate. ``bull_argument`` / ``bear_argument`` hold the two cases.
      * ``risk_history`` and ``resize_log`` (risk_committee, resize_worker): offline, with
        ``OFFLINE_WALKTHROUGH_INPUT``, two names are actionable (DLPH and GRNT) and the tilt is DEFENSIVE. The desk sizes DLPH at 15.7% of NAV against the 15% limit and GRNT at
        14.3%, so ONLY DLPH is sent back (REQUIRED_SCALE 0.9533, MIN_SCALE 0.9066) and GRNT is never touched. Pass 2
        finds no breach and the loop ends. The stand-in PM proposes 0.5, which MIN_SCALE lifts to 0.9066.
      * ``blotter`` (execution_plan): shares, notional, % of NAV, child-order count and estimated cost per name, then
        totals; ``execution`` has the child orders themselves. ``note`` is the desk note and whether it passed the
        number audit.
    Then change the walkthrough input:
      * ``"max_revisions": 0``: the committee may not send anything back, trims DLPH itself, and ``cap_forced`` is
        true (look at ``risk_history[0].action`` = "trim" and the row's status "trimmed").
      * ``"nav_usd": 2000000000``: the same weights, but each order is about 85-90% of a whole day's average volume,
        so the participation cap forces over 200 child orders spread over 17-19 sessions and the estimated cost rises
        from about 7 bps to about 78 bps.
      * with an API key, ``"focus_hint": "technology"``: if three technology names turn out actionable, a NET_LONG or
        NEUTRAL tilt can put the group above 35%, and then the committee sends back every member of the group, not
        just one name.

EXTEND IT
    Add a short-borrow limit (a fifth check in ``_desk_math.check_limits`` plus its cure scale), replace the 13 equal
    half-hour slots with a volume curve, or add a human approval of the blotter with ``interrupt()`` before
    ``execution_plan`` (that needs a checkpointer, which Studio provides).

LIMITS
    The market is synthetic and the cost model is a textbook square-root impact with an uncalibrated coefficient.
    There are no borrow costs, no intraday data, no corporate actions, and no order is ever sent anywhere. Headlines
    are untrusted text that models read; here they are synthetic, but a real feed would need injection hardening.
    A production desk would also persist every committee pass for audit and version its prompts.
"""

from __future__ import annotations

import math
import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, RetryPolicy, Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import read_csv
from deepdive.common.decisions import choice, decide, predicate, score
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.examples import _desk_math as dm

META = ExampleMeta(
    id="portfolio_desk_lite",
    title="Portfolio desk (lite): parallel research, debate, risk committee, execution",
    category="Multi-agent",
    problem="Turn eight fictional stocks into one risk-checked daily trade blotter: research every name in parallel, "
    "hear a bull and a bear case, let a risk committee send only the breaching names back for resizing, and plan "
    "the execution, with code guaranteeing the limits.",
    features=[
        "Send map-reduce",
        "custom keyed reducer",
        "Command fan-out / skip",
        "parallel branches + join",
        "selective re-dispatch loop (Send from a conditional edge)",
        "loop cap in state",
        "input/output schemas",
        "RetryPolicy",
    ],
    models=["OpenAI chat", "OpenAI Decisions API"],
    datasets=["prices.csv", "news_headlines.csv", "companies.csv"],
    difficulty="advanced",
)

# Names with clear trends, so that LIVE the gate lets several through and the whole pipeline (debate, sizing, selective
# re-dispatch) is exercised. An earlier default of four low-signal names gave a correct but empty "stand aside" day.
EXAMPLE_INPUT = {"focus_hint": "Look at ACMR, BORE, GRNT, HELO and CBLT today", "nav_usd": 10_000_000, "max_revisions": 2}
# The input the offline walkthrough in the docstring (and its test) is written against: with the offline stand-ins two
# names are actionable and exactly one of them breaches a limit, which shows the selective re-dispatch.
OFFLINE_WALKTHROUGH_INPUT = {"focus_hint": "Look at GRNT, DLPH, ACMR and FTHM today", "nav_usd": 10_000_000, "max_revisions": 2}

DEFAULT_NAV = 10_000_000.0
DEFAULT_MAX_REVISIONS = 2
MAX_REVISIONS_LIMIT = 5  # a bigger input is clamped: every round is a model call per breaching name
DISCLAIMER = "FICTIONAL market and synthetic data. Not investment advice."
CONVICTION_LEVELS = ["low", "medium", "high"]
RIGOR_LEVELS = ["weak", "moderate", "strong"]
RIGOR_TEXT = {
    "weak": "ignores contrary data or invents support",
    "moderate": "uses some of the data",
    "strong": "specific numbers throughout and honest about risks",
}
DECISIONS_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.5)


# --------------------------------------------------------------------------- reducer
def merge_by_ticker(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Merge two lists of ``{"ticker": ..., ...}`` records: a ticker in ``right`` replaces the one in ``left``.

    Parallel workers each return a one-record list; the result is sorted by ticker, so it does not depend on which
    worker finished first. A resize worker uses the same reducer to replace one leg of the book."""
    merged = {r["ticker"]: r for r in (left or [])}
    merged.update({r["ticker"]: r for r in (right or [])})
    return [merged[k] for k in sorted(merged)]


# --------------------------------------------------------------------------- schemas (structured output)
class CioMandate(BaseModel):
    """The CIO's choice of names to research today and how much gross exposure to allow."""

    focus_tickers: list[str] = Field(description="3 to 5 tickers, copied exactly from the universe table")
    gross_budget: float = Field(
        description="Gross exposure budget as a fraction of NAV, between 0.25 and 0.70 (0.45 means 45%)"
    )
    rationale: str = Field(description="Two sentences that cite numbers from the table")


class Idea(BaseModel):
    """One trade idea inside a debate case."""

    ticker: str = Field(description="A ticker from the dossiers")
    action: Literal["LONG", "SHORT", "AVOID"]
    argument: str = Field(description="One sentence that cites at least one number from that ticker's dossier")


class DebateCase(BaseModel):
    """One side of the debate."""

    thesis: str = Field(description="Two or three sentences")
    ideas: list[Idea] = Field(description="At most 3 ideas")
    key_risks: list[str] = Field(description="At most 3 ways this case could be wrong")
    anticipated_rebuttal: str = Field(description="The strongest point the other side will make, and a one-sentence reply")


class BullCase(DebateCase):
    """The bull analyst's case (a distinct class so the two branches can be told apart in tests and traces)."""


class BearCase(DebateCase):
    """The bear analyst's case."""


class ResizeProposal(BaseModel):
    """The PM's answer when the risk committee sends one position back."""

    scale: float = Field(
        description="Fraction of the current notional to keep: 1.0 unchanged, 0.5 halve. Between MIN_SCALE and REQUIRED_SCALE."
    )
    rationale: str = Field(description="One sentence that cites a number from the message")


# --------------------------------------------------------------------------- state
class DeskIn(TypedDict, total=False):
    focus_hint: str  # e.g. "semiconductors and software"; optional
    nav_usd: float  # default 10,000,000
    max_revisions: int  # risk-committee resize rounds before it trims by itself; default 2, clamped to 0..5


class TickerJob(TypedDict):
    """Payload of one ``Send("ticker_worker", ...)``."""

    ticker: str


class ResizeJob(TypedDict):
    """Payload of one ``Send("resize_worker", ...)``."""

    leg: dict
    breaches: list[dict]
    required_scale: float
    nav_usd: float
    round: int
    max_revisions: int
    research: str


class DeskState(DeskIn, total=False):
    mandate: dict
    dossiers: Annotated[list[dict], merge_by_ticker]
    review: dict
    bull_argument: dict
    bear_argument: dict
    verdict: dict
    book: dict
    legs: Annotated[list[dict], merge_by_ticker]
    revision_count: int
    risk: dict
    risk_history: Annotated[list[dict], operator.add]
    resize_log: Annotated[list[dict], operator.add]
    cap_forced: bool
    execution: list[dict]
    blotter: dict
    note: dict
    mode: str
    disclaimer: str


class DeskOut(TypedDict):
    mandate: dict
    dossiers: list[dict]
    bull_argument: dict
    bear_argument: dict
    verdict: dict
    blotter: dict
    execution: list[dict]
    note: dict
    risk_history: list[dict]
    resize_log: list[dict]
    revision_count: int
    cap_forced: bool
    mode: str
    disclaimer: str


# --------------------------------------------------------------------------- data helpers
def _companies() -> list[dict[str, Any]]:
    df = read_csv("companies.csv")
    return [
        {
            "ticker": r.ticker,
            "company": r.company,
            "sector": r.sector,
            "beta": float(r.beta),
            "group": dm.sector_group(r.sector),
        }
        for r in df.itertuples()
    ]


def _universe() -> list[dict[str, Any]]:
    """The eight companies with the headline numbers the CIO needs."""
    prices = read_csv("prices.csv")
    out = []
    for c in _companies():
        an = dm.compute_analytics(prices[prices.ticker == c["ticker"]])
        out.append({**c, **{k: an[k] for k in ("ret_5d", "ret_20d", "vol_20d", "last_close")}})
    return out


# --------------------------------------------------------------------------- node: cio_plan
CIO_PROMPT = (
    "You are the chief investment officer of a small fund that trades a FICTIONAL market of eight made-up companies. "
    "Nothing here is real or investment advice. Using the universe table and the PM's hint, choose 3 to 5 tickers to "
    "research today and a gross exposure budget. Rules: use ONLY tickers that appear in the table, spelled exactly; "
    "prefer names with a reason to look (large recent moves, high volatility, a match with the hint); do not put every "
    "pick in one sector group unless the hint asks for that theme (concentration is the risk committee's job, but a "
    "diversified list wastes less of its time). gross_budget is a fraction of NAV between 0.25 and 0.70; choose lower "
    "when the table shows large recent drops or high volatility. Keep the rationale to two sentences and cite numbers "
    "from the table. Do not invent news."
)


def cio_plan(state: DeskState) -> dict:
    """Normalise the inputs, let the CIO model pick names and a budget, and validate what it says."""
    nav = state.get("nav_usd")
    nav = DEFAULT_NAV if nav is None else float(nav)
    if not nav > 0:
        raise ValueError("nav_usd must be a positive number of dollars")
    max_rev = state.get("max_revisions")
    max_rev = DEFAULT_MAX_REVISIONS if max_rev is None else int(max_rev)
    max_rev = min(max(max_rev, 0), MAX_REVISIONS_LIMIT)
    hint = str(state.get("focus_hint") or "").strip()

    universe = _universe()
    table = "\n".join(
        f"{u['ticker']} | {u['company']} | sector {u['sector']} | group {u['group']} | beta {u['beta']} | "
        f"5d {dm.pct(u['ret_5d'])} | 20d {dm.pct(u['ret_20d'])} | vol {dm.pct(u['vol_20d'], False)} per day"
        for u in universe
    )
    request = f"PM HINT: {hint or '(none)'}\nNAV USD: {nav:,.0f}\n\nUNIVERSE\n{table}"
    try:
        m = structured_llm(CioMandate).invoke([SystemMessage(CIO_PROMPT), HumanMessage(request)])
        raw, budget, rationale, error = list(m.focus_tickers), m.gross_budget, m.rationale, None
    except Exception as exc:  # noqa: BLE001 - the CIO is advisory: a failed call must not stop the desk
        # (the error is recorded in the mandate, so it is visible in Studio)
        raw, budget, rationale, error = [], None, "", f"{type(exc).__name__}: {exc}"

    valid, dropped = dm.validate_focus(raw, {u["ticker"] for u in universe})
    if len(valid) >= 3:
        focus, source = valid, "model"
    else:
        focus, source = dm.fallback_focus(hint, universe), "fallback"
        rationale = f"Deterministic fallback (hint matches first, then the largest 20-day moves): {', '.join(focus)}."
    mandate = {
        "focus_tickers": focus,
        "gross_budget": dm.clamp_budget(budget),
        "rationale": rationale,
        "source": source,
        "dropped": dropped,
        "model_error": error,
    }
    return {
        "nav_usd": nav,
        "max_revisions": max_rev,
        "revision_count": 0,
        "cap_forced": False,
        "mandate": mandate,
        # Studio reuses the thread: drop what an earlier run left in the accumulating channels (a keyed reducer never
        # forgets a ticker) and the debate that a no-trade run skips and would not overwrite.
        "dossiers": Overwrite([]),
        "legs": Overwrite([]),
        "risk_history": Overwrite([]),
        "resize_log": Overwrite([]),
        "bull_argument": {},
        "bear_argument": {},
    }


def fan_out_research(state: DeskState) -> list[Send]:
    """Map step: one ``ticker_worker`` per chosen name."""
    return [Send("ticker_worker", {"ticker": t}) for t in state["mandate"]["focus_tickers"]]


# --------------------------------------------------------------------------- node: ticker_worker
def _ticker_questions(ticker: str) -> list[dict[str, Any]]:
    return [
        predicate(
            "setup_supportive",
            f"Is there enough directional evidence in the trend, momentum, volatility and recent headlines for {ticker} "
            "to justify opening a position today, long OR short? A clear downtrend counts as evidence for a short "
            "just as a clear uptrend does for a long. Answer low only when the evidence is missing, stale or "
            "flatly contradictory.",
        ),
        choice(
            "direction",
            f"Which position does the evidence above best support for {ticker}?",
            {
                "LONG": "trend, momentum and news broadly positive and the price is not stretched",
                "SHORT": "trend, momentum and news broadly negative, or a stretched price that is rolling over",
                "AVOID": "signals conflict, or an event makes either side unattractive",
            },
        ),
        score(
            "conviction",
            f"How strong is the evidence for the position you would take in {ticker}?",
            {
                "low": "signals conflict or rest on one data point",
                "medium": "most signals agree",
                "high": "trend, momentum, volatility and headlines all agree",
            },
        ),
    ]


def ticker_worker(job: TickerJob) -> dict:
    """Deterministic analytics plus ONE Decisions API call for one name; returns a one-record dossier list."""
    ticker = str(job["ticker"]).upper()
    meta = next((c for c in _companies() if c["ticker"] == ticker), None)
    if meta is None:
        raise ValueError(f"unknown ticker {ticker!r}")
    prices = read_csv("prices.csv")
    news = read_csv("news_headlines.csv")
    an = dm.compute_analytics(prices[prices.ticker == ticker])
    heads = dm.recent_headlines(news[news.ticker == ticker], an["asof"])
    evidence = dm.fact_sheet(meta, an, heads)

    res = decide(evidence, _ticker_questions(ticker))
    refused = [q for q in ("setup_supportive", "direction", "conviction") if res.refused(q)]
    p_setup = res.p("setup_supportive")  # a refusal reads as 0.0, which closes the gate
    pick = res.pick("direction")
    direction, gated = dm.resolve_direction(pick, p_setup)
    dir_ans, conv_ans = res.answers.get("direction"), res.answers.get("conviction")
    conv_probs = dict(conv_ans.probabilities) if conv_ans is not None else {}
    expected = dm.expected_level(conv_probs, CONVICTION_LEVELS, fallback=res.score("conviction"))
    dossier = {
        **meta,
        "analytics": an,
        "headlines": heads,
        "p_setup": round(p_setup, 4),
        "direction_pick": pick,
        "direction_probs": {k: round(v, 4) for k, v in (dir_ans.probabilities if dir_ans is not None else {}).items()},
        "direction": direction,
        "gated": gated,
        "conviction_level": res.level("conviction", "low"),
        "conviction_probs": {k: round(v, 4) for k, v in conv_probs.items()},
        "conviction_score": round(expected, 4),
        "refused": refused,
        "decision_model": res.model,
        "decision_mode": res.mode,
    }
    return {"dossiers": [dossier]}


# --------------------------------------------------------------------------- node: dossier_review (barrier + routing)
def _dossier_text(d: dict[str, Any]) -> str:
    probs = d.get("direction_probs") or {}
    read = " / ".join(f"{k} {v:.0%}" for k, v in probs.items()) or "n/a"
    gate = " (the setup gate overrode the pick)" if d.get("gated") else ""
    return (
        f"{dm.fact_sheet(d, d['analytics'], d['headlines'])}\n"
        f"DESK MODEL READ: direction {d['direction']}{gate}; P(setup supportive) {d['p_setup']:.0%}; "
        f"direction probabilities {read}; conviction {d['conviction_level']}"
    )


def dossier_review(state: DeskState) -> Command[Literal["bull_case", "bear_case", "desk"]]:
    """Fan-in barrier. Debate only if some name is actionable; otherwise skip straight to an empty book."""
    dossiers = state["dossiers"]
    actionable = [d["ticker"] for d in dossiers if d["direction"] in ("LONG", "SHORT")]
    review = {
        "actionable": actionable,
        "standing_aside": [d["ticker"] for d in dossiers if d["direction"] == "AVOID"],
        "gated": [d["ticker"] for d in dossiers if d["gated"]],
        "debate": bool(actionable),
        "summary": [
            f"{d['ticker']}: {d['direction']} (P(setup) {d['p_setup']:.0%}, conviction {d['conviction_level']})"
            for d in dossiers
        ],
    }
    if actionable:
        return Command(update={"review": review}, goto=["bull_case", "bear_case"])
    verdict = {
        "skipped": True,
        "tilt": "NEUTRAL",
        "gross_scale": dm.TILTS["NEUTRAL"]["gross_scale"],
        "reason": "No name passed the research gate, so there was nothing to debate.",
    }
    return Command(update={"review": review, "verdict": verdict}, goto="desk")


# --------------------------------------------------------------------------- nodes: bull_case / bear_case
BULL_PROMPT = (
    "You are the BULL analyst on a small desk that trades a FICTIONAL market of made-up companies. Nothing here is real "
    "or investment advice. Make the strongest HONEST case for taking risk in the names below, using only the facts in "
    "the dossiers (numbers and headlines). Rules: at most 3 ideas, each a LONG on a ticker from the dossiers, and each "
    "argument must cite at least one number from that ticker's dossier; at most 3 key_risks (ways your case could be "
    "wrong); in anticipated_rebuttal give the strongest point the bear analyst will make against you (they have not "
    "spoken yet) and answer it in one sentence. If the dossiers do not support a bull case, say so plainly in the "
    "thesis and propose fewer ideas. Never invent prices, news or events."
)
BEAR_PROMPT = (
    "You are the BEAR analyst on a small desk that trades a FICTIONAL market of made-up companies. Nothing here is real "
    "or investment advice. Make the strongest HONEST case for caution in the names below, using only the facts in the "
    "dossiers (numbers and headlines). Rules: at most 3 ideas, each a SHORT or AVOID on a ticker from the dossiers, and "
    "each argument must cite at least one number from that ticker's dossier; at most 3 key_risks (ways your case could "
    "be wrong); in anticipated_rebuttal give the strongest point the bull analyst will make against you (they have not "
    "spoken yet) and answer it in one sentence. If the dossiers do not support a bear case, say so plainly in the "
    "thesis and propose fewer ideas. Never invent prices, news or events."
)


def make_advocate(side: Literal["bull", "bear"]):
    """Build the bull or bear node from one function (they differ only in prompt and schema)."""
    schema = BullCase if side == "bull" else BearCase
    prompt = BULL_PROMPT if side == "bull" else BEAR_PROMPT

    def advocate(state: DeskState) -> dict:
        dossiers = state["dossiers"]
        focus = [d["ticker"] for d in dossiers]
        pack = "DOSSIERS\n\n" + "\n\n---\n\n".join(_dossier_text(d) for d in dossiers)
        case = structured_llm(schema).invoke([SystemMessage(prompt), HumanMessage(pack)])
        ideas = []
        for idea in case.ideas:
            t = idea.ticker.strip().upper()
            if t in focus:  # drop ideas about tickers that are not on the desk today
                ideas.append({**idea.model_dump(), "ticker": t})
        return {f"{side}_argument": {**case.model_dump(exclude={"ideas"}), "ideas": ideas[:3], "side": side}}

    advocate.__name__ = f"{side}_case"
    return advocate


# --------------------------------------------------------------------------- node: arbiter
def _case_text(label: str, case: dict[str, Any]) -> str:
    ideas = "\n".join(f"- {i['action']} {i['ticker']}: {i['argument']}" for i in case.get("ideas", [])) or "- (no ideas)"
    risks = "; ".join(case.get("key_risks", [])) or "(none given)"
    return (
        f"{label}\nThesis: {case.get('thesis', '')}\nIdeas:\n{ideas}\nKey risks: {risks}\n"
        f"Anticipated rebuttal: {case.get('anticipated_rebuttal', '')}"
    )


def arbiter(state: DeskState) -> dict:
    """ONE Decisions API call judges the debate; code turns the answers into a tilt."""
    evidence = (
        "DOSSIERS\n\n"
        + "\n\n---\n\n".join(_dossier_text(d) for d in state["dossiers"])
        + "\n\n=====\n\n"
        + _case_text("BULL CASE", state.get("bull_argument") or {})
        + "\n\n"
        + _case_text("BEAR CASE", state.get("bear_argument") or {})
    )
    questions = [
        choice(
            "tilt",
            "Weighing the bull and bear cases against the dossier data, which overall tilt should today's book take?",
            {
                "NET_LONG": "risk-on: the data supports mostly long positions with small shorts as hedges",
                "NEUTRAL": "balanced: long and short ideas are about equally supported, so run a smaller book",
                "DEFENSIVE": "risk-off: the data supports a small book that leans short",
            },
        ),
        predicate(
            "tail_risk_elevated",
            "Do the dossiers show elevated risk of a sharp drawdown across these names in the next few days "
            "(deep drawdowns, extreme z-scores or RSI, rising volatility, or negative headline clusters)?",
        ),
        score(
            "bull_rigor",
            "How rigorous is the BULL case: does it rest on the numbers and headlines in the dossiers?",
            RIGOR_TEXT,
        ),
        score(
            "bear_rigor",
            "How rigorous is the BEAR case: does it rest on the numbers and headlines in the dossiers?",
            RIGOR_TEXT,
        ),
    ]
    res = decide(evidence, questions)
    pick = res.pick("tilt")
    tilt_pick = pick if pick in dm.TILTS else "DEFENSIVE"  # a refused or unknown tilt fails safe
    p_tail = res.p("tail_risk_elevated", 0.5)
    tilt, overridden = dm.apply_tail_override(tilt_pick, p_tail)

    def rigor(name: str) -> dict[str, Any]:
        ans = res.answers.get(name)
        probs = dict(ans.probabilities) if ans is not None else {}
        return {
            "level": res.level(name, "weak"),
            "expected": round(dm.expected_level(probs, RIGOR_LEVELS, fallback=res.score(name)), 4),
            "probs": {k: round(v, 4) for k, v in probs.items()},
        }

    bull, bear = rigor("bull_rigor"), rigor("bear_rigor")
    tilt_ans = res.answers.get("tilt")
    verdict = {
        "skipped": False,
        "tilt": tilt,
        "tilt_pick": tilt_pick,
        "tilt_probs": {k: round(v, 4) for k, v in (tilt_ans.probabilities if tilt_ans is not None else {}).items()},
        "tail_risk_p": round(p_tail, 4),
        "tail_override": overridden,
        "bull_rigor": bull,
        "bear_rigor": bear,
        "winner": dm.debate_winner(bull["expected"], bear["expected"]),
        "gross_scale": dm.TILTS[tilt]["gross_scale"],
        "refused": [q for q in ("tilt", "tail_risk_elevated", "bull_rigor", "bear_rigor") if res.refused(q)],
        "decision_model": res.model,
        "decision_mode": res.mode,
    }
    return {"verdict": verdict}


# --------------------------------------------------------------------------- node: desk
def desk(state: DeskState) -> dict:
    """Deterministic sizing from tilt, conviction and inverse volatility. Not limit-aware on purpose."""
    nav = float(state["nav_usd"])
    tilt = state["verdict"]["tilt"]
    budget = state["mandate"]["gross_budget"]
    gross_usd = dm.floor_lot(nav * budget * dm.TILTS[tilt]["gross_scale"])
    candidates = [d for d in state["dossiers"] if d["direction"] in ("LONG", "SHORT")]
    legs = dm.size_book(candidates, tilt, gross_usd, nav) if candidates else []
    book = {
        "tilt": tilt,
        "gross_budget": budget,
        "gross_scale": dm.TILTS[tilt]["gross_scale"],
        "effective_gross_usd": gross_usd,
        "proposed_gross_usd": sum(leg["notional_usd"] for leg in legs),
        "n_legs": len(legs),
    }
    return {"legs": legs, "book": book}


# --------------------------------------------------------------------------- node: risk_committee (+ its router)
def risk_committee(state: DeskState) -> dict:
    """Run every limit check. Dispatch resizes while the cap allows; at the cap, trim deterministically."""
    nav = float(state["nav_usd"])
    legs = state.get("legs") or []
    avoid = {d["ticker"] for d in state["dossiers"] if d["direction"] == "AVOID"}
    budget = float(state["book"]["effective_gross_usd"])
    rev = int(state.get("revision_count") or 0)
    cap = int(state["max_revisions"])

    breaches = dm.check_limits(legs, nav=nav, gross_budget_usd=budget, avoid=avoid)
    update: dict[str, Any] = {}
    risk: dict[str, Any] = {"action": "pass", "round": rev, "breaches": breaches, "to_resize": {}, "trims": []}
    if breaches and rev < cap:
        risk.update(action="resize", round=rev + 1, to_resize=dm.cure_scales(breaches))
        update["revision_count"] = rev + 1
    elif breaches:
        new_legs, trims = dm.enforce_limits(legs, nav=nav, gross_budget_usd=budget, avoid=avoid)
        if dm.check_limits(new_legs, nav=nav, gross_budget_usd=budget, avoid=avoid):  # unreachable by construction
            raise RuntimeError("risk engine invariant violated: a deterministic trim left a breach")
        changed = {t["ticker"] for t in trims}
        update["legs"] = [leg for leg in new_legs if leg["ticker"] in changed]
        update["cap_forced"] = True
        risk.update(action="trim", trims=trims)
        legs = new_legs

    live = [leg for leg in legs if leg["notional_usd"] > 0]
    gross = sum(leg["notional_usd"] for leg in live)
    history = {
        "pass": len(state.get("risk_history") or []) + 1,
        "action": risk["action"],
        "revisions_dispatched": update.get("revision_count", rev),
        "breaches": [b["detail"] for b in breaches],
        "breaching_names": dm.breaching_names(breaches),
        "gross_pct_nav": round(gross / nav, 6),
        "group_pct_nav": dm.group_exposure(legs, nav),
        "book_usd": {leg["ticker"]: leg["notional_usd"] for leg in legs},
    }
    return {**update, "risk": risk, "risk_history": [history]}


def _research_line(d: dict[str, Any]) -> str:
    return (
        f"direction {d['direction']} (P(setup supportive) {d['p_setup']:.0%}), conviction {d['conviction_level']}, "
        f"5d {dm.pct(d['analytics']['ret_5d'])}, 20d {dm.pct(d['analytics']['ret_20d'])}, "
        f"drawdown from 60d high {dm.pct(d['analytics']['drawdown_60d'])}, RSI {d['analytics']['rsi_14']:.0f}"
    )


def route_after_risk(state: DeskState) -> list[Send] | str:
    """Selective re-dispatch: a ``Send`` to ``resize_worker`` for each breaching name and no one else."""
    risk = state.get("risk") or {}
    if risk.get("action") != "resize":
        return "execution_plan"
    legs = {leg["ticker"]: leg for leg in state["legs"]}
    dossiers = {d["ticker"]: d for d in state["dossiers"]}
    return [
        Send(
            "resize_worker",
            {
                "leg": legs[ticker],
                "breaches": [b for b in risk["breaches"] if ticker in b["tickers"]],
                "required_scale": scale,
                "nav_usd": state["nav_usd"],
                "round": risk["round"],
                "max_revisions": state["max_revisions"],
                "research": _research_line(dossiers[ticker]),
            },
        )
        for ticker, scale in sorted(risk["to_resize"].items())
    ]


# --------------------------------------------------------------------------- node: resize_worker
RESIZE_PROMPT = (
    "You are the portfolio manager of a small desk trading a FICTIONAL market. The independent risk committee has sent "
    "ONE of your positions back because it breaches a limit. Decide how much of it to keep. scale is the fraction of the "
    "current notional you keep: 1.0 leaves it unchanged, 0.5 halves it. REQUIRED_SCALE is the largest scale that "
    "satisfies every limit; MIN_SCALE is the smallest the desk accepts (it never cuts more than twice what the limit "
    "requires). Choose a scale from MIN_SCALE to REQUIRED_SCALE inclusive. Choose REQUIRED_SCALE unless the research "
    "line gives a concrete reason to cut deeper (weak conviction, a deep drawdown, an adverse reading). Never choose "
    "a scale above REQUIRED_SCALE. Answer with the scale and ONE sentence that cites a number from the message. Do not "
    "discuss other positions."
)


def resize_worker(job: ResizeJob) -> dict:
    """Scale one breaching position down. The model proposes; ``dm.applied_scale`` bounds what is applied."""
    leg, required, nav = job["leg"], float(job["required_scale"]), float(job["nav_usd"])
    if required <= 0:
        proposed, source = 0.0, "rule"
        rationale = "Position is not permitted by the risk rules; dropped without asking the model."
    else:
        message = (
            f"TICKER: {leg['ticker']}\n"
            f"SIDE: {leg['side']}   CURRENT NOTIONAL USD: {leg['notional_usd']:,.0f} "
            f"({leg['notional_usd'] / nav:.1%} of NAV)\n"
            f"REVISION ROUND: {job['round']} of {job['max_revisions']}\n"
            "BREACHES:\n" + "\n".join(f"- {b['detail']}" for b in job["breaches"]) + "\n"
            f"REQUIRED_SCALE: {required:.4f}\n"
            f"MIN_SCALE: {dm.min_scale(required):.4f}\n"
            f"RESEARCH: {job['research']}"
        )
        p = structured_llm(ResizeProposal, "fast").invoke([SystemMessage(RESIZE_PROMPT), HumanMessage(message)])
        proposed, rationale, source = p.scale, p.rationale, "model"
    applied = dm.applied_scale(proposed, required)
    new_leg = dm.apply_scale(leg, applied, nav=nav, step=f"resize round {job['round']}", status="resized")
    shown = float(proposed) if isinstance(proposed, (int, float)) and math.isfinite(proposed) else None
    entry = {
        "round": job["round"],
        "ticker": leg["ticker"],
        "breaches": [b["code"] for b in job["breaches"]],
        "required_scale": round(required, 4),
        "min_scale": dm.min_scale(required),
        "proposed_scale": None if shown is None else round(shown, 4),
        "applied_scale": applied,
        "before_usd": leg["notional_usd"],
        "after_usd": new_leg["notional_usd"],
        "source": source,
        "rationale": rationale,
    }
    return {"legs": [new_leg], "resize_log": [entry]}


# --------------------------------------------------------------------------- node: execution_plan
def execution_plan(state: DeskState) -> dict:
    """Shares, child orders, participation and cost per name, then the final blotter. Fully deterministic."""
    nav = float(state["nav_usd"])
    dossiers = {d["ticker"]: d for d in state["dossiers"]}
    rows: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    for leg in state.get("legs") or []:
        price = float(leg["last_price"])
        shares = int(math.floor(leg["notional_usd"] / price)) if leg["notional_usd"] > 0 else 0
        if shares <= 0:
            continue
        d = dossiers[leg["ticker"]]
        plan = dm.plan_execution(shares, price, d["analytics"]["adv_20d_shares"], d["analytics"]["vol_20d"])
        notional = round(shares * price, 2)
        rows.append(
            {
                "ticker": leg["ticker"],
                "company": d["company"],
                "side": leg["side"],
                "order": "BUY" if leg["side"] == "LONG" else "SELL SHORT",
                "shares": shares,
                "price": price,
                "notional_usd": notional,
                "pct_nav": round(notional / nav, 6),
                "group": leg["group"],
                "p_direction": d["direction_probs"].get(d["direction_pick"] or "", None),
                "conviction": d["conviction_level"],
                "status": leg["status"],
                "slices": plan["n_slices"],
                "est_cost_bps": plan["est_cost_bps"],
                "est_cost_usd": plan["est_cost_usd"],
            }
        )
        execution.append({"ticker": leg["ticker"], **plan})

    # Last line of defence: re-check the limits on the final share counts. A blotter that fails is never emitted.
    avoid = {t for t, d in dossiers.items() if d["direction"] == "AVOID"}
    final_legs = [
        {"ticker": r["ticker"], "side": r["side"], "notional_usd": r["notional_usd"], "group": r["group"]} for r in rows
    ]
    budget = float(state["book"]["effective_gross_usd"])
    problems = dm.check_limits(final_legs, nav=nav, gross_budget_usd=budget, avoid=avoid)
    if problems:
        raise ValueError("refusing to build a blotter that breaches limits: " + "; ".join(p["detail"] for p in problems))

    long_usd = sum((r["notional_usd"] for r in rows if r["side"] == "LONG"), 0.0)
    short_usd = sum((r["notional_usd"] for r in rows if r["side"] == "SHORT"), 0.0)
    gross = long_usd + short_usd
    cost = sum((r["est_cost_usd"] for r in rows), 0.0)
    asof = next(iter(dossiers.values()))["analytics"]["asof"] if dossiers else None
    blotter = {
        "asof": asof,
        "nav_usd": nav,
        "tilt": state["book"]["tilt"],
        "tilt_gross_scale": state["book"]["gross_scale"],
        "gross_budget_usd": state["book"]["effective_gross_usd"],
        "rows": rows,
        "standing_aside": sorted(avoid),
        "totals": {
            "n_positions": len(rows),
            "long_usd": round(long_usd, 2),
            "short_usd": round(short_usd, 2),
            "gross_usd": round(gross, 2),
            "net_usd": round(long_usd - short_usd, 2),
            "gross_pct_nav": round(gross / nav, 6),
            "net_pct_nav": round((long_usd - short_usd) / nav, 6),
            "est_cost_usd": round(cost, 2),
            "est_cost_bps": round(cost / gross * 10_000.0, 3) if gross else 0.0,
            "group_pct_nav": dm.group_exposure(final_legs, nav),
        },
        "revisions_used": int(state.get("revision_count") or 0),
        "max_revisions": int(state["max_revisions"]),
        "cap_forced": bool(state.get("cap_forced")),
    }
    return {"execution": execution, "blotter": blotter}


# --------------------------------------------------------------------------- node: desk_note
NOTE_PROMPT = (
    "You write the end-of-day desk note for the PM of a small desk trading a FICTIONAL market. Nothing here is real or "
    "investment advice. Write 3 to 5 plain sentences, no bullets, no markdown, no dates. Use ONLY the FACTS block. Quote "
    "a number only by copying it exactly as written there (same digits, same commas); never round, abbreviate "
    "($1.2M), convert, add up or compute a new figure. Mention the tilt, the largest position, what the risk committee "
    "changed (revision rounds, and whether the cap forced a trim) and the estimated execution cost. If FACTS list no "
    "positions, say the desk is standing aside. Do not give investment advice or suggest actions beyond the blotter."
)


def _usd(x: float) -> str:
    return f"{x:,.0f}"


def _pc(x: float) -> str:
    return f"{x * 100:.1f}"


def note_facts(blotter: dict[str, Any]) -> str:
    """Every figure the desk note may quote, written once, in the exact form it should be copied."""
    t, rows = blotter["totals"], blotter["rows"]
    lines = [
        f"NAV: {_usd(blotter['nav_usd'])} USD",
        f"TILT: {blotter['tilt']} (gross scale {blotter['tilt_gross_scale']})",
        f"POSITIONS: {t['n_positions']}",
        f"GROSS: {_usd(t['gross_usd'])} USD ({_pc(t['gross_pct_nav'])}% of NAV)",
        f"NET: {_usd(t['net_usd'])} USD ({_pc(t['net_pct_nav'])}% of NAV)",
        f"RISK REVISION ROUNDS USED: {blotter['revisions_used']} of {blotter['max_revisions']}",
        f"REVISION CAP FORCED A TRIM: {'yes' if blotter['cap_forced'] else 'no'}",
    ]
    if rows:
        big = max(rows, key=lambda r: r["notional_usd"])
        lines.append(
            f"LARGEST POSITION: {big['side']} {big['ticker']} {_usd(big['notional_usd'])} USD "
            f"({_pc(big['pct_nav'])}% of NAV)"
        )
        lines.append(f"ESTIMATED EXECUTION COST: {_usd(t['est_cost_usd'])} USD ({t['est_cost_bps']:.1f} bps of gross)")
        lines.append("POSITION DETAIL:")
        lines += [
            f"- {r['side']} {r['ticker']} {r['shares']:,} shares at {r['price']:.2f} = {_usd(r['notional_usd'])} USD "
            f"({_pc(r['pct_nav'])}% of NAV), {r['slices']} child orders, estimated cost {r['est_cost_bps']:.1f} bps"
            for r in rows
        ]
    if blotter["standing_aside"]:
        lines.append("STANDING ASIDE: " + ", ".join(blotter["standing_aside"]))
    return "\n".join(lines)


def template_note(blotter: dict[str, Any]) -> str:
    """The note used when the model's note quotes a number that is not in the facts (or when there is nothing to trade)."""
    t, rows = blotter["totals"], blotter["rows"]
    aside = f" Standing aside: {', '.join(blotter['standing_aside'])}." if blotter["standing_aside"] else ""
    if not rows:
        return f"The desk is standing aside today: no name passed the research gate and no orders are planned.{aside}"
    big = max(rows, key=lambda r: r["notional_usd"])
    forced = " The revision cap forced a deterministic trim." if blotter["cap_forced"] else ""
    return (
        f"Tilt {blotter['tilt']}. {t['n_positions']} positions with gross {_usd(t['gross_usd'])} USD "
        f"({_pc(t['gross_pct_nav'])}% of NAV) and net {_usd(t['net_usd'])} USD ({_pc(t['net_pct_nav'])}% of NAV). "
        f"Largest position: {big['side']} {big['ticker']} {_usd(big['notional_usd'])} USD ({_pc(big['pct_nav'])}% of NAV). "
        f"The risk committee used {blotter['revisions_used']} of {blotter['max_revisions']} revision rounds.{forced} "
        f"Estimated execution cost {_usd(t['est_cost_usd'])} USD ({t['est_cost_bps']:.1f} bps of gross).{aside}"
    )


def desk_note(state: DeskState) -> dict:
    """A model writes the note; code verifies that every number in it is in the blotter facts."""
    blotter = state["blotter"]
    base = {"mode": mode_label(), "disclaimer": DISCLAIMER}
    if not blotter["rows"]:  # nothing to say that a template cannot say: no model call
        return {**base, "note": {"text": template_note(blotter), "source": "template", "unsupported_numbers": []}}
    facts = note_facts(blotter)
    msg = get_llm("fast").invoke(
        [
            SystemMessage(NOTE_PROMPT),
            HumanMessage(
                "Write the desk note for today's blotter. Follow the rules in the system message exactly. Copy numbers "
                "from FACTS character for character; do not round, convert or compute anything. FACTS follow below.\n\n"
                f"FACTS\n{facts}"
            ),
        ]
    )
    text = str(msg.text).strip()
    bad = dm.unsupported_numbers(text, facts)
    if bad or not text:
        replaced = {"text": template_note(blotter), "source": "template", "unsupported_numbers": bad}
        return {**base, "note": {**replaced, "rejected_model_note": text}}
    return {**base, "note": {"text": text, "source": "model", "unsupported_numbers": []}}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(DeskState, input_schema=DeskIn, output_schema=DeskOut)
    g.add_node("cio_plan", cio_plan)
    g.add_node("ticker_worker", ticker_worker, retry_policy=DECISIONS_RETRY)
    g.add_node("dossier_review", dossier_review, destinations=("bull_case", "bear_case", "desk"))
    g.add_node("bull_case", make_advocate("bull"))
    g.add_node("bear_case", make_advocate("bear"))
    g.add_node("arbiter", arbiter, retry_policy=DECISIONS_RETRY)
    g.add_node("desk", desk)
    g.add_node("risk_committee", risk_committee)
    g.add_node("resize_worker", resize_worker)
    g.add_node("execution_plan", execution_plan)
    g.add_node("desk_note", desk_note)

    g.add_edge(START, "cio_plan")
    g.add_conditional_edges("cio_plan", fan_out_research, ["ticker_worker"])  # Send x N
    g.add_edge("ticker_worker", "dossier_review")  # runs once, after every worker of the superstep finished
    g.add_edge(["bull_case", "bear_case"], "arbiter")  # explicit join: wait for BOTH cases
    g.add_edge("arbiter", "desk")
    g.add_edge("desk", "risk_committee")
    g.add_conditional_edges("risk_committee", route_after_risk, ["resize_worker", "execution_plan"])  # Send x k, or move on
    g.add_edge("resize_worker", "risk_committee")  # the cycle
    g.add_edge("execution_plan", "desk_note")
    g.add_edge("desk_note", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

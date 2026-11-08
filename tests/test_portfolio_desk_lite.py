"""Behaviour tests for portfolio_desk_lite (FICTIONAL market, synthetic data).

All runs are offline. Model behaviour is pinned with the stand-ins from deepdive.common.fakes / decisions. One caveat
shapes the helpers below: ``set_decision_override`` is keyed by question name only, so it cannot give two tickers
different directions. The ``decisions`` fixture wraps ``ex.decide`` and sets the override per ticker under a lock.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Overwrite, RetryPolicy, Send

from deepdive.common.data import read_csv
from deepdive.common.decisions import Answer, set_decision_override
from deepdive.common.fakes import set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example, lint_openai_strict, normalise, run_twice
from deepdive.examples import _desk_math as dm
from deepdive.examples import portfolio_desk_lite as ex

NAV = 10_000_000.0
ALL = ["ACMR", "BORE", "CBLT", "DLPH", "EMBR", "FTHM", "GRNT", "HELO"]


# --------------------------------------------------------------------------- helpers
class Decisions:
    """Per-ticker pins for the Decisions API stand-in, plus a record of every call."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[tuple[str, ...], str | None, str]] = []  # (question names, ticker, evidence text)
        self.direction: dict[str, str] = {}
        self.default_direction = "LONG"
        self.conviction: str | dict[str, str] = "medium"
        self.setup: float | dict[str, float] = 0.9
        self.delay: dict[str, float] = {}
        self.fail_once: set[str] = set()
        self.refuse: set[str] = set()  # question names whose answer is replaced by a refusal
        self._lock = threading.Lock()
        self._real = ex.decide
        monkeypatch.setattr(ex, "decide", self._decide)

    @staticmethod
    def _get(table, ticker, default):
        return table.get(ticker, default) if isinstance(table, dict) else table

    def _decide(self, text, questions, **kw):
        names = tuple(q["name"] for q in questions)
        ticker = re.search(r"TICKER: (\w+)", text).group(1) if "direction" in names else None
        if ticker and self.delay.get(ticker):
            time.sleep(self.delay[ticker])
        with self._lock:
            self.calls.append((names, ticker, text))
            if ticker in self.fail_once:
                self.fail_once.discard(ticker)
                raise ConnectionError("simulated transient network error")
            if ticker:
                set_decision_override("direction", self.direction.get(ticker, self.default_direction))
                set_decision_override("conviction", self._get(self.conviction, ticker, "medium"))
                set_decision_override("setup_supportive", self._get(self.setup, ticker, 0.9))
            res = self._real(text, questions, **kw)
        for q in self.refuse & set(names):
            res.answers[q] = Answer(name=q, type="refusal")
        return res

    def tilt_calls(self):
        return [c for c in self.calls if "tilt" in c[0]]

    def ticker_calls(self):
        return [c for c in self.calls if c[1]]


@pytest.fixture
def decisions(monkeypatch):
    d = Decisions(monkeypatch)
    set_decision_override("tilt", "NEUTRAL")
    set_decision_override("tail_risk_elevated", 0.1)
    return d


def choose(*tickers, budget=0.5):
    """Make the CIO model pick exactly these tickers."""
    set_structured_handler(ex.CioMandate, lambda text: ex.CioMandate(focus_tickers=list(tickers), gross_budget=budget, rationale="r"))


class ResizeModel:
    """A scripted PM for the resize workers. Records (ticker, required_scale) per call; thread-safe by construction."""

    def __init__(self, scale_fn):
        self.scale_fn = scale_fn
        self.calls: list[tuple[str, float, str]] = []

    def __call__(self, text):
        ticker = re.search(r"TICKER: (\w+)", text).group(1)
        required = float(re.search(r"REQUIRED_SCALE: ([0-9.]+)", text).group(1))
        self.calls.append((ticker, required, text))
        return ex.ResizeProposal(scale=self.scale_fn(required), rationale="scripted")


def install_resizer(scale_fn):
    model = ResizeModel(scale_fn)
    set_structured_handler(ex.ResizeProposal, model)
    return model


def diligent():
    return install_resizer(lambda required: required)  # echoes the cure scale it was given


def lazy(scale=0.99):
    return install_resizer(lambda required: scale)  # always under-cuts


@dataclass
class Run:
    state: dict
    events: list[tuple[str, dict]] = field(default_factory=list)

    @property
    def out(self) -> dict:
        return {k: v for k, v in self.state.items() if k in ex.DeskOut.__annotations__}

    def nodes(self, name):
        return [u for n, u in self.events if n == name]


def run(payload=None, *, recursion_limit=100) -> Run:
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": recursion_limit}
    events = []
    for chunk in g.stream(payload if payload is not None else {}, cfg, stream_mode="updates"):
        events.extend(chunk.items())
    return Run(g.get_state(cfg).values, events)


def make_leg(ticker, side, notional, nav=NAV):
    prices = read_csv("prices.csv")
    an = dm.compute_analytics(prices[prices.ticker == ticker])
    group = next(c["group"] for c in ex._companies() if c["ticker"] == ticker)
    return {
        "ticker": ticker, "side": side, "notional_usd": float(notional), "pct_nav": notional / nav, "group": group,
        "last_price": an["last_close"], "status": "proposed", "basis": {},
        "history": [{"step": "desk", "scale": 1.0, "notional_usd": float(notional)}],
    }  # fmt: skip


def fixed_book(monkeypatch, sizes):
    """Replace the desk's sizing with fixed legs {ticker: (side, notional)} so risk scenarios are exact."""
    monkeypatch.setattr(dm, "size_book", lambda candidates, tilt, gross_usd, nav: [make_leg(t, s, n, nav) for t, (s, n) in sorted(sizes.items())])


def assert_blotter_clean(out):
    """Recompute every limit from the blotter rows alone (not through the engine under test)."""
    b = out["blotter"]
    nav = b["nav_usd"]
    avoid = {d["ticker"] for d in out["dossiers"] if d["direction"] == "AVOID"}
    groups: dict[str, float] = {}
    gross = long = short = 0.0
    for r in b["rows"]:
        assert r["notional_usd"] <= 0.15 * nav + 0.01, f"{r['ticker']} above 15% of NAV"
        assert r["shares"] * r["price"] == pytest.approx(r["notional_usd"], abs=0.01)
        assert r["ticker"] not in avoid or r["side"] == "LONG", f"SHORT in AVOID name {r['ticker']}"
        assert r["shares"] > 0 and r["notional_usd"] > 0
        groups[r["group"]] = groups.get(r["group"], 0.0) + r["notional_usd"]
        gross += r["notional_usd"]
        long += r["notional_usd"] if r["side"] == "LONG" else 0.0
        short += r["notional_usd"] if r["side"] == "SHORT" else 0.0
    assert all(v <= 0.35 * nav + 0.01 for v in groups.values()), groups
    assert gross <= b["gross_budget_usd"] + 0.01
    t = b["totals"]
    assert t["gross_usd"] == pytest.approx(gross, abs=0.01) and t["net_usd"] == pytest.approx(long - short, abs=0.01)
    assert t["n_positions"] == len(b["rows"])
    plans = {p["ticker"]: p for p in out["execution"]}
    for r in b["rows"]:  # child orders add up and respect the participation cap
        p = plans[r["ticker"]]
        assert sum(c["shares"] for c in p["child_orders"]) == r["shares"]
        assert max(c["shares"] for c in p["child_orders"]) <= p["max_child_shares"]
        assert r["slices"] == p["n_slices"] == len(p["child_orders"])
    # whatever the note says (model text that passed the audit, or the template), it only quotes numbers in the facts
    assert dm.unsupported_numbers(ex.template_note(b), ex.note_facts(b)) == []
    assert dm.unsupported_numbers(out["note"]["text"], ex.note_facts(b)) == []


# --------------------------------------------------------------------------- contract and shape
def test_example_contract():
    check_example(ex)


def test_graph_structure_is_what_the_docstring_draws():
    dg = ex.graph.get_graph()
    nodes = set(dg.nodes) - {"__start__", "__end__"}
    assert nodes == {
        "cio_plan", "ticker_worker", "dossier_review", "bull_case", "bear_case", "arbiter", "desk", "risk_committee",
        "resize_worker", "execution_plan", "desk_note",
    }  # fmt: skip
    edges = {(e.source, e.target) for e in dg.edges}
    assert {("cio_plan", "ticker_worker"), ("ticker_worker", "dossier_review"), ("bull_case", "arbiter"), ("bear_case", "arbiter")} <= edges
    assert {("dossier_review", "bull_case"), ("dossier_review", "bear_case"), ("dossier_review", "desk")} <= edges
    assert {("risk_committee", "resize_worker"), ("resize_worker", "risk_committee"), ("risk_committee", "execution_plan")} <= edges


def test_studio_sees_a_clean_input_form_and_result():
    inp = ex.graph.get_input_jsonschema()["properties"]
    assert set(inp) == {"focus_hint", "nav_usd", "max_revisions"}
    out = set(ex.graph.get_output_jsonschema()["properties"])
    assert {"blotter", "verdict", "note", "risk_history", "resize_log", "cap_forced", "mode", "disclaimer"} <= out
    assert not out & {"legs", "book", "risk", "review"}  # internals stay internal


def overwrite_as_json(obj):
    """How the API server serialises the ``Overwrite`` markers that ``cio_plan`` writes; anything else must be plain JSON."""
    if isinstance(obj, Overwrite):
        return {"value": obj.value, "type": obj.type}
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def test_example_input_runs_offline_end_to_end_and_everything_is_json():
    r = run(ex.EXAMPLE_INPUT)
    out = r.out
    assert set(out) == set(ex.DeskOut.__annotations__)
    assert out["mode"] == "offline" and "FICTIONAL" in out["disclaimer"] and "not investment advice" in out["disclaimer"].lower()
    assert out["mandate"]["source"] == "fallback" and len(out["mandate"]["focus_tickers"]) in (4, 5)
    assert set(out["mandate"]["focus_tickers"]) == {"ACMR", "BORE", "GRNT", "HELO", "CBLT"}  # the hint's names
    assert out["blotter"]["rows"], "the offline stand-ins make at least one name actionable for this input"
    assert_blotter_clean(out)
    json.dumps(out, allow_nan=False)
    for node, update in r.events:  # every step's state update is serialisable (Studio and the API need it)
        json.dumps(update, allow_nan=False, default=overwrite_as_json)
    # selectivity holds in the default run: exactly the names the first committee pass flagged were sent back
    first = r.state["risk_history"][0]
    assert {e["ticker"] for e in out["resize_log"]} == set(first["breaching_names"])
    assert out["note"]["source"] == "model" and out["note"]["text"].startswith("[offline]")


@pytest.mark.parametrize("schema", [ex.CioMandate, ex.BullCase, ex.BearCase, ex.ResizeProposal])
def test_every_structured_output_schema_is_openai_strict_compatible(schema):
    """Offline runs cannot notice a strict-mode problem, so lint every schema this example sends to a model."""
    assert lint_openai_strict(schema) == []


def test_the_try_it_section_of_the_docstring_is_true_offline():
    """TRY IT IN STUDIO promises these behaviours for the offline stand-ins; if a stand-in changes, update the docstring.

    The numbered walkthrough is written against OFFLINE_WALKTHROUGH_INPUT (EXAMPLE_INPUT is tuned for a LIVE model)."""
    base = run(ex.OFFLINE_WALKTHROUGH_INPUT)
    assert base.out["verdict"]["tilt"] == "DEFENSIVE" and base.out["cap_forced"] is False
    first = base.state["risk_history"][0]
    assert first["breaching_names"] == ["DLPH"] and first["book_usd"] == {"DLPH": 1_573_400.0, "GRNT": 1_426_500.0}
    assert [(e["ticker"], e["required_scale"], e["min_scale"]) for e in base.out["resize_log"]] == [("DLPH", 0.9533, 0.9066)]
    assert [r["status"] for r in base.out["blotter"]["rows"]] == ["resized", "proposed"], "GRNT is never touched"
    assert [d["ticker"] for d in base.out["dossiers"] if d["direction"] != "AVOID"] == ["DLPH", "GRNT"]
    zero = run({**ex.OFFLINE_WALKTHROUGH_INPUT, "max_revisions": 0})
    assert zero.out["cap_forced"] is True and zero.out["resize_log"] == [] and not zero.nodes("resize_worker")
    assert [r["status"] for r in zero.out["blotter"]["rows"]] == ["trimmed", "proposed"]
    huge = run({**ex.OFFLINE_WALKTHROUGH_INPUT, "nav_usd": 2_000_000_000}).out
    assert max(r["slices"] for r in huge["blotter"]["rows"]) > 13 and max(p["sessions"] for p in huge["execution"]) > 1


# --------------------------------------------------------------------------- cio_plan
def test_cio_model_pick_is_normalised_validated_and_used(decisions):
    set_structured_handler(
        ex.CioMandate,
        lambda text: ex.CioMandate(focus_tickers=["acmr", "BORE", "BORE", "XXXX", " cblt "], gross_budget=45, rationale="because"),
    )
    r = run({})
    m = r.out["mandate"]
    assert m["focus_tickers"] == ["ACMR", "BORE", "CBLT"] and m["source"] == "model"
    assert m["dropped"] == ["BORE", "XXXX"]  # the duplicate and the unknown ticker
    assert m["gross_budget"] == 0.45 and m["rationale"] == "because"  # "45" is read as 45%
    assert [d["ticker"] for d in r.out["dossiers"]] == ["ACMR", "BORE", "CBLT"]


@pytest.mark.parametrize("raw", [[], ["ZZZ", "YYY", "XXX"], ["ACMR", "BORE", "NOPE"], ["ACMR", "acmr", "ACMR"]])
def test_cio_junk_or_too_few_valid_tickers_fall_back_deterministically(decisions, raw):
    set_structured_handler(ex.CioMandate, lambda text: ex.CioMandate(focus_tickers=raw, gross_budget=0.3, rationale="r"))
    a, b = run({"focus_hint": "software"}).out["mandate"], run({"focus_hint": "software"}).out["mandate"]
    assert a == b and a["source"] == "fallback" and 4 <= len(a["focus_tickers"]) <= 5
    assert a["focus_tickers"][0] == "FTHM"  # the hint mentions the software sector; hinted names come first
    assert set(a["focus_tickers"]) <= set(ALL) and a["gross_budget"] == 0.3  # a valid budget survives a junk ticker list
    assert "fallback" in a["rationale"].lower()


def test_cio_model_error_is_recorded_not_fatal(decisions):
    def boom(text):
        raise RuntimeError("structured output refused")

    set_structured_handler(ex.CioMandate, boom)
    m = run({}).out["mandate"]
    assert m["source"] == "fallback" and "RuntimeError" in m["model_error"] and m["gross_budget"] == dm.DEFAULT_BUDGET


def test_cio_prompt_carries_the_universe_and_the_hint(decisions):
    seen = []
    set_structured_handler(ex.CioMandate, lambda text: seen.append(text) or ex.CioMandate(focus_tickers=["ACMR", "BORE", "CBLT"], gross_budget=0.4, rationale="r"))
    run({"focus_hint": "defensive names", "nav_usd": 2_500_000})
    (text,) = seen
    assert "PM HINT: defensive names" in text and "NAV USD: 2,500,000" in text
    assert all(f"{t} |" in text for t in ALL) and "group technology" in text and "FICTIONAL" in text


def test_input_normalisation_and_validation():
    assert ex.cio_plan({})["nav_usd"] == ex.DEFAULT_NAV and ex.cio_plan({})["max_revisions"] == ex.DEFAULT_MAX_REVISIONS
    assert ex.cio_plan({"max_revisions": 99})["max_revisions"] == ex.MAX_REVISIONS_LIMIT
    assert ex.cio_plan({"max_revisions": -4})["max_revisions"] == 0
    assert ex.cio_plan({"max_revisions": 0})["max_revisions"] == 0  # zero is a real value, not "missing"
    assert ex.cio_plan({"nav_usd": "2500000"})["nav_usd"] == 2_500_000.0
    for bad in (0, -5, float("nan")):
        with pytest.raises(ValueError, match="nav_usd"):
            ex.cio_plan({"nav_usd": bad})


# --------------------------------------------------------------------------- Send map-reduce
def test_fan_out_sends_one_worker_per_chosen_name():
    sends = ex.fan_out_research({"mandate": {"focus_tickers": ["DLPH", "GRNT", "HELO"]}})
    assert [(s.node, s.arg) for s in sends] == [("ticker_worker", {"ticker": t}) for t in ("DLPH", "GRNT", "HELO")]


def test_each_name_is_researched_once_and_dossiers_merge_in_ticker_order_whoever_finishes_first(decisions):
    choose("ACMR", "BORE", "DLPH", "HELO")
    decisions.delay = {"ACMR": 0.30, "BORE": 0.20, "DLPH": 0.10, "HELO": 0.0}  # finish in REVERSE alphabetical order
    r = run({})
    finished = [u["dossiers"][0]["ticker"] for u in r.nodes("ticker_worker")]
    assert finished == ["HELO", "DLPH", "BORE", "ACMR"], "the test is only meaningful if completion order differs from ticker order"
    assert [d["ticker"] for d in r.out["dossiers"]] == ["ACMR", "BORE", "DLPH", "HELO"]
    assert len(r.nodes("dossier_review")) == 1, "the barrier runs once, after every worker"
    assert len(decisions.ticker_calls()) == 4, "exactly one Decisions call per name"


def test_merge_by_ticker_reducer():
    left = [{"ticker": "B", "v": 1}, {"ticker": "D", "v": 1}]
    right = [{"ticker": "C", "v": 2}, {"ticker": "B", "v": 2}]
    assert ex.merge_by_ticker(left, right) == [{"ticker": "B", "v": 2}, {"ticker": "C", "v": 2}, {"ticker": "D", "v": 1}]
    assert ex.merge_by_ticker(None, right) == [{"ticker": "B", "v": 2}, {"ticker": "C", "v": 2}]
    assert ex.merge_by_ticker(left, None) == left and ex.merge_by_ticker(None, None) == []


def test_transient_decisions_error_is_retried_not_fatal(decisions, monkeypatch):
    monkeypatch.setattr(ex, "DECISIONS_RETRY", RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False))
    choose("ACMR", "BORE", "DLPH")
    decisions.fail_once = {"BORE"}
    r = run({})
    assert [d["ticker"] for d in r.out["dossiers"]] == ["ACMR", "BORE", "DLPH"]
    assert sum(1 for c in decisions.ticker_calls() if c[1] == "BORE") == 2  # failed once, retried once


# --------------------------------------------------------------------------- ticker_worker / Decisions gates
def test_dossier_has_hand_checkable_analytics_visible_probabilities_and_no_labels(decisions):
    choose("ACMR", "BORE", "DLPH")
    r = run({})
    prices, news = read_csv("prices.csv"), read_csv("news_headlines.csv")
    for d in r.out["dossiers"]:
        t = d["ticker"]
        assert d["analytics"] == dm.compute_analytics(prices[prices.ticker == t])
        assert d["analytics"]["asof"] == "2026-10-30"
        assert set(d["direction_probs"]) == {"LONG", "SHORT", "AVOID"} and sum(d["direction_probs"].values()) == pytest.approx(1.0, abs=2e-4)
        assert set(d["conviction_probs"]) == {"low", "medium", "high"} and 0.0 <= d["p_setup"] <= 1.0
        assert 0.0 <= d["conviction_score"] <= 2.0
        assert d["decision_mode"] == "offline" and "offline" in d["decision_model"]
        window = news[(news.ticker == t) & (news.date >= "2026-09-30") & (news.date <= "2026-10-30")]
        assert 0 < len(d["headlines"]) <= 6 and {h["headline"] for h in d["headlines"]} <= set(window.headline)
        assert all(set(h) == {"date", "headline"} for h in d["headlines"])
    ((names, ticker, text),) = [c for c in decisions.ticker_calls() if c[1] == "ACMR"]
    assert names == ("setup_supportive", "direction", "conviction")  # one call, three questions
    heads = text.split("HEADLINES (last 30 days, newest first):\n")[1].splitlines()
    ours = {d["ticker"]: d for d in r.out["dossiers"]}["ACMR"]["headlines"]
    assert "RSI(14):" in text and heads == [f"- {h['date']}  {h['headline']}" for h in ours], "headlines only; no sentiment labels"


@pytest.mark.parametrize("direction", ["LONG", "SHORT", "AVOID"])
def test_decision_override_pins_the_direction(decisions, direction):
    choose("ACMR", "DLPH", "GRNT")
    decisions.default_direction = direction
    out = run({}).out
    assert {d["direction"] for d in out["dossiers"]} == {direction}
    assert {d["direction_pick"] for d in out["dossiers"]} == {direction}
    if direction == "AVOID":
        assert out["blotter"]["rows"] == []
    else:
        assert {r["side"] for r in out["blotter"]["rows"]} == {direction}
        assert {r["order"] for r in out["blotter"]["rows"]} == {"BUY" if direction == "LONG" else "SELL SHORT"}


def test_setup_gate_turns_a_pick_into_avoid_at_the_threshold(decisions):
    choose("ACMR", "DLPH", "GRNT")
    decisions.default_direction = "LONG"
    decisions.setup = {"ACMR": 0.10, "DLPH": 0.40, "GRNT": 0.95}
    r = run({})
    by = {d["ticker"]: d for d in r.out["dossiers"]}
    assert (by["ACMR"]["direction_pick"], by["ACMR"]["direction"], by["ACMR"]["gated"]) == ("LONG", "AVOID", True)
    assert (by["DLPH"]["direction"], by["DLPH"]["gated"]) == ("LONG", False)  # exactly at the gate still passes
    assert by["GRNT"]["direction"] == "LONG"
    assert r.state["review"]["gated"] == ["ACMR"] and r.state["review"]["standing_aside"] == ["ACMR"]
    assert "ACMR" not in {leg["ticker"] for leg in r.state["legs"]} and "ACMR" in r.out["blotter"]["standing_aside"]


def test_conviction_scales_the_size_through_the_expected_level(decisions):
    choose("DLPH", "EMBR", "GRNT")
    decisions.conviction = {"DLPH": "low", "EMBR": "medium", "GRNT": "high"}
    out = run({}).out
    score = {d["ticker"]: d["conviction_score"] for d in out["dossiers"]}
    assert score["DLPH"] < score["EMBR"] < score["GRNT"]
    assert score["EMBR"] == pytest.approx(1.0, abs=0.01) and score["GRNT"] == pytest.approx(1.85, abs=0.01)  # 0.05*0 + 0.9*1 + 0.05*2 and 0.05 + 0.9*2


def test_refused_answers_fail_safe(decisions):
    choose("ACMR", "DLPH", "GRNT")
    decisions.refuse = {"direction"}
    r = run({})
    assert all(d["direction"] == "AVOID" and d["direction_pick"] is None and "direction" in d["refused"] for d in r.out["dossiers"])
    assert r.out["blotter"]["rows"] == []
    decisions.refuse = {"setup_supportive"}  # P(setup) reads as 0 -> the gate closes even for a LONG pick
    r2 = run({})
    assert all(d["direction"] == "AVOID" and d["gated"] for d in r2.out["dossiers"])


def test_nothing_actionable_skips_the_debate_and_the_arbiter_and_stands_aside(decisions):
    choose("ACMR", "DLPH", "GRNT")
    decisions.default_direction = "AVOID"
    bull_calls, bear_calls = [], []
    set_structured_handler(ex.BullCase, lambda t: bull_calls.append(t))
    set_structured_handler(ex.BearCase, lambda t: bear_calls.append(t))
    chat_calls = []
    set_chat_handler(lambda messages, tools: chat_calls.append(messages) or AIMessage(content="should not be called"))
    r = run({})
    out = r.out
    assert not bull_calls and not bear_calls and not r.nodes("bull_case") and not r.nodes("bear_case") and not r.nodes("arbiter")
    assert decisions.tilt_calls() == [], "no arbiter Decisions call when there is nothing to debate"
    assert out["verdict"]["skipped"] is True and out["verdict"]["tilt"] == "NEUTRAL"
    assert out["blotter"]["rows"] == [] and out["blotter"]["totals"]["n_positions"] == 0 and out["blotter"]["totals"]["gross_usd"] == 0.0
    assert out["resize_log"] == [] and out["cap_forced"] is False
    assert out["note"]["source"] == "template" and "standing aside" in out["note"]["text"]
    assert not chat_calls, "an empty blotter needs no model-written note"


# --------------------------------------------------------------------------- debate and arbiter
def test_bull_and_bear_run_concurrently_and_the_arbiter_reads_both(decisions):
    choose("ACMR", "BORE", "CBLT", "DLPH", budget=0.5)
    barrier = threading.Barrier(2, timeout=5)  # both handlers wait for the other: sequential execution would break it

    def case(cls, marker, ideas):
        def handler(text):
            barrier.wait()
            return cls(thesis=marker, ideas=ideas, key_risks=["r1", "r2", "r3", "r4"][:3], anticipated_rebuttal=f"{marker} rebuttal")

        return handler

    bull_ideas = [ex.Idea(ticker=t, action="LONG", argument=f"a {t}") for t in ("acmr", "ZZZZ", "BORE", "CBLT", "DLPH")]
    bear_ideas = [ex.Idea(ticker="dlph", action="SHORT", argument="b")]
    set_structured_handler(ex.BullCase, case(ex.BullCase, "BULL-THESIS-MARKER", bull_ideas))
    set_structured_handler(ex.BearCase, case(ex.BearCase, "BEAR-THESIS-MARKER", bear_ideas))
    r = run({})
    assert r.out["bull_argument"]["thesis"] == "BULL-THESIS-MARKER" and r.out["bear_argument"]["thesis"] == "BEAR-THESIS-MARKER"
    assert [i["ticker"] for i in r.out["bull_argument"]["ideas"]] == ["ACMR", "BORE", "CBLT"]  # unknown ticker dropped, capped at 3, upper-cased
    assert r.out["bear_argument"]["ideas"][0]["ticker"] == "DLPH" and r.out["bear_argument"]["side"] == "bear"
    ((_, _, evidence),) = decisions.tilt_calls()
    assert "BULL-THESIS-MARKER" in evidence and "BEAR-THESIS-MARKER" in evidence and "BULL-THESIS-MARKER rebuttal" in evidence
    assert all(f"TICKER: {t}" in evidence for t in ("ACMR", "BORE", "CBLT", "DLPH")), "the arbiter sees the dossiers too"
    order = [n for n, _ in r.events]
    assert order.index("arbiter") > max(order.index("bull_case"), order.index("bear_case")), "the join waits for both cases"


def test_debate_prompts_are_one_sided_and_name_the_fiction():
    assert "BULL" in ex.BULL_PROMPT and "BEAR" in ex.BEAR_PROMPT
    for p in (ex.BULL_PROMPT, ex.BEAR_PROMPT, ex.CIO_PROMPT, ex.RESIZE_PROMPT, ex.NOTE_PROMPT):
        assert "FICTIONAL" in p
    assert "LONG" in ex.BULL_PROMPT and "SHORT or AVOID" in ex.BEAR_PROMPT and "Never invent" in ex.BULL_PROMPT


@pytest.mark.parametrize("tilt, scale", [("NET_LONG", 1.0), ("NEUTRAL", 0.8), ("DEFENSIVE", 0.6)])
def test_decision_override_pins_the_tilt_and_the_tilt_scales_gross(decisions, tilt, scale):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.5)
    set_decision_override("tilt", tilt)
    r = run({"nav_usd": NAV})
    v, b = r.out["verdict"], r.out["blotter"]
    assert (v["tilt"], v["tilt_pick"], v["tail_override"], v["gross_scale"]) == (tilt, tilt, False, scale)
    assert max(v["tilt_probs"], key=v["tilt_probs"].get) == tilt
    assert b["tilt"] == tilt and b["tilt_gross_scale"] == scale
    assert b["gross_budget_usd"] == dm.floor_lot(NAV * 0.5 * scale)
    assert r.state["book"]["effective_gross_usd"] == b["gross_budget_usd"]
    proposed = sum(l["history"][0]["notional_usd"] for l in r.state["legs"])
    assert b["gross_budget_usd"] - 100 * len(r.state["legs"]) < proposed <= b["gross_budget_usd"]  # the desk uses the whole budget (rounding aside)


def test_tilt_changes_the_long_short_mix(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.5)
    decisions.direction = {"ACMR": "LONG", "BORE": "LONG", "DLPH": "SHORT", "EMBR": "SHORT"}

    def mix(tilt):
        set_decision_override("tilt", tilt)
        legs = run({}).state["legs"]
        first = {l["ticker"]: l["history"][0]["notional_usd"] for l in legs}
        long = sum(v for t, v in first.items() if t in ("ACMR", "BORE"))
        return long / sum(first.values())

    assert mix("NET_LONG") > mix("NEUTRAL") > mix("DEFENSIVE")  # net-long tilt favours the long book, defensive the shorts


@pytest.mark.parametrize(
    "pick, p_tail, expected, overridden",
    [
        ("NET_LONG", 0.95, "NEUTRAL", True),
        ("NET_LONG", 0.70, "NEUTRAL", True),
        ("NET_LONG", 0.69, "NET_LONG", False),
        ("NEUTRAL", 0.95, "DEFENSIVE", True),
        ("DEFENSIVE", 0.99, "DEFENSIVE", False),
    ],
)
def test_high_tail_risk_moves_the_tilt_one_notch_toward_defensive(decisions, pick, p_tail, expected, overridden):
    choose("ACMR", "BORE", "DLPH")
    set_decision_override("tilt", pick)
    set_decision_override("tail_risk_elevated", p_tail)
    v = run({}).out["verdict"]
    assert (v["tilt"], v["tilt_pick"], v["tail_override"], v["tail_risk_p"]) == (expected, pick, overridden, p_tail)


def test_refused_tilt_fails_safe_to_defensive(decisions):
    choose("ACMR", "BORE", "DLPH")
    decisions.refuse = {"tilt"}
    v = run({}).out["verdict"]
    assert v["tilt"] == "DEFENSIVE" and "tilt" in v["refused"]


def test_rigor_scores_grade_the_debate_but_do_not_move_money(decisions):
    choose("ACMR", "BORE", "DLPH")
    set_decision_override("bull_rigor", "strong")
    set_decision_override("bear_rigor", "weak")
    strong = run({}).out
    set_decision_override("bull_rigor", "weak")
    set_decision_override("bear_rigor", "strong")
    weak = run({}).out
    assert strong["verdict"]["winner"] == "bull" and weak["verdict"]["winner"] == "bear"
    assert strong["verdict"]["bull_rigor"]["level"] == "strong" and strong["verdict"]["bear_rigor"]["level"] == "weak"
    assert strong["blotter"]["rows"] == weak["blotter"]["rows"]  # same tilt, same book


# --------------------------------------------------------------------------- desk sizing
def test_desk_sizing_is_conviction_weighted_inverse_volatility_by_hand(decisions):
    set_decision_override("tilt", "NEUTRAL")
    dossiers = [
        {"ticker": "AAA", "direction": "LONG", "conviction_score": 1.0, "group": "technology", "analytics": {"vol_20d": 0.02, "last_close": 50.0}},
        {"ticker": "BBB", "direction": "LONG", "conviction_score": 0.0, "group": "technology", "analytics": {"vol_20d": 0.02, "last_close": 80.0}},
        {"ticker": "CCC", "direction": "SHORT", "conviction_score": 2.0, "group": "defensives", "analytics": {"vol_20d": 0.03, "last_close": 20.0}},
        {"ticker": "ZZZ", "direction": "AVOID", "conviction_score": 2.0, "group": "defensives", "analytics": {"vol_20d": 0.01, "last_close": 20.0}},
    ]
    state = {"nav_usd": NAV, "verdict": {"tilt": "NEUTRAL"}, "mandate": {"gross_budget": 0.5}, "dossiers": dossiers}
    update = ex.desk(state)
    # effective gross = 10M * 0.5 * 0.8 = 4,000,000; raw weights 50 : 25 : 50 -> 1,600,000 / 800,000 / 1,600,000
    assert {l["ticker"]: l["notional_usd"] for l in update["legs"]} == {"AAA": 1_600_000, "BBB": 800_000, "CCC": 1_600_000}
    assert update["book"]["effective_gross_usd"] == 4_000_000 and update["book"]["n_legs"] == 3 and update["book"]["tilt"] == "NEUTRAL"


def test_avoid_names_never_get_positions(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.5)
    decisions.default_direction = "LONG"
    decisions.direction = {"ACMR": "AVOID", "EMBR": "AVOID"}
    r = run({})
    held = {l["ticker"] for l in r.state["legs"]}
    assert held == {"BORE", "DLPH"}
    assert {x["ticker"] for x in r.out["blotter"]["rows"]} <= {"BORE", "DLPH"}
    assert r.out["blotter"]["standing_aside"] == ["ACMR", "EMBR"]
    assert all(h["book_usd"].keys() <= {"BORE", "DLPH"} for h in r.out["risk_history"])
    assert_blotter_clean(r.out)


def test_the_desk_is_not_limit_aware_so_concentration_reaches_the_committee(decisions):
    """Three technology names at a 45%+ budget put about 45% of NAV in one group against a 35% limit: that is the point
    of the separation of duties, so the first committee pass must flag the group."""
    choose("ACMR", "CBLT", "FTHM", budget=0.5)
    set_decision_override("tilt", "NET_LONG")  # gross scale 1.0 -> 5,000,000 across three technology longs
    r = run({})
    first = r.state["risk_history"][0]
    assert first["action"] == "resize"
    assert first["group_pct_nav"]["technology"] > 0.35
    assert any("group 'technology'" in b for b in first["breaches"])
    assert set(first["breaching_names"]) >= {"ACMR", "CBLT", "FTHM"}


# --------------------------------------------------------------------------- risk committee: selective re-dispatch
def committee_state(sizes, *, budget=5_600_000.0, revision=0, cap=2, avoid=()):
    dossiers = [ex.ticker_worker({"ticker": t})["dossiers"][0] for t in sorted(sizes)]
    for d in dossiers:
        d["direction"] = "AVOID" if d["ticker"] in avoid else "LONG"
    return {
        "nav_usd": NAV, "max_revisions": cap, "revision_count": revision, "dossiers": dossiers,
        "legs": [make_leg(t, s, n) for t, (s, n) in sorted(sizes.items())], "book": {"effective_gross_usd": budget}, "risk_history": [],
    }  # fmt: skip


def test_router_sends_only_the_breaching_name_to_a_resize_worker():
    state = committee_state({"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    update = ex.risk_committee(state)
    assert update["risk"]["action"] == "resize" and update["revision_count"] == 1 and update["risk"]["to_resize"] == {"ACMR": 0.75}
    sends = ex.route_after_risk({**state, **update})
    assert len(sends) == 1 and all(isinstance(s, Send) and s.node == "resize_worker" for s in sends)
    job = sends[0].arg
    assert job["leg"]["ticker"] == "ACMR" and job["required_scale"] == 0.75 and job["round"] == 1 and job["max_revisions"] == 2
    assert [b["code"] for b in job["breaches"]] == ["SINGLE_NAME"] and "direction LONG" in job["research"]


def test_router_goes_to_execution_when_the_book_is_clean_or_the_cap_trimmed_it():
    clean = committee_state({"ACMR": ("LONG", 1_000_000), "BORE": ("LONG", 1_000_000)})
    upd = ex.risk_committee(clean)
    assert upd["risk"]["action"] == "pass" and "revision_count" not in upd and "cap_forced" not in upd
    assert ex.route_after_risk({**clean, **upd}) == "execution_plan"
    capped = committee_state({"ACMR": ("LONG", 2_000_000)}, revision=2, cap=2)
    upd = ex.risk_committee(capped)
    assert upd["risk"]["action"] == "trim" and upd["cap_forced"] is True and "revision_count" not in upd
    assert [l["notional_usd"] for l in upd["legs"]] == [1_500_000] and upd["legs"][0]["status"] == "trimmed"
    assert ex.route_after_risk({**capped, **upd}) == "execution_plan"


def test_a_group_breach_implicates_every_member_of_the_group_and_nobody_else():
    sizes = {"ACMR": ("LONG", 1_300_000), "CBLT": ("LONG", 1_300_000), "FTHM": ("LONG", 1_300_000), "DLPH": ("LONG", 1_000_000)}
    state = committee_state(sizes)  # technology = 3.9M = 39% (limit 35%); no single name above 15%
    upd = ex.risk_committee(state)
    assert [b["code"] for b in upd["risk"]["breaches"]] == ["SECTOR_GROUP"]
    assert upd["risk"]["to_resize"] == {"ACMR": 0.8974, "CBLT": 0.8974, "FTHM": 0.8974}  # 3.5M / 3.9M rounded down to 4 d.p.
    assert sorted(s.arg["leg"]["ticker"] for s in ex.route_after_risk({**state, **upd})) == ["ACMR", "CBLT", "FTHM"]


def test_only_the_breaching_name_is_resized_in_a_full_run_and_the_others_are_untouched(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    set_decision_override("tilt", "NEUTRAL")  # effective gross 10M * 0.7 * 0.8 = 5.6M
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    model = diligent()
    r = run({})
    assert [c[0] for c in model.calls] == ["ACMR"], "exactly one resize worker, for the one breaching name"
    assert [u["resize_log"][0]["ticker"] for u in r.nodes("resize_worker")] == ["ACMR"]
    assert len(r.nodes("risk_committee")) == 2 and [h["action"] for h in r.out["risk_history"]] == ["resize", "pass"]
    legs = {l["ticker"]: l for l in r.state["legs"]}
    for t in ("BORE", "DLPH", "EMBR"):
        assert legs[t]["notional_usd"] == 1_000_000 and legs[t]["status"] == "proposed" and len(legs[t]["history"]) == 1
    assert legs["ACMR"]["notional_usd"] == 1_500_000 and legs["ACMR"]["status"] == "resized"
    assert [h["step"] for h in legs["ACMR"]["history"]] == ["desk", "resize round 1"]
    (entry,) = r.out["resize_log"]
    assert (entry["round"], entry["required_scale"], entry["proposed_scale"], entry["applied_scale"]) == (1, 0.75, 0.75, 0.75)
    assert (entry["before_usd"], entry["after_usd"], entry["source"], entry["breaches"]) == (2_000_000, 1_500_000, "model", ["SINGLE_NAME"])
    assert r.out["revision_count"] == 1 and r.out["cap_forced"] is False
    assert_blotter_clean(r.out)


def test_the_resize_prompt_names_one_position_and_states_the_cure(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    model = diligent()
    run({})
    (_, _, text) = model.calls[0]
    assert "TICKER: ACMR" in text and "CURRENT NOTIONAL USD: 2,000,000 (20.0% of NAV)" in text
    assert "REQUIRED_SCALE: 0.7500" in text and "MIN_SCALE: 0.5000" in text
    assert "REVISION ROUND: 1 of 2" in text and "ACMR LONG 2,000,000 is 20.0% of NAV (limit 15%)" in text
    for other in ("BORE", "DLPH", "EMBR"):
        assert f"TICKER: {other}" not in text and f"{other} LONG" not in text, "a resize worker sees only its own position"


def test_a_group_breach_resizes_the_whole_group_and_leaves_other_groups_alone(decisions, monkeypatch):
    choose("ACMR", "CBLT", "FTHM", "DLPH", budget=0.7)
    set_decision_override("tilt", "NEUTRAL")
    fixed_book(monkeypatch, {"ACMR": ("LONG", 1_300_000), "CBLT": ("LONG", 1_300_000), "FTHM": ("LONG", 1_300_000), "DLPH": ("LONG", 1_000_000)})
    model = diligent()
    r = run({})
    assert sorted(c[0] for c in model.calls) == ["ACMR", "CBLT", "FTHM"] and {c[1] for c in model.calls} == {0.8974}
    assert {e["ticker"] for e in r.out["resize_log"]} == {"ACMR", "CBLT", "FTHM"} and all(e["breaches"] == ["SECTOR_GROUP"] for e in r.out["resize_log"])
    assert r.state["risk_history"][-1]["group_pct_nav"]["technology"] <= 0.35 and r.out["blotter"]["totals"]["group_pct_nav"]["technology"] <= 0.35
    assert {l["ticker"]: l["notional_usd"] for l in r.state["legs"]}["DLPH"] == 1_000_000
    assert r.out["cap_forced"] is False
    assert_blotter_clean(r.out)


def test_a_gross_breach_implicates_every_live_leg(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.25)
    set_decision_override("tilt", "DEFENSIVE")  # effective gross = 10M * 0.25 * 0.6 = 1.5M
    fixed_book(monkeypatch, {"ACMR": ("LONG", 800_000), "BORE": ("LONG", 800_000), "DLPH": ("LONG", 400_000), "EMBR": ("LONG", 0.0)})
    model = diligent()
    r = run({})
    assert [b.split(" ")[0] for b in r.out["risk_history"][0]["breaches"]] == ["book"]
    assert sorted(c[0] for c in model.calls) == ["ACMR", "BORE", "DLPH"]  # the zero-notional leg is not "in" the breach
    assert r.out["blotter"]["totals"]["gross_usd"] <= 1_500_000
    assert_blotter_clean(r.out)


# --------------------------------------------------------------------------- the loop is bounded
@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_the_loop_stops_at_max_revisions_and_records_that_the_cap_forced_the_trim(decisions, monkeypatch, cap):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    model = lazy(0.99)  # a PM who keeps proposing a 1% cut
    r = run({"max_revisions": cap})
    actions = [h["action"] for h in r.out["risk_history"]]
    assert actions == ["resize"] * cap + ["trim"], "one committee pass per round, then the deterministic trim"
    assert len(model.calls) == cap and [e["round"] for e in r.out["resize_log"]] == list(range(1, cap + 1))
    assert r.out["revision_count"] == cap and r.out["cap_forced"] is True and r.out["blotter"]["cap_forced"] is True
    assert r.out["blotter"]["revisions_used"] == cap and r.out["blotter"]["max_revisions"] == cap
    trim = r.state["risk"]["trims"][0]
    assert trim["ticker"] == "ACMR" and 1_499_000 < trim["after_usd"] <= 1_500_000  # cure scale rounds down: a hair under the cap
    (row,) = [x for x in r.out["blotter"]["rows"] if x["ticker"] == "ACMR"]
    assert row["status"] == "trimmed" and row["notional_usd"] <= 1_500_000
    acmr = {l["ticker"]: l for l in r.state["legs"]}["ACMR"]
    assert [h["step"] for h in acmr["history"]] == ["desk"] + [f"resize round {i}" for i in range(1, cap + 1)] + ["cap trim"]
    assert "REVISION CAP FORCED A TRIM: yes" in ex.note_facts(r.out["blotter"])
    assert_blotter_clean(r.out)


def test_a_diligent_pm_ends_the_loop_early_without_a_forced_trim(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    diligent()
    r = run({"max_revisions": 5})
    assert [h["action"] for h in r.out["risk_history"]] == ["resize", "pass"] and r.out["revision_count"] == 1 and r.out["cap_forced"] is False
    assert "REVISION CAP FORCED A TRIM: no" in ex.note_facts(r.out["blotter"])


def test_an_undercut_is_accepted_and_goes_round_again_but_a_wild_proposal_is_bounded(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    proposals = iter([0.9, 0.01])  # round 1: too little; round 2: absurdly deep
    install_resizer(lambda required: next(proposals))
    r = run({"max_revisions": 3})
    log = r.out["resize_log"]
    # round 2 needs 1.5M / 1.8M = 0.8333 (a 16.7% cut); the model's absurd 0.01 is lifted to min_scale = 2 * 0.8333 - 1 = 0.6666 (a 33.3% cut)
    assert [(e["proposed_scale"], e["applied_scale"]) for e in log] == [(0.9, 0.9), (0.01, 0.6666)]
    assert log[0]["after_usd"] == 1_800_000 and log[1]["before_usd"] == 1_800_000 and log[1]["required_scale"] == 0.8333 and log[1]["min_scale"] == 0.6666
    assert log[1]["after_usd"] == dm.floor_lot(1_800_000 * 0.6666) == 1_199_800  # deeper than needed, but bounded
    assert r.out["cap_forced"] is False and r.out["revision_count"] == 2


def test_junk_and_inflating_proposals_never_increase_a_position(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    install_resizer(lambda required: 3.0)  # tries to INCREASE the position
    r = run({"max_revisions": 1})
    assert [e["applied_scale"] for e in r.out["resize_log"]] == [1.0]
    assert r.out["resize_log"][0]["after_usd"] == 2_000_000  # unchanged; then the cap trims it
    assert r.out["cap_forced"] is True and {l["ticker"]: l["notional_usd"] for l in r.state["legs"]}["ACMR"] == 1_500_000
    install_resizer(lambda required: float("nan"))
    r2 = run({"max_revisions": 1})
    assert r2.out["resize_log"][0]["proposed_scale"] is None and r2.out["resize_log"][0]["applied_scale"] == 1.0


def test_the_graph_always_terminates_within_a_small_recursion_limit(decisions, monkeypatch):
    """Worst case: every round breaches and the PM never helps. max_revisions=5 is the largest accepted value."""
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    fixed_book(monkeypatch, {"ACMR": ("LONG", 2_000_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    lazy(1.0)
    r = run({"max_revisions": 99}, recursion_limit=35)  # 99 is clamped to 5: 6 committee passes + 5 resize rounds + 8 other supersteps
    assert r.out["revision_count"] == 5 and r.out["cap_forced"] is True and len(r.out["risk_history"]) == 6
    with pytest.raises(GraphRecursionError):  # the limit really is what bounds the loop in this test, not luck
        run({"max_revisions": 99}, recursion_limit=15)


# --------------------------------------------------------------------------- forbidden positions
def test_committee_drops_a_forbidden_short_by_rule_without_asking_the_model(decisions, monkeypatch):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.7)
    decisions.default_direction = "LONG"
    decisions.direction = {"ACMR": "AVOID"}
    # a misbehaving desk opens a SHORT in the name research marked AVOID; the committee must catch it anyway
    fixed_book(monkeypatch, {"ACMR": ("SHORT", 800_000), "BORE": ("LONG", 1_000_000), "DLPH": ("LONG", 1_000_000), "EMBR": ("LONG", 1_000_000)})
    model = diligent()
    r = run({})
    first = r.out["risk_history"][0]
    assert first["action"] == "resize" and first["breaching_names"] == ["ACMR"] and "research marked it AVOID" in first["breaches"][0]
    assert model.calls == [], "a position the rules forbid is dropped without a model call"
    (entry,) = r.out["resize_log"]
    assert (entry["source"], entry["applied_scale"], entry["after_usd"], entry["breaches"]) == ("rule", 0.0, 0.0, ["AVOID_SHORT"])
    assert {l["ticker"]: l for l in r.state["legs"]}["ACMR"]["status"] == "dropped"
    assert "ACMR" not in {x["ticker"] for x in r.out["blotter"]["rows"]} and r.out["cap_forced"] is False
    assert_blotter_clean(r.out)


def test_execution_plan_refuses_a_book_that_fails_the_limits():
    state = {
        "nav_usd": NAV, "max_revisions": 2, "revision_count": 0, "book": {"effective_gross_usd": 9e9, "tilt": "NEUTRAL", "gross_scale": 0.8},
        "dossiers": [ex.ticker_worker({"ticker": "ACMR"})["dossiers"][0]], "legs": [make_leg("ACMR", "LONG", 3_000_000)],
    }  # fmt: skip
    with pytest.raises(ValueError, match="refusing to build a blotter that breaches limits"):
        ex.execution_plan(state)


# --------------------------------------------------------------------------- execution
def test_execution_plan_matches_the_slicing_rules(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.5)
    r = run({})
    plans = {p["ticker"]: p for p in r.out["execution"]}
    dossiers = {d["ticker"]: d for d in r.out["dossiers"]}
    assert plans and set(plans) == {x["ticker"] for x in r.out["blotter"]["rows"]}
    for row in r.out["blotter"]["rows"]:
        a = dossiers[row["ticker"]]["analytics"]
        assert row["shares"] == int(row["notional_usd"] // a["last_close"]) or abs(row["shares"] * a["last_close"] - row["notional_usd"]) < 0.01
        expected = dm.plan_execution(row["shares"], a["last_close"], a["adv_20d_shares"], a["vol_20d"])
        assert plans[row["ticker"]] == {"ticker": row["ticker"], **expected}
        assert row["est_cost_bps"] == expected["est_cost_bps"] and row["slices"] == expected["n_slices"]
    t = r.out["blotter"]["totals"]
    assert t["est_cost_usd"] == pytest.approx(sum(x["est_cost_usd"] for x in r.out["blotter"]["rows"]), abs=0.01)
    assert t["est_cost_bps"] == pytest.approx(t["est_cost_usd"] / t["gross_usd"] * 1e4, abs=0.01)
    assert_blotter_clean(r.out)


def test_a_bigger_fund_needs_many_more_child_orders_and_spills_over_sessions(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR", budget=0.5)
    small = run({"nav_usd": 10_000_000}).out
    huge = run({"nav_usd": 2_000_000_000}).out
    s_slices = {x["ticker"]: x["slices"] for x in small["blotter"]["rows"]}
    h_slices = {x["ticker"]: x["slices"] for x in huge["blotter"]["rows"]}
    assert all(h_slices[t] > s_slices[t] for t in s_slices if t in h_slices) and max(h_slices.values()) > 13
    assert max(p["sessions"] for p in huge["execution"]) > 1
    assert huge["blotter"]["totals"]["est_cost_bps"] > small["blotter"]["totals"]["est_cost_bps"]  # impact grows with participation
    assert_blotter_clean(huge)


# --------------------------------------------------------------------------- the desk note
def echo_gross(messages, tools):
    facts = messages[-1].content
    gross = re.search(r"GROSS: ([\d,]+) USD \(([\d.]+)% of NAV\)", facts)
    return AIMessage(content=f"Gross exposure is {gross.group(1)} USD, {gross.group(2)}% of NAV.")


def test_a_note_that_only_quotes_the_blotter_is_kept(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR")
    set_chat_handler(echo_gross)
    note = run({}).out["note"]
    assert note["source"] == "model" and note["unsupported_numbers"] == [] and note["text"].startswith("Gross exposure is ")


@pytest.mark.parametrize("sentence, bad", [("Expect a 7919 USD windfall.", ["7919"]), ("Gross is about $1.2M.", ["1.2"]), ("We hold 99 positions.", ["99"])])
def test_a_note_with_a_number_that_is_not_in_the_blotter_is_replaced_by_the_template(decisions, sentence, bad):
    choose("ACMR", "BORE", "DLPH", "EMBR")
    set_chat_handler(lambda messages, tools: AIMessage(content=sentence))
    out = run({}).out
    n = out["note"]
    assert n["source"] == "template" and n["unsupported_numbers"] == bad and n["rejected_model_note"] == sentence
    assert n["text"] == ex.template_note(out["blotter"]) and dm.unsupported_numbers(n["text"], ex.note_facts(out["blotter"])) == []


def test_an_empty_model_note_is_replaced_too(decisions):
    choose("ACMR", "BORE", "DLPH")
    set_chat_handler(lambda messages, tools: AIMessage(content="   "))
    assert run({}).out["note"]["source"] == "template"


def test_note_prompt_gives_the_model_only_the_facts(decisions):
    choose("ACMR", "BORE", "DLPH", "EMBR")
    seen = []
    set_chat_handler(lambda messages, tools: seen.append(messages) or AIMessage(content="ok"))
    out = run({}).out
    system, human = seen[0][0].content, seen[0][-1].content
    assert system == ex.NOTE_PROMPT and "copying it exactly" in system
    assert ex.note_facts(out["blotter"]) in human
    assert re.search(r"\d", human.split("FACTS")[0]) is None, "the preface carries no digits"
    assert "dossier" not in human.lower() and "headline" not in human.lower()


def test_template_note_and_facts_agree_for_every_shape_of_blotter():
    rows = [
        {"ticker": "ACMR", "side": "LONG", "shares": 7_300, "price": 205.19, "notional_usd": 1_497_887.0, "pct_nav": 0.149789, "slices": 2, "est_cost_bps": 12.3},
        {"ticker": "HELO", "side": "SHORT", "shares": 10_000, "price": 74.01, "notional_usd": 740_100.0, "pct_nav": 0.07401, "slices": 1, "est_cost_bps": 9.9},
    ]  # fmt: skip
    for cap_forced, standing in [(False, []), (True, ["BORE"])]:
        b = {
            "nav_usd": NAV, "tilt": "DEFENSIVE", "tilt_gross_scale": 0.6, "rows": rows, "standing_aside": standing,
            "totals": {"n_positions": 2, "gross_usd": 2_237_987.0, "net_usd": 757_787.0, "gross_pct_nav": 0.223799, "net_pct_nav": 0.075779, "est_cost_usd": 2_100.0, "est_cost_bps": 9.4},
            "revisions_used": 2, "max_revisions": 2, "cap_forced": cap_forced,
        }  # fmt: skip
        facts, note = ex.note_facts(b), ex.template_note(b)
        assert dm.unsupported_numbers(note, facts) == []
        assert "LARGEST POSITION: LONG ACMR 1,497,887 USD (15.0% of NAV)" in facts and ("forced a deterministic trim" in note) is cap_forced
        assert ("Standing aside: BORE" in note) is bool(standing)


# --------------------------------------------------------------------------- the invariants, fuzzed
@pytest.mark.parametrize("seed", range(12))
def test_final_blotter_respects_every_limit_whatever_the_models_decide(decisions, seed):
    rng = random.Random(seed)
    focus = rng.sample(ALL, rng.randint(3, 5))
    choose(*focus, budget=rng.choice([0.25, 0.45, 0.7, 45]))
    decisions.direction = {t: rng.choice(["LONG", "LONG", "SHORT", "AVOID"]) for t in ALL}
    decisions.conviction = {t: rng.choice(["low", "medium", "high"]) for t in ALL}
    decisions.setup = {t: rng.choice([0.1, 0.5, 0.9]) for t in ALL}
    set_decision_override("tilt", rng.choice(["NET_LONG", "NEUTRAL", "DEFENSIVE"]))
    set_decision_override("tail_risk_elevated", rng.choice([0.1, 0.9]))
    install_resizer(rng.choice([lambda req: req, lambda req: 0.99, lambda req: req * 0.8, lambda req: 1.0]))
    payload = {"nav_usd": rng.choice([2e6, 1e7, 3.5e7, 2e9]), "max_revisions": rng.randint(0, 3)}
    r = run(payload)
    out = r.out
    assert_blotter_clean(out)
    assert out["revision_count"] <= payload["max_revisions"] and len(out["risk_history"]) <= payload["max_revisions"] + 1
    assert out["cap_forced"] == (out["risk_history"][-1]["action"] == "trim")
    assert {e["ticker"] for e in out["resize_log"]} <= {t for h in out["risk_history"] for t in h["breaching_names"]}
    held = {l["ticker"] for l in r.state["legs"]}
    assert all(d["ticker"] not in held for d in out["dossiers"] if d["direction"] == "AVOID"), "an AVOID name never gets a position"
    json.dumps(out, allow_nan=False)


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
def choose_from_hint():
    """A scripted CIO that picks the tickers named in the PM hint, so each run on a thread can have its own names."""

    def pick(text):
        hint = re.search(r"PM HINT: (.*)", text).group(1)
        names = [t for t in re.findall(r"[A-Z]{4}", hint) if t in ALL]
        return ex.CioMandate(focus_tickers=names, gross_budget=0.5, rationale="r")

    set_structured_handler(ex.CioMandate, pick)


FIRST_HINT = {"focus_hint": "Look at ACMR, BORE, GRNT, HELO and CBLT today", "nav_usd": NAV, "max_revisions": 2}


@pytest.mark.parametrize(
    "second_hint, avoid",
    [
        ("Look at DLPH, EMBR and FTHM today", ()),  # a different set of names, all tradable
        ("Look at DLPH, EMBR and FTHM today", ("DLPH", "EMBR", "FTHM")),  # nothing actionable: no debate this time
    ],
    ids=["other-names", "other-names-all-avoid"],
)
def test_a_second_hint_on_the_same_thread_is_the_same_as_on_a_fresh_thread(decisions, second_hint, avoid):
    choose_from_hint()
    diligent()
    decisions.direction.update({t: "AVOID" for t in avoid})
    second = {**FIRST_HINT, "focus_hint": second_hint}
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), FIRST_HINT, second)

    focus = ["DLPH", "EMBR", "FTHM"]
    assert [d["ticker"] for d in reused.output["dossiers"]] == focus, "dossiers of the first run must not survive"
    assert {r["ticker"] for r in reused.output["blotter"]["rows"]} <= set(focus), "the blotter lists only names the CIO picked"
    assert normalise(reused.output) == normalise(fresh.output)


def test_the_risk_history_and_resize_log_start_empty_on_a_reused_thread(decisions):
    """Both are append-only (operator.add): without a reset the second run's pass numbers continue the first run's."""
    choose_from_hint()
    lazy(0.99)
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    first = g.invoke(FIRST_HINT, cfg)
    assert first["risk_history"], "the first run must have written a history for this test to mean anything"
    again = g.invoke(FIRST_HINT, cfg)
    assert [h["pass"] for h in again["risk_history"]] == [h["pass"] for h in first["risk_history"]]
    assert len(again["resize_log"]) == len(first["resize_log"])

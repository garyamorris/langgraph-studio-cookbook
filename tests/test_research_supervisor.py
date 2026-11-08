from __future__ import annotations

import math
import re

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import set_structured_handler
from deepdive.common.testing import check_example, normalise, run_twice
from deepdive.examples import research_supervisor as ex


# --------------------------------------------------------------------------- scripting helpers
class Script:
    """Registers valid scripted answers for every structured call and records the prompts each one received."""

    def __init__(self, ticker="ACMR", decisions=("market_team", "news_team", "writer", "FINISH"), draft=None):
        self.ticker = ticker
        self.prompts: dict[str, list[str]] = {}
        self.heads = ex.recent_headlines(ticker)
        seq = [ex.Decision(next=n, instruction=f"focus on {n}") for n in decisions]
        self._register(ex.Decision, seq)
        self._register(ex.RiskNotes, [ex.RiskNotes(notes=[
            ex.RiskNote(text="Volatility is elevated and the stock sits below its recent high.", basis=["M5", "M6"]),
            ex.RiskNote(text="Invented basis.", basis=["M99"]),  # must be dropped
        ])])
        self._register(ex.SentimentBatch, [ex.SentimentBatch(items=[
            ex.Labelled(headline_id=h["headline_id"], sentiment=("positive", "negative", "neutral", "mixed")[i % 4])
            for i, h in enumerate(self.heads)
        ] + [ex.Labelled(headline_id="H-9999", sentiment="positive")])])  # unknown id: ignored
        self._register(ex.CatalystList, [ex.CatalystList(catalysts=[
            ex.Catalyst(text="A reported event.", headline_ids=[self.heads[0]["headline_id"]]),
            ex.Catalyst(text="Made up.", headline_ids=["H-9999"]),  # must be dropped
        ])])
        self._register(ex.BriefDraft, [draft or self.good_draft()])

    @staticmethod
    def good_draft():
        return ex.BriefDraft(
            headline="Price momentum is mixed with elevated volatility.",
            headline_fact_ids=["M2", "M5"],
            bullets=[
                ex.BriefBullet(section="price_and_risk", text="The stock returned +0.0% over five days.", fact_ids=["M2"]),
                ex.BriefBullet(section="price_and_risk", text="Volatility is elevated and the stock sits below its recent high.", fact_ids=["R1"]),
                ex.BriefBullet(section="news_and_sentiment", text="Headlines are a mix of tones.", fact_ids=["N1", "N2"]),
                ex.BriefBullet(section="catalysts", text="A reported event.", fact_ids=["C1"]),
            ],
        )

    def _register(self, schema, values):
        i = {"n": 0}
        name = schema.__name__
        self.prompts[name] = []

        def handler(text):
            self.prompts[name].append(text)
            v = values[min(i["n"], len(values) - 1)]
            i["n"] += 1
            return v

        set_structured_handler(schema, handler)

    def calls(self, schema):
        return len(self.prompts[schema.__name__])


def forbid_models():
    def boom(_text):
        raise AssertionError("no model call is allowed on this path")

    for schema in (ex.Decision, ex.RiskNotes, ex.SentimentBatch, ex.CatalystList, ex.BriefDraft):
        set_structured_handler(schema, boom)


def run(inp, **kw):
    return ex.build_graph(**kw).invoke(inp)


def executed_nodes(inp):
    """Names of every node that ran, including inside team subgraphs, with their namespace path."""
    out = []
    for ns, upd in ex.build_graph().stream(inp, stream_mode="updates", subgraphs=True):
        for node in upd:
            out.append((tuple(n.split(":")[0] for n in ns), node))
    return out


# --------------------------------------------------------------------------- tests
def test_contract():
    check_example(ex)


@pytest.mark.parametrize("ticker", ["ACMR", "BORE", "CBLT", "DLPH", "EMBR", "FTHM", "GRNT", "HELO"])
def test_price_analytics_match_pandas(ticker):
    df = read_csv("prices.csv")
    s = df[df["ticker"] == ticker].sort_values("date")["close"].reset_index(drop=True)
    m = ex.price_metrics(ticker)
    approx = lambda x: pytest.approx(float(x), rel=1e-9)  # noqa: E731
    assert m["last_close"] == approx(s.iloc[-1])
    assert m["ret_5d"] == approx(s.pct_change(5).iloc[-1])
    assert m["ret_20d"] == approx(s.pct_change(20).iloc[-1])
    assert m["ret_60d"] == approx(s.pct_change(60).iloc[-1])
    assert m["vol_20d"] == approx(s.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252))
    assert m["high_60d"] == approx(s.rolling(60).max().iloc[-1])
    assert m["drawdown_60d"] == approx(s.iloc[-1] / s.rolling(60).max().iloc[-1] - 1) and m["drawdown_60d"] <= 0
    assert m["sma_50"] == approx(s.rolling(50).mean().iloc[-1])
    assert m["vs_sma_50"] == approx(s.iloc[-1] / s.rolling(50).mean().iloc[-1] - 1)
    assert m["as_of"] == df[df["ticker"] == ticker]["date"].max()


def test_metric_facts_in_the_brief_show_the_pandas_numbers():
    s = Script()
    out = run({"ticker": "ACMR"})
    df = read_csv("prices.csv")
    close = df[df["ticker"] == "ACMR"].sort_values("date")["close"].reset_index(drop=True)
    facts = {f["id"]: f["text"] for f in out["facts"]}
    assert facts["M2"] == f"5-day return {close.pct_change(5).iloc[-1] * 100:+.1f}%."
    assert facts["M1"].startswith(f"Last close {close.iloc[-1]:.2f}")
    assert s.calls(ex.RiskNotes) == 1


def test_scripted_supervisor_runs_market_news_writer_then_finishes_and_merges_team_outputs():
    s = Script()
    out = run({"ticker": "acmr", "question": "Brief me."})
    assert out["route"] == ["market_team", "news_team", "writer", "FINISH"]
    assert [t["why"] for t in out["trail"]] == ["model", "model", "model", "rule: brief already written"]
    assert s.calls(ex.Decision) == 3, "once the writer has run there is nothing left to decide, so no model call"
    ids = [f["id"] for f in out["facts"]]
    assert ids[:8] == [f"M{i}" for i in range(1, 9)] and "R1" in ids and "R2" not in ids, "invalid risk note dropped"
    assert {"N1", "N2", "N3", "C1"} <= set(ids) and "C2" not in ids, "invalid catalyst dropped"
    assert out["ticker"] == "ACMR" and out["company"] == "Acme Robotics" and out["mode"] == "offline"
    # the supervisor's per-worker instruction reached the team's model call, and only as a hint
    assert "focus on market_team" in s.prompts["RiskNotes"][0]
    assert "focus on news_team" in s.prompts["CatalystList"][0]
    assert out["audit"]["bullets_kept"] == 4 and not out["audit"]["fact_sheet_fallback"]


def test_every_claim_in_the_brief_is_traceable_to_a_fact():
    Script()
    out = run({"ticker": "ACMR"})
    brief, valid = out["brief"], {f["id"] for f in out["facts"]}
    assert "Not investment advice" in brief and out["disclaimer"] == ex.DISCLAIMER
    claim_lines = [ln for ln in brief.splitlines() if ln.startswith("- ") or ln.startswith("**")]
    assert len(claim_lines) == 5  # headline + 4 bullets
    for ln in claim_lines:
        cited = ln[ln.rindex("[") + 1 : ln.rindex("]")].split(", ")
        assert cited and set(cited) <= valid, ln
    # notes and catalysts carry the ids they are based on
    by_id = {f["id"]: f for f in out["facts"]}
    assert by_id["R1"]["basis"] == ["M5", "M6"] and by_id["C1"]["basis"] == [Script().heads[0]["headline_id"]]


def test_hop_budget_forces_the_writer_and_skips_the_remaining_team():
    s = Script(decisions=("market_team",))  # the supervisor would keep asking for market_team
    out = run({"ticker": "HELO", "max_hops": 1})
    assert out["route"] == ["market_team", "writer", "FINISH"]
    assert out["trail"][1]["why"].startswith("rule: hop budget spent")
    assert s.calls(ex.Decision) == 1, "after the budget is spent no further supervisor model call is made"
    assert s.calls(ex.SentimentBatch) == 0 and s.calls(ex.CatalystList) == 0, "news_team never ran"
    assert "Not consulted for this brief: news_team." in out["brief"]
    assert not any(f["id"].startswith("N") for f in out["facts"])


def test_budget_bounds_are_clamped():
    cmd = ex.intake({"ticker": "ACMR", "max_hops": 100})
    assert cmd.update["max_hops"] == ex.MAX_HOPS_LIMIT
    assert ex.intake({"ticker": "ACMR", "max_hops": 0}).update["max_hops"] == ex.DEFAULT_MAX_HOPS  # 0 means "unset"
    assert ex.intake({"ticker": "ACMR", "max_hops": -3}).update["max_hops"] == 1
    assert ex.intake({"ticker": "ACMR", "max_hops": "x"}).update["max_hops"] == ex.DEFAULT_MAX_HOPS


def test_a_worker_never_runs_twice_even_if_the_supervisor_keeps_asking():
    Script(decisions=("market_team",))  # asks for market_team every single time
    ran = executed_nodes({"ticker": "ACMR"})
    assert sum(1 for path, n in ran if n == "compute_metrics") == 1
    assert sum(1 for _, n in ran if n == "market_team") == 1 and sum(1 for _, n in ran if n == "news_team") == 1
    s = Script(decisions=("market_team",))  # fresh counters for a second, separate run
    out = run({"ticker": "ACMR"})
    assert out["route"] == ["market_team", "news_team", "writer", "FINISH"]
    assert "never twice" in out["trail"][1]["why"] and out["trail"][1]["asked"] == "market_team"
    assert s.calls(ex.RiskNotes) == 1 and s.calls(ex.SentimentBatch) == 1


def test_finish_before_the_brief_exists_is_turned_into_a_writer_call():
    Script(decisions=("market_team", "FINISH"))
    out = run({"ticker": "ACMR"})
    assert out["route"] == ["market_team", "writer", "FINISH"]
    assert out["trail"][1]["asked"] == "FINISH" and out["trail"][1]["routed"] == "writer"
    assert out["brief"].strip(), "a run can never end without a brief"


def test_writer_cannot_go_first_and_a_team_can_be_skipped_when_the_question_allows_it():
    Script(decisions=("writer",))
    out = run({"ticker": "ACMR"})
    assert out["route"][0] == "market_team" and "needs at least one team report" in out["trail"][0]["why"]
    # news-only question: supervisor sends news_team then writer; market_team never runs
    Script(decisions=("news_team", "writer"))
    ran = executed_nodes({"ticker": "ACMR", "question": "What does the news say?"})
    assert "compute_metrics" not in [n for _, n in ran] and "fetch_headlines" in [n for _, n in ran]


@pytest.mark.parametrize(
    "asked,done,hops,max_hops,expected",
    [
        ("market_team", [], 0, 5, ("market_team", "model")),
        ("news_team", ["market_team"], 1, 5, ("news_team", "model")),
        ("writer", ["market_team"], 1, 5, ("writer", "model")),
        ("writer", [], 0, 5, ("market_team", "rule: writer needs at least one team report")),
        ("market_team", ["market_team"], 1, 5, ("news_team", "rule: market_team already ran, never twice")),
        ("news_team", ["market_team", "news_team"], 2, 5, ("writer", "rule: news_team already ran, never twice")),
        ("FINISH", ["market_team"], 1, 5, ("writer", "rule: cannot finish before the brief is written")),
        ("market_team", [], 3, 3, ("writer", "rule: hop budget spent, writer runs anyway")),
        ("", ["market_team", "news_team", "writer"], 3, 5, ("FINISH", "rule: brief already written")),
        ("news_team", ["market_team", "news_team", "writer"], 3, 5, ("FINISH", "rule: brief already written")),
    ],
)
def test_routing_rules(asked, done, hops, max_hops, expected):
    assert ex.route_decision(asked, done, hops, max_hops) == expected


def test_unknown_ticker_is_a_clean_error_and_no_model_runs():
    forbid_models()
    out = run({"ticker": "ZZZZ"})
    assert "Unknown ticker 'ZZZZ'" in out["error"] and "ACMR" in out["error"]
    assert out["brief"] == "" and out["route"] == [] and out["facts"] == []
    assert out["disclaimer"] == ex.DISCLAIMER
    assert run({"ticker": ""})["error"].startswith("Unknown ticker ''")


def test_writer_drops_untraceable_bullets_and_flags_unverified_numbers():
    draft = ex.BriefDraft(
        headline="Headline with a made-up source.",
        headline_fact_ids=["M99"],
        bullets=[
            ex.BriefBullet(section="price_and_risk", text="Kept: five-day return is +4.0%.", fact_ids=["M2"]),
            ex.BriefBullet(section="price_and_risk", text="Dropped: unknown id.", fact_ids=["M2", "X7"]),
            ex.BriefBullet(section="catalysts", text="Dropped: no source at all.", fact_ids=[]),
            ex.BriefBullet(section="news_and_sentiment", text="", fact_ids=["N1"]),
        ],
    )
    Script(draft=draft)
    out = run({"ticker": "ACMR"})
    assert out["audit"]["bullets_kept"] == 1
    reasons = {d["text"]: d["reason"] for d in out["audit"]["dropped"]}
    assert reasons["Dropped: unknown id."].startswith("unknown fact id: X7")
    assert reasons["Dropped: no source at all."] == "no fact ids" and "" in reasons
    assert "Dropped" not in out["brief"] and "made-up source" not in out["brief"], "headline with an unknown id is omitted"
    flagged = out["audit"]["unverified_numbers"]
    assert flagged and flagged[0]["numbers"] and "4.0" in flagged[0]["numbers"], "'+4.0%' is not in fact M2"
    assert "Kept: five-day return" in out["brief"], "unverified numbers are reported, not silently deleted"


def test_nothing_traceable_falls_back_to_the_fact_sheet():
    draft = ex.BriefDraft(headline="x", headline_fact_ids=["Q1"],
                          bullets=[ex.BriefBullet(section="catalysts", text="Unsourced claim.", fact_ids=["Q1"])])
    Script(draft=draft)
    out = run({"ticker": "ACMR"})
    assert out["audit"]["fact_sheet_fallback"] and "## Fact sheet" in out["brief"]
    assert "Unsourced claim" not in out["brief"]
    assert all(f"[{f['id']}]" in out["brief"] for f in out["facts"]), "every fact is printed with its id"


def test_team_steps_stream_under_their_own_namespaces():
    Script()
    ran = executed_nodes({"ticker": "ACMR"})
    assert [n for p, n in ran if p == ("market_team",)] == ["compute_metrics", "write_risk_notes"]
    assert [n for p, n in ran if p == ("news_team",)] == [
        "fetch_headlines", "classify_sentiment", "tally_sentiment", "extract_catalysts"]
    top = [n for p, n in ran if p == ()]
    assert top == ["intake", "supervisor", "market_team", "supervisor", "news_team", "supervisor", "writer", "supervisor"]


def test_team_internals_stay_private_to_the_subgraphs():
    Script()
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "priv"}}
    out = g.invoke({"ticker": "ACMR"}, cfg)
    state = g.get_state(cfg).values
    for private in ("metrics", "headlines", "labels", "tally", "catalysts", "series"):
        assert private not in state and private not in out
    assert set(state["market"]) == {"status", "facts"} and set(state["news"]) == {"status", "facts"}


def test_news_classification_ignores_unknown_ids_and_counts_the_rest_in_code():
    s = Script()
    out = run({"ticker": "ACMR"})
    n1 = next(f for f in out["facts"] if f["id"] == "N1")["text"]
    # 12 headlines labelled positive/negative/neutral/mixed in rotation = 3 each; the made-up H-9999 label is ignored
    assert "3 positive, 3 negative, 3 neutral, 3 mixed (0 unlabelled)" in n1
    assert next(f for f in out["facts"] if f["id"] == "N2")["text"].startswith("Net sentiment +0.00")
    assert len(s.heads) == 12 and len({h["headline"] for h in s.heads}) == 12, "duplicates collapsed"


def test_ticker_without_headlines_skips_the_news_models(monkeypatch):
    Script(decisions=("news_team", "writer"))
    monkeypatch.setattr(ex, "recent_headlines", lambda ticker, limit=12: [])
    forbidden = []
    set_structured_handler(ex.SentimentBatch, lambda t: forbidden.append("classify"))
    set_structured_handler(ex.CatalystList, lambda t: forbidden.append("catalysts"))
    out = run({"ticker": "ACMR"})
    assert forbidden == []
    assert [f["text"] for f in out["facts"]] == ["No headlines found for this ticker."]


def test_supervisor_instruction_is_cleaned_before_it_reaches_workers():
    long = "Focus on\n\n the   drawdown " + "x" * 500
    set_structured_handler(ex.Decision, lambda t: ex.Decision(next="market_team", instruction=long))
    out = run({"ticker": "ACMR", "max_hops": 1})
    assert len(out["trail"][0]["instruction"]) == ex.MAX_INSTRUCTION_CHARS
    assert out["trail"][0]["instruction"].startswith("Focus on the drawdown xxx")


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
def route_by_progress(text: str):
    """Scripted supervisor that depends only on the prompt (market, news, writer, FINISH), so every run decides alike."""
    done = re.search(r"Done so far: (.*)", text).group(1)
    nxt = next((w for w in ("market_team", "news_team", "writer") if w not in done), "FINISH")
    return ex.Decision(next=nxt, instruction=f"focus on {nxt}")


@pytest.mark.parametrize(
    "first, second",
    [
        ({"ticker": "ACMR", "question": "Give me a one-page brief."}, {"ticker": "CBLT", "question": "Is the weakness in the news?"}),
        ({"ticker": "ACMR", "question": "Give me a one-page brief."}, {"ticker": "ZZZZ"}),  # an error run after a good one
        ({"ticker": "ZZZZ"}, {"ticker": "CBLT", "question": "Is the weakness in the news?"}),  # a good run after an error
    ],
    ids=["new-ticker", "unknown-ticker-after-brief", "brief-after-unknown-ticker"],
)
def test_a_second_run_on_the_same_thread_is_the_same_as_on_a_fresh_one(first, second):
    set_structured_handler(ex.Decision, route_by_progress)
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), first, second)
    assert normalise(reused.output) == normalise(fresh.output)


def test_a_new_ticker_on_a_used_thread_gets_its_own_brief_route_and_audit_trail():
    set_structured_handler(ex.Decision, route_by_progress)
    graph = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    graph.invoke({"ticker": "ACMR"}, cfg)
    out = graph.invoke({"ticker": "CBLT", "question": "Is the weakness in the news?"}, cfg)
    assert out["company"] == "Cobalt Semiconductor" and out["brief"].startswith("# Cobalt Semiconductor (CBLT)")
    assert out["route"] == ["market_team", "news_team", "writer", "FINISH"]
    assert [t["hop"] for t in out["trail"]] == [1, 2, 3, 4]

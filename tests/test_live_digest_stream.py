from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import live_digest_stream as ex

DIGEST = "Headline sentence.\n\nWhat customers like\nSimple and reliable.\n\nRisks to escalate\nR-0315 - burning plug."


def events(inp=None, graph=None, config=None):
    return list(ex.iter_ui_events(graph or ex.build_graph(), inp or ex.EXAMPLE_INPUT, config))


def test_contract():
    check_example(ex)


def test_progress_is_monotonic_and_ends_at_100():
    evs = events()
    pcts = [e["pct"] for e in evs if e["kind"] == "progress"]
    assert len(pcts) >= 8
    assert pcts == sorted(pcts) and len(set(pcts)) == len(pcts), f"progress must only move forward: {pcts}"
    assert pcts[0] < 20 and pcts[-1] == 100
    assert evs[-1]["kind"] == "stage_done" and evs[-1]["node"] == "finish", "finish is the last thing that happens"


def test_tokens_come_only_from_write_digest_and_equal_the_final_digest():
    set_chat_handler(lambda messages, tools: AIMessage(DIGEST))
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    evs = events(graph=g, config=cfg)
    tokens = [e for e in evs if e["kind"] == "token"]
    assert len(tokens) > 5, "the digest must arrive in many chunks, not one"
    assert {t["node"] for t in tokens} == {"write_digest"} and all(t["path"] == () for t in tokens)
    final = g.get_state(cfg).values["digest"]
    assert final == DIGEST
    assert "".join(t["text"] for t in tokens) == final


def test_internal_model_tokens_do_not_leak_into_the_ui_stream():
    """normalise_stream is pure, so feed it a hand-made stream containing every kind of leak."""
    md = lambda node: {"langgraph_node": node}  # noqa: E731
    raw = [
        (("analysis:abc",), "messages", (AIMessageChunk(content='{"themes": ['), md("theme_finder"))),  # classifier JSON
        ((), "messages", (AIMessageChunk(content="Hello "), md("write_digest"))),
        (("analysis:abc",), "messages", (AIMessageChunk(content="nested"), md("write_digest"))),  # same name, wrong level
        ((), "messages", (AIMessageChunk(content=""), md("write_digest"))),  # the empty 'last chunk' marker
        ((), "messages", (AIMessageChunk(content="world"), md("write_digest"))),
        ((), "updates", {"__interrupt__": ()}),
    ]
    out = list(ex.normalise_stream(raw))
    assert [e["text"] for e in out] == ["Hello ", "world"]


def test_real_run_has_internal_calls_that_would_leak_if_unfiltered():
    """The unfiltered messages stream really does contain non-write_digest tokens (a plain model call in a node)."""
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    from deepdive.common.models import get_llm

    class S(TypedDict, total=False):
        out: str

    def internal(state):
        get_llm().invoke([HumanMessage("classify this")])
        return {}

    def write_digest(state):
        return {"out": str(get_llm().invoke([HumanMessage("write")]).text)}

    g = StateGraph(S)
    g.add_node("internal", internal)
    g.add_node("write_digest", write_digest)
    g.add_edge(START, "internal")
    g.add_edge("internal", "write_digest")
    g.add_edge("write_digest", END)
    cg = g.compile()
    raw = list(cg.stream({}, stream_mode=["updates", "messages"], subgraphs=True))
    nodes = {md["langgraph_node"] for _, mode, p in raw if mode == "messages" for md in [p[1]]}
    assert nodes == {"internal", "write_digest"}
    ui = list(ex.normalise_stream(raw))
    assert {e["node"] for e in ui if e["kind"] == "token"} == {"write_digest"}


def test_subgraph_events_carry_their_namespace_and_parent_events_do_not():
    evs = events()
    inner = [e for e in evs if e["kind"] in ("progress", "partial", "stage_done") and e["path"]]
    assert {e["path"] for e in inner} == {("analysis",)}
    assert all(e["ns"] and e["ns"][0].startswith("analysis:") for e in inner)
    inner_nodes = [e["node"] for e in inner if e["kind"] == "stage_done"]
    assert inner_nodes == ["ratings", "theme_finder", "risk_scan"], "the three analysis steps run in order inside the subgraph"
    outer = [e for e in evs if e["kind"] == "stage_done" and not e["path"]]
    assert [e["node"] for e in outer] == ["load", "analysis", "write_digest", "finish"]


def test_partial_results_arrive_before_the_first_token():
    evs = events()
    first_token = next(i for i, e in enumerate(evs) if e["kind"] == "token")
    partial_idx = {e["name"]: i for i, e in enumerate(evs) if e["kind"] == "partial"}
    assert set(partial_idx) == {"rating_histogram", "themes", "risk_flags"}
    assert max(partial_idx.values()) < first_token
    assert partial_idx["rating_histogram"] < partial_idx["themes"] < partial_idx["risk_flags"]


def test_rating_histogram_matches_pandas_for_the_newest_reviews():
    evs = events({"product": "Lumen Cam Pro", "max_reviews": 25})
    hist = next(e for e in evs if e["kind"] == "partial" and e["name"] == "rating_histogram")["data"]
    df = read_csv("product_reviews.csv")
    newest = df[df["product"] == "Lumen Cam Pro"].sort_values(["date", "review_id"], ascending=False).head(25)
    expected = newest["rating"].value_counts().reindex(range(1, 6), fill_value=0)
    assert [h["count"] for h in hist["histogram"]] == expected.tolist()
    assert hist["n"] == 25 and hist["mean"] == round(newest["rating"].mean(), 2)


def test_final_output_is_clean_and_complete():
    out = ex.graph.invoke(ex.EXAMPLE_INPUT)
    assert set(out) == {"product", "n_reviews", "rating_stats", "themes", "risks", "digest", "digest_cites_all_risks", "mode"}
    assert out["n_reviews"] == 40 and out["mode"] == "offline"
    assert "reviews" not in out, "the raw review rows are internal state"


def test_risk_scan_flags_exactly_the_dangerous_reviews_in_the_dataset():
    df = read_csv("product_reviews.csv")
    reviews = [{"review_id": r.review_id, "rating": r.rating, "title": r.title, "text": r.text, "date": r.date} for r in df.itertuples()]
    flagged = {f["review_id"]: f["categories"] for f in ex.scan_risks(reviews)}
    assert set(flagged) == {"R-0167", "R-0262", "R-0315", "R-0316", "R-0317", "R-0318", "R-0319", "R-0320"}
    assert flagged["R-0315"] == ["fire_or_injury"] and "privacy" in flagged["R-0317"] and "lockout" in flagged["R-0262"]


def test_digest_must_cite_flagged_reviews_is_checked_in_code():
    set_chat_handler(lambda m, t: AIMessage("Everything looks fine."))
    out = ex.graph.invoke({"product": "Lumen Plug 4-pack", "max_reviews": 40})
    assert out["risks"] and out["digest_cites_all_risks"] is False
    set_chat_handler(lambda m, t: AIMessage("Risk: R-0315 burning plug."))
    assert ex.graph.invoke({"product": "Lumen Plug 4-pack", "max_reviews": 40})["digest_cites_all_risks"] is True


def _theme_input():
    return {"product": "Lumen Plug 4-pack", "max_reviews": 40}


def test_model_themes_are_verified_and_recounted_by_code():
    rows = read_csv("product_reviews.csv")
    newest = rows[rows["product"] == "Lumen Plug 4-pack"].sort_values(["date", "review_id"], ascending=False).head(40)
    crash_ids = newest[newest.text.str.contains("App crashes", case=False)].review_id.tolist()
    report = ex.ThemeReport(
        themes=[
            ex.Theme(name="App crashes", summary="Renaming a plug crashes the app.", review_ids=crash_ids + ["R-9999", "NOT-A-REVIEW"]),
            ex.Theme(name="Imaginary", summary="Invented.", review_ids=["R-9998", "R-9997"]),  # no valid ids: dropped
            ex.Theme(name="One-off", summary="Single review.", review_ids=[crash_ids[0]]),  # below minimum support: dropped
        ]
    )
    set_structured_handler(ex.ThemeReport, scripted([report]))
    out = ex.graph.invoke(_theme_input())
    assert [t["name"] for t in out["themes"]] == ["App crashes"]
    t = out["themes"][0]
    assert t["source"] == "model" and t["n"] == len(crash_ids), "the count comes from valid ids, not from the model"
    assert t["sentiment"] == "complaint" and t["avg_rating"] <= 3.0


def test_themes_fall_back_to_keywords_when_the_model_output_is_unusable():
    set_structured_handler(ex.ThemeReport, scripted([ex.ThemeReport(themes=[ex.Theme(name="x", summary="y", review_ids=["R-9999"])])]))
    out = ex.graph.invoke(_theme_input())
    assert out["themes"] and {t["source"] for t in out["themes"]} == {"keywords"}


def test_unknown_product_is_a_clean_error_event_not_a_hang():
    evs = events({"product": "Lumen Toaster"})
    assert evs[-1]["kind"] == "error" and evs[-1]["error_type"] == "ValueError"
    assert "Lumen Plug 4-pack" in evs[-1]["message"], "the error lists the valid products"
    assert not any(e["kind"] == "token" for e in evs)
    assert [e["pct"] for e in evs if e["kind"] == "progress"] == [5]


def test_max_reviews_is_clamped_and_product_match_is_case_insensitive():
    out = ex.graph.invoke({"product": "lumen plug 4-pack", "max_reviews": 1})
    assert out["product"] == "Lumen Plug 4-pack" and out["n_reviews"] == ex.MIN_REVIEWS
    out = ex.graph.invoke({"product": "Lumen Plug 4-pack", "max_reviews": 10_000})
    assert out["n_reviews"] == int((read_csv("product_reviews.csv")["product"] == "Lumen Plug 4-pack").sum())


def test_pace_adds_a_delay_per_stage_but_is_capped(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(ex.time, "sleep", slept.append)
    ex.graph.invoke({**ex.EXAMPLE_INPUT, "pace_s": 99})
    assert len(slept) == 5 and set(slept) == {ex.MAX_PACE_S}  # load, three analysis steps, write_digest
    slept.clear()
    ex.graph.invoke(ex.EXAMPLE_INPUT)
    assert slept == []


def test_cli_prints_a_live_view(capsys):
    set_chat_handler(lambda m, t: AIMessage(DIGEST))
    assert ex.main(["Lumen Plug 4-pack"]) == 0
    text = capsys.readouterr().out
    assert "[100%]" in text and "rating_histogram" in text and "Headline sentence." in text
    assert ex.main(["Nope"]) == 1

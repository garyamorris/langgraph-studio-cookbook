"""Tests for the shared layer: datasets, retrieval, offline models, Decisions wrapper."""

from __future__ import annotations

from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deepdive.common import decisions as dec
from deepdive.common.data import db_connect, db_schema, kb_docs, read_csv, read_jsonl
from deepdive.common.fakes import FakeChatModel, scripted, set_chat_handler, set_structured_handler
from deepdive.common.models import get_llm
from deepdive.common.retrieval import get_retriever
from deepdive.common.testing import lint_openai_strict
from deepdive.common.tools import calculator, safe_eval


# ------------------------------------------------------------------ data
def test_datasets_load_and_are_consistent():
    assert len(read_csv("support_tickets.csv")) == 240
    assert {"order_id", "revenue", "product"} <= set(read_csv("sales.csv").columns)
    refunds = read_jsonl("refund_requests.jsonl")
    con = db_connect()
    ids = {r[0] for r in con.execute("SELECT order_id FROM orders")}
    assert all(r["order_id"] in ids for r in refunds)
    assert "CREATE TABLE orders" in db_schema()


def test_read_csv_returns_copies():
    a = read_csv("sales.csv")
    a.loc[:, "revenue"] = 0
    assert read_csv("sales.csv").revenue.sum() > 0


def test_db_is_read_only():
    con = db_connect()
    with pytest.raises(Exception):
        con.execute("DELETE FROM orders")


def test_kb_corpora_present():
    assert len(kb_docs("langgraph")) >= 12 and len(kb_docs("support")) >= 8


# ------------------------------------------------------------------ retrieval
def test_bm25_finds_the_right_article():
    r = get_retriever("support")
    assert r.search("the plug smells like burning and is hot", 1)[0].doc.id == "safety-incidents"
    # Lexical retrieval needs lexical overlap (a lone rare word like "card" can outrank the topical word -
    # that weakness is what the corrective-RAG example's grader exists to catch).
    assert r.search("when is my refund issued to the original payment method", 1)[0].doc.id == "returns-and-refunds"
    assert get_retriever("langgraph").search("how do I fan out in parallel with Send", 1)[0].doc.id == "send-and-map-reduce"


def test_customer_facing_retriever_never_serves_internal_articles():
    public = get_retriever("support")
    assert all(d.audience == "public" for d in public.docs)
    for q in ("refund approval matrix manager approval thresholds", "how long do refunds take to my card"):
        assert all(h.doc.id != "refund-approval-matrix" for h in public.search(q, 5))
    # internal tooling can opt in
    assert get_retriever("support", "all").search("manager approval required refund amount", 1)[0].doc.id == "refund-approval-matrix"


# ------------------------------------------------------------------ fake chat model
class Verdict(BaseModel):
    verdict: Literal["approve", "reject", "revise"]
    approved: bool
    needs_revision: bool
    score: float = Field(ge=0, le=10)
    notes: list[str]


def test_get_llm_is_offline_in_tests():
    assert isinstance(get_llm(), FakeChatModel)


def test_structured_output_is_valid_deterministic_and_loop_friendly():
    llm = get_llm().with_structured_output(Verdict)
    a, b = llm.invoke("hello"), llm.invoke("hello")
    assert a == b and isinstance(a, Verdict)
    assert a.approved is True and a.needs_revision is False  # defaults let loops terminate
    assert 0 <= a.score <= 10 and a.notes


def test_structured_handler_scripts_a_sequence():
    set_structured_handler(Verdict, scripted([
        Verdict(verdict="revise", approved=False, needs_revision=True, score=3, notes=["x"]),
        {"verdict": "approve", "approved": True, "needs_revision": False, "score": 9, "notes": ["ok"]},
    ]))
    llm = get_llm().with_structured_output(Verdict)
    assert llm.invoke("a").approved is False
    assert llm.invoke("b").approved is True  # dict coerced to the model
    assert llm.invoke("c").approved is True  # last value repeats


@tool
def lookup_price(product: str) -> str:
    """Look up the price of a product."""
    return f"{product}: $10"


@tool
def lookup_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


def test_tool_calling_roundtrip():
    llm = get_llm().bind_tools([lookup_weather, lookup_price])
    first = llm.invoke([HumanMessage("what is the price of the hub?")])
    assert first.tool_calls and first.tool_calls[0]["name"] == "lookup_price"
    final = llm.invoke([HumanMessage("what is the price?"), first, ToolMessage("Hub: $10", tool_call_id=first.tool_calls[0]["id"])])
    assert not final.tool_calls and "Hub: $10" in str(final.content)


def test_fake_model_streams_tokens_and_tool_calls_reassemble():
    llm = get_llm()
    chunks = list(llm.stream([HumanMessage("tell me about the hub please")]))
    assert len(chunks) > 3, "text replies must stream as multiple word chunks"
    assert "".join(str(c.content) for c in chunks) == llm.invoke([HumanMessage("tell me about the hub please")]).content

    bound = llm.bind_tools([lookup_price])
    merged = None
    for c in bound.stream([HumanMessage("what is the price of the hub?")]):
        merged = c if merged is None else merged + c
    assert merged.tool_calls and merged.tool_calls[0]["name"] == "lookup_price"


def test_chat_handler_override():
    set_chat_handler(lambda messages, tools: AIMessage("scripted"))
    assert get_llm().invoke("x").content == "scripted"


# ------------------------------------------------------------------ strict-schema lint
def test_lint_flags_free_form_dicts_only():
    class Ok(BaseModel):
        a: str
        b: list[int]

    class Bad(BaseModel):
        free: dict[str, str]

    assert lint_openai_strict(Ok) == []
    assert any("free-form" in p for p in lint_openai_strict(Bad))


# ------------------------------------------------------------------ semantic store
def test_store_semantic_search_ranks_by_overlap():
    from deepdive.common.embeddings import DIMS, embed, make_store

    assert len(embed(["x"])[0]) == DIMS
    store = make_store()
    ns = ("memories", "u1")
    store.put(ns, "a", {"text": "lives in a three storey house with wifi dead zones upstairs"})
    store.put(ns, "b", {"text": "prefers matter compatible gadgets and dislikes subscriptions"})
    store.put(("memories", "u2"), "c", {"text": "wifi dead zones everywhere"})
    hits = store.search(ns, query="my wifi has dead zones in the house", limit=2)
    assert hits[0].key == "a" and all(h.namespace == ns for h in hits)


# ------------------------------------------------------------------ calculator
def test_calculator_is_safe():
    assert safe_eval("(129 * 3) * 0.9") == pytest.approx(348.3)
    assert calculator.invoke({"expression": "2 ** 3"}) == "8"
    assert calculator.invoke({"expression": "__import__('os').system('echo hi')"}).startswith("error")


# ------------------------------------------------------------------ decisions wrapper
QS = [
    dec.predicate("urgent", "Is it urgent?"),
    dec.choice("team", "Which team?", {"billing": "payments", "tech": "devices", "other": ""}),
    dec.score("anger", "How angry?", ["calm", "annoyed", "furious"]),
]


def test_question_builders_match_the_api_shape():
    assert QS[0] == {"name": "urgent", "type": "predicate", "instructions": "Is it urgent?"}
    assert QS[1]["choices"][0] == {"value": "billing", "description": "payments"}
    assert QS[1]["choices"][2] == {"value": "other"}
    assert [lv["label"] for lv in QS[2]["levels"]] == ["calm", "annoyed", "furious"]


def test_offline_decisions_are_stable_and_well_formed():
    r1, r2 = dec.decide("my router is down", QS), dec.decide("my router is down", QS)
    assert r1.mode == "offline" and r1.model_dump() == r2.model_dump()
    assert 0 <= r1.p("urgent") <= 1
    assert r1.pick("team") in {"billing", "tech", "other"}
    assert sum(r1.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert r1.level("anger") in {"calm", "annoyed", "furious"}
    assert 0 <= r1.score("anger") <= 2
    assert not r1.refused("urgent") and r1.refused("missing")


def test_decision_overrides_pin_answers():
    dec.set_decision_override("urgent", 0.97)
    dec.set_decision_override("team", "billing", confidence=0.8)
    dec.set_decision_override("anger", "furious")
    r = dec.decide("x", QS)
    assert r.p("urgent") == 0.97 and r.pick("team") == "billing" and r.conf("team") == pytest.approx(0.8)
    assert r.level("anger") == "furious"


def test_decide_validates_questions():
    with pytest.raises(ValueError):
        dec.decide("x", [])
    with pytest.raises(ValueError):
        dec.decide("x", [dec.predicate("a", "?"), dec.predicate("a", "?")])


def test_compact_is_json_friendly():
    import json

    json.dumps(dec.decide("x", QS).compact())

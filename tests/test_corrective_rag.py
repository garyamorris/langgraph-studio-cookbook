from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common import decisions
from deepdive.common.data import kb_docs
from deepdive.common.decisions import Answer, DecisionResult, set_decision_override
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.retrieval import get_retriever
from deepdive.common.testing import check_example, lint_openai_strict, normalise, run_twice
from deepdive.examples import corrective_rag as ex

HITL = "human-in-the-loop-interrupts"
PAUSE_Q = "How do I pause a graph and wait for a human to approve?"  # BM25 ranks the right article 4th
SIGN_OFF_Q = "How can I let a person sign off before the workflow continues?"  # no useful lexical overlap at first
CAPITAL_Q = "What is the capital of France?"  # shares no word with any article


def pin(relevant: dict[str, float] | None = None, corpus: str = "langgraph", default: float = 0.05):
    """Pin the Decisions API stand-in for EVERY article of a corpus: `relevant` ids get their value, the rest `default`."""
    for d in kb_docs(corpus):
        set_decision_override(ex.question_name(d.id), default)
    for doc_id, p in (relevant or {}).items():
        set_decision_override(ex.question_name(doc_id), p)


def chat(text: str = f"Use interrupt() to pause and resume with a Command. [{HITL}]"):
    """Pin the generator; returns the list of system prompts it was shown (empty list = it was never called)."""
    prompts: list[str] = []

    def handler(messages, tools):
        prompts.append(str(messages[0].content))
        return AIMessage(text)

    set_chat_handler(handler)
    return prompts


def rewrites(*queries: str):
    seen: list[str] = []
    nxt = scripted([ex.Rewrite(query=q, reason="r") for q in queries])

    def handler(text: str):
        seen.append(text)
        return nxt(text)

    set_structured_handler(ex.Rewrite, handler)
    return seen


@pytest.fixture
def decide_calls(monkeypatch):
    """Record every Decisions API call the graph makes (input text + question names)."""
    calls: list[tuple[str, list[str]]] = []
    real = decisions.decide

    def spy(input, questions, **kw):
        calls.append((str(input), [q["name"] for q in questions]))
        return real(input, questions, **kw)

    monkeypatch.setattr(ex, "decide", spy)
    return calls


def test_contract_and_offline_run():
    res = check_example(ex)
    assert res.output["status"] in {"answered", "answered_uncited", "abstained"}
    assert res.output["mode"] == "offline" and res.output["threshold"] == ex.RELEVANCE_THRESHOLD


def test_every_schema_is_openai_strict_compatible():
    assert lint_openai_strict(ex.Rewrite) == []


def test_the_motivating_failure_exists_in_the_data():
    """BM25 alone ranks three wrong articles above the right one - the reason this graph grades before generating."""
    ranked = [h.doc.id for h in get_retriever("langgraph").search(PAUSE_Q, k=4)]
    assert HITL in ranked and ranked.index(HITL) == 3
    assert ranked[0] != HITL


# --------------------------------------------------------------------------- grading and generation
def test_partial_relevance_generation_sees_only_the_kept_article_and_probabilities_are_surfaced(decide_calls):
    pin({HITL: 0.93, "checkpointers-and-threads": 0.35})  # the second is plausible but below the threshold
    prompts = chat()
    out = ex.graph.invoke({"question": PAUSE_Q})

    assert len(prompts) == 1
    assert f"[{HITL}]" in prompts[0]
    for wrong in ("send-and-map-reduce", "checkpointers-and-threads", "subgraphs"):
        assert f"[{wrong}]" not in prompts[0], f"{wrong} was graded irrelevant and must not reach the generator"

    assert out["status"] == "answered" and out["citations"] == [HITL] and out["dropped_citations"] == []
    assert out["rewrites"] == 0 and out["queries"] == [PAUSE_Q]
    assert [s["id"] for s in out["sources"]] == [HITL] and out["sources"][0]["probability"] == 0.93

    (rnd,) = out["grading"]
    by_id = {a["id"]: a for a in rnd["articles"]}
    assert len(by_id) == ex.TOP_K
    assert by_id[HITL]["kept"] and by_id[HITL]["probability"] == 0.93
    assert by_id["checkpointers-and-threads"]["probability"] == 0.35 and not by_id["checkpointers-and-threads"]["kept"]
    assert rnd["articles"][0]["bm25"] >= rnd["articles"][-1]["bm25"], "the BM25 order is preserved next to the probabilities"

    # ONE batched Decisions API call for the whole round, one predicate per retrieved article, in the same input
    assert len(decide_calls) == 1
    inp, names = decide_calls[0]
    assert names == [ex.question_name(a["id"]) for a in rnd["articles"]]
    assert PAUSE_Q in inp and all(f"[{a['id']}]" in inp for a in rnd["articles"])
    json.dumps(out)


def test_threshold_is_inclusive_and_named():
    assert ex.RELEVANCE_THRESHOLD == 0.6
    ids = [h.doc.id for h in get_retriever("langgraph").search(PAUSE_Q, k=4)]
    pin({ids[0]: 0.6, ids[1]: 0.599, ids[2]: 0.61})
    chat("An answer. [" + ids[0] + "]")
    out = ex.graph.invoke({"question": PAUSE_Q})
    kept = {a["id"] for a in out["grading"][0]["articles"] if a["kept"]}
    assert kept == {ids[0], ids[2]}
    assert [s["id"] for s in out["sources"]] == [ids[2], ids[0]], "sources are ordered by probability, best first"


def test_a_refused_grade_counts_as_not_relevant(monkeypatch):
    ids = [h.doc.id for h in get_retriever("langgraph").search(PAUSE_Q, k=4)]

    def fake_decide(input, questions, **kw):
        answers = {q["name"]: Answer(name=q["name"], type="predicate", probability=0.99) for q in questions}
        answers[ex.question_name(ids[0])] = Answer(name=ex.question_name(ids[0]), type="refusal")
        return DecisionResult(model="m", mode="live", answers=answers)

    monkeypatch.setattr(ex, "decide", fake_decide)
    chat(f"x [{ids[1]}]")
    out = ex.graph.invoke({"question": PAUSE_Q})
    first = out["grading"][0]["articles"][0]
    assert first["id"] == ids[0] and first["refused"] and not first["kept"] and first["probability"] == 0.0
    assert ids[0] not in out["citations"]


# --------------------------------------------------------------------------- rewrite loop and abstain
def test_irrelevant_hits_trigger_a_rewrite_and_the_second_round_can_succeed(decide_calls):
    pin({HITL: 0.9})
    seen = rewrites("interrupt resume human approval")
    prompts = chat()
    out = ex.graph.invoke({"question": SIGN_OFF_Q})

    assert out["rewrites"] == 1 and out["queries"] == [SIGN_OFF_Q, "interrupt resume human approval"]
    assert [r["round"] for r in out["grading"]] == [1, 2]
    assert not any(a["kept"] for a in out["grading"][0]["articles"])
    assert out["grading"][1]["articles"][0]["id"] == HITL and out["grading"][1]["articles"][0]["kept"]
    assert out["status"] == "answered" and [s["id"] for s in out["sources"]] == [HITL]
    assert len(prompts) == 1 and len(decide_calls) == 2, "one batched grading call per round"
    # the rewriter is told what was tried and what failed, and is shown the table of contents
    assert SIGN_OFF_Q in seen[0] and "Human-in-the-loop with interrupt()" in seen[0]
    # grading in round 2 is still against the ORIGINAL question
    assert SIGN_OFF_Q in decide_calls[1][0] and "QUESTION\n" + SIGN_OFF_Q in decide_calls[1][0]


def test_all_irrelevant_rewrites_twice_then_abstains_and_never_generates(decide_calls):
    pin({})  # everything is irrelevant
    seen = rewrites("checkpointer persistence thread", "state schema reducer")
    prompts = chat("SHOULD NEVER BE CALLED")
    out = ex.graph.invoke({"question": SIGN_OFF_Q})

    assert prompts == [], "generation must never be called when nothing is relevant"
    assert out["status"] == "abstained" and out["sources"] == [] and out["citations"] == []
    assert out["rewrites"] == ex.MAX_REWRITES == 2 and len(seen) == 2, "the rewrite cap fires"
    assert len(out["queries"]) == 3 and len(out["grading"]) == 3 and len(decide_calls) == 3
    assert "SHOULD NEVER BE CALLED" not in out["answer"]
    assert "could not find this in the LangGraph handbook" in out["answer"]
    assert "closest articles" in out["answer"]
    assert "p=0.05" in out["answer"], "the closest articles are listed with the grader's probabilities"
    best_titles = {a["title"] for rd in out["grading"] for a in rd["articles"]}
    assert any(t in out["answer"] for t in best_titles)
    assert "0.60 relevance threshold" in out["answer"]


def test_no_lexical_hits_skips_grading_and_still_abstains_honestly(decide_calls):
    prompts = chat("SHOULD NEVER BE CALLED")
    rewrites("capital France geography", "France Paris")
    out = ex.graph.invoke({"question": CAPITAL_Q})
    assert out["status"] == "abstained" and prompts == []
    assert decide_calls == [], "nothing was retrieved, so there is nothing to grade and no API call to pay for"
    assert out["rewrites"] == 2 and all(rd["articles"] == [] for rd in out["grading"])
    assert "No article shares any vocabulary" in out["answer"]


def test_a_repeated_rewrite_ends_the_search_immediately(decide_calls):
    pin({})
    rewrites(SIGN_OFF_Q.upper() + "  ")  # same query up to case and whitespace
    prompts = chat("never")
    out = ex.graph.invoke({"question": SIGN_OFF_Q})
    assert out["status"] == "abstained" and prompts == []
    assert out["rewrites"] == 1 and len(out["queries"]) == 1 and len(decide_calls) == 1, "no pointless second retrieval"


# --------------------------------------------------------------------------- the deterministic citation check
def test_fabricated_and_unprovided_citations_are_stripped_but_code_is_untouched():
    pin({HITL: 0.95})
    chat(
        "Call `interrupt(payload)` inside a node and read `state[key]` afterwards [human-in-the-loop-interrupts]. "
        "You can also use Send [send-and-map-reduce] and a made up page [langgraph-bible]. "
        "See [the docs](http://x.test) and item [0] for details."
    )
    out = ex.graph.invoke({"question": PAUSE_Q})
    assert out["citations"] == [HITL]
    assert out["dropped_citations"] == ["send-and-map-reduce", "langgraph-bible"]
    ans = out["answer"]
    assert f"[{HITL}]" in ans
    assert "send-and-map-reduce" not in ans and "langgraph-bible" not in ans
    assert "`state[key]`" in ans, "bracketed text inside code spans is not a citation"
    assert "[the docs](http://x.test)" in ans and "[0]" in ans, "links and indexes are not citations"
    assert "  " not in ans and " ." not in ans, "stripping leaves tidy text"
    assert out["status"] == "answered"


def test_an_answer_left_with_no_valid_citation_is_flagged():
    pin({HITL: 0.95})
    chat("Just use interrupt. [langgraph-bible]")
    out = ex.graph.invoke({"question": PAUSE_Q})
    assert out["status"] == "answered_uncited" and out["citations"] == []
    assert out["dropped_citations"] == ["langgraph-bible"]
    assert "could not be checked against a source" in out["answer"]
    assert "[langgraph-bible]" not in out["answer"]


def test_verify_text_unit():
    text, valid, dropped = ex.verify_text("A [a-b]. B [c]. Again [a-b].\n```\nx[y]\n```", ["a-b"])
    assert valid == ["a-b"] and dropped == ["c"]
    assert text == "A [a-b]. B. Again [a-b].\n```\nx[y]\n```"


# --------------------------------------------------------------------------- corpus and inputs
def test_support_corpus_never_retrieves_internal_articles():
    pin({"returns-and-refunds": 0.9}, corpus="support")
    prompts = chat("Refunds are issued within 5 business days. [returns-and-refunds]")
    out = ex.graph.invoke({"question": "How long do refunds take?", "corpus": "support"})
    assert out["corpus"] == "support" and out["status"] == "answered"
    assert "Lumen Home support policies" in prompts[0]
    retrieved_ids = {a["id"] for rd in out["grading"] for a in rd["articles"]}
    assert "refund-approval-matrix" not in retrieved_ids, "internal-only articles must never reach a customer-facing answer"
    assert out["citations"] == ["returns-and-refunds"]


def test_bad_inputs_are_rejected_clearly():
    with pytest.raises(ValueError, match="question"):
        ex.graph.invoke({"question": "  "})
    with pytest.raises(ValueError, match="Unknown corpus"):
        ex.graph.invoke({"question": "hi", "corpus": "wiki"})


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
@pytest.fixture
def grade_by_question(monkeypatch):
    """Only the handbook's interrupts article answers PAUSE_Q; nothing answers any other question."""

    def fake_decide(input, questions, **kw):
        asked = re.search(r"QUESTION\n(.*?)\n\nCANDIDATE", str(input), re.S).group(1)
        answers = {}
        for q in questions:
            p = 0.95 if asked == PAUSE_Q and q["name"] == ex.question_name(HITL) else 0.05
            answers[q["name"]] = Answer(name=q["name"], type="predicate", probability=p)
        return DecisionResult(model="m", mode="offline", answers=answers)

    monkeypatch.setattr(ex, "decide", fake_decide)


def nonsense_rewrites():
    """A stateless rewriter (same answer for the same prompt on any thread): every round gets a new query that matches nothing."""

    def handler(text: str):
        tried = re.search(r"Queries already tried: (\[.*\])", text).group(1)
        return ex.Rewrite(query=f"zzqx{tried.count(',') + 1} wvvk", reason="r")

    set_structured_handler(ex.Rewrite, handler)


@pytest.mark.parametrize(
    "first, second",
    [
        ({"question": CAPITAL_Q}, {"question": PAUSE_Q}),  # abstained run (rewrite budget spent), then an answerable one
        ({"question": PAUSE_Q}, {"question": CAPITAL_Q}),  # answered run, then one that must abstain
        ({"question": PAUSE_Q}, {"question": "How long do refunds take?", "corpus": "support"}),
    ],
    ids=["abstain-then-answer", "answer-then-abstain", "other-corpus"],
)
def test_a_second_question_on_the_same_thread_is_the_same_as_on_a_fresh_thread(grade_by_question, first, second):
    nonsense_rewrites()  # the rewrites retrieve nothing, so a CAPITAL_Q run spends its whole budget
    chat()
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), first, second)
    assert normalise(reused.output) == normalise(fresh.output)


def test_after_an_abstained_run_the_next_question_searches_with_its_own_words_and_a_full_rewrite_budget(grade_by_question):
    nonsense_rewrites()
    chat()
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    gave_up = g.invoke({"question": CAPITAL_Q}, cfg)
    assert gave_up["status"] == "abstained" and gave_up["rewrites"] == ex.MAX_REWRITES
    out = g.invoke({"question": PAUSE_Q}, cfg)
    assert out["status"] == "answered" and out["citations"] == [HITL]
    assert out["queries"] == [PAUSE_Q] and out["rewrites"] == 0
    assert [rd["round"] for rd in out["grading"]] == [1] and out["grading"][0]["query"] == PAUSE_Q

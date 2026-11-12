from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepdive.common import decisions
from deepdive.common.data import kb_docs
from deepdive.common.decisions import set_decision_override
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example, lint_openai_strict, normalise, run_example, run_graph, run_twice
from deepdive.examples import adaptive_qa_router as ex
from deepdive.examples import corrective_rag, sql_analyst

HITL = "human-in-the-loop-interrupts"
RETURNS_BY_REGION = (
    "SELECT c.region AS region, COUNT(DISTINCT o.order_id) AS returned_orders FROM orders o "
    "JOIN customers c ON c.customer_id = o.customer_id WHERE o.status = 'returned' "
    "GROUP BY c.region ORDER BY returned_orders DESC, region"
)


# --------------------------------------------------------------------------- scaffolding
def decision(route, kb="langgraph", ask=None):
    return ex.RouteDecision(route=route, knowledge_base=kb, reason=f"because {route}", clarifying_question=ask)


def guess(route, kb="langgraph"):
    return ex.RouteGuess(route=route, knowledge_base=kb, reason=f"best guess {route}")


class Seen:
    """Scripted structured handler that remembers every prompt it was called with."""

    def __init__(self, schema, values):
        self.prompts: list[str] = []
        nxt = scripted(values)

        def handler(text: str):
            self.prompts.append(text)
            return nxt(text)

        set_structured_handler(schema, handler)

    def __len__(self):
        return len(self.prompts)


def new_graph():
    return ex.build_graph(checkpointer=InMemorySaver())


def pin_handbook(relevant: str = HITL, corpus: str = "langgraph"):
    for d in kb_docs(corpus):
        set_decision_override(corrective_rag.question_name(d.id), 0.05)
    set_decision_override(corrective_rag.question_name(relevant), 0.95)


def chat(reply: str):
    prompts: list[str] = []

    def handler(messages, tools):
        prompts.append("\n".join(str(m.content) for m in messages))
        return AIMessage(reply)

    set_chat_handler(handler)
    return prompts


# --------------------------------------------------------------------------- contract and the smoke run
def test_contract_and_offline_run():
    res = check_example(ex)
    assert res.output["route"] in {"handbook", "database", "small_talk"}


def test_every_schema_is_openai_strict_compatible():
    # RouteGuess only runs after two clarifications, so the contract check's offline run never lints it.
    assert lint_openai_strict(ex.RouteDecision) == [] and lint_openai_strict(ex.RouteGuess) == []


def test_default_smoke_run_exercises_the_interrupt():
    """If a prompt edit changes the offline stand-in's routing, re-pick EXAMPLE_INPUT / EXAMPLE_RESUMES (see the module)."""
    res = run_example(ex)
    assert len(res.interrupts) >= 1, "EXAMPLE_INPUT must reach clarify so the smoke test and Studio exercise interrupt()"
    assert res.interrupts[0]["ask"] and res.interrupts[0]["original_question"] == ex.EXAMPLE_INPUT["question"]
    assert res.output["clarifications"][0]["answer"] == ex.EXAMPLE_RESUMES[0]
    assert res.output["route"] != "clarify" and res.output["answer"]


# --------------------------------------------------------------------------- the four routes
def test_handbook_route_runs_corrective_rag_and_surfaces_its_output(monkeypatch):
    Seen(ex.RouteDecision, [decision("handbook", "langgraph")])
    pin_handbook()
    chat(f"Call interrupt() to pause, then resume with a Command. [{HITL}]")
    asked: list[str] = []
    real = decisions.decide
    monkeypatch.setattr(corrective_rag, "decide", lambda inp, qs, **kw: (asked.append(str(inp)), real(inp, qs, **kw))[1])

    q = "How do I pause a graph and wait for a human to approve?"
    out = new_graph().invoke({"question": q}, {"configurable": {"thread_id": "h1"}})

    assert out["route"] == "handbook" and out["reason"] == "because handbook"
    assert f"[{HITL}]" in out["answer"], "the subgraph's verified, cited answer is the router's answer"
    d = out["details"]
    assert d["subgraph"] == "corrective_rag" and d["status"] == "answered" and d["citations"] == [HITL]
    assert d["corpus"] == "langgraph" and d["sources"][0]["id"] == HITL and d["sources"][0]["probability"] == 0.95
    assert q in asked[0], "the grader received the user's question"
    assert out["clarifications"] == [] and out["effective_question"] == q


def test_handbook_route_can_target_the_support_knowledge_base():
    Seen(ex.RouteDecision, [decision("handbook", "support")])
    pin_handbook("returns-and-refunds", "support")
    chat("Refunds take 5 business days. [returns-and-refunds]")
    out = new_graph().invoke({"question": "How long do refunds take?"}, {"configurable": {"thread_id": "h2"}})
    assert out["details"]["corpus"] == "support" and out["details"]["citations"] == ["returns-and-refunds"]


def test_handbook_route_surfaces_an_abstention_instead_of_an_invented_answer():
    Seen(ex.RouteDecision, [decision("handbook")])
    for d in kb_docs("langgraph"):
        set_decision_override(corrective_rag.question_name(d.id), 0.1)
    set_structured_handler(corrective_rag.Rewrite, scripted([corrective_rag.Rewrite(query="state reducer", reason="r"), corrective_rag.Rewrite(query="routing edges", reason="r")]))
    generated = chat("SHOULD NOT BE CALLED")
    out = new_graph().invoke({"question": "How do I pause a graph and wait for a human to approve?"}, {"configurable": {"thread_id": "h3"}})
    assert out["details"]["status"] == "abstained" and out["details"]["rewrites"] == 2
    assert generated == [] and "could not find this" in out["answer"]


def test_database_route_runs_sql_analyst_and_surfaces_rows_and_sql():
    Seen(ex.RouteDecision, [decision("database")])
    sql_prompts = Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql=RETURNS_BY_REGION, rationale="all-time returns by customer region")])
    chat("North America had the most returned orders (17).")
    out = new_graph().invoke({"question": "Which region had the most returned orders?"}, {"configurable": {"thread_id": "d1"}})

    assert out["route"] == "database"
    d = out["details"]
    assert d["subgraph"] == "sql_analyst" and d["status"] == "answered" and d["attempts"] == 1
    assert d["rows"][0] == {"region": "North America", "returned_orders": 17} and d["row_count"] == 4
    assert d["sql"] == RETURNS_BY_REGION and RETURNS_BY_REGION in out["answer"]
    assert "Question: Which region had the most returned orders?" in sql_prompts.prompts[0]
    assert "schema_context" not in out and "run_sql" not in out, "subgraph internals must not leak into the parent state"
    json.dumps(out)


def test_database_failures_surface_the_repair_history():
    Seen(ex.RouteDecision, [decision("database")])
    Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="DROP TABLE orders", rationale="oops")])
    chat("never")
    out = new_graph().invoke({"question": "drop the orders table"}, {"configurable": {"thread_id": "d2"}})
    assert out["details"]["status"] == "failed" and out["details"]["attempts"] == 3
    assert len(out["details"]["failed_attempts"]) == 3 and "could not produce a safe" in out["answer"]


def test_small_talk_answers_directly_and_runs_neither_specialist(monkeypatch):
    Seen(ex.RouteDecision, [decision("small_talk")])
    sql_prompts = Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="SELECT 1", rationale="r")])
    grader_calls: list[int] = []
    monkeypatch.setattr(corrective_rag, "decide", lambda *a, **k: grader_calls.append(1))
    prompts = chat("You are welcome!")
    out = new_graph().invoke({"question": "thanks, that helped!"}, {"configurable": {"thread_id": "s1"}})
    assert out["route"] == "small_talk" and out["answer"] == "You are welcome!" and out["details"] == {"subgraph": None}
    assert len(sql_prompts) == 0 and grader_calls == [], "no specialist may run for small talk"
    assert "Do not state any fact" in prompts[0]


# --------------------------------------------------------------------------- clarification
def test_clarify_interrupts_then_resumes_and_reroutes_with_the_answer_appended():
    cls = Seen(
        ex.RouteDecision,
        [decision("clarify", ask="Last month's what: orders, revenue or tickets?"), decision("database")],
    )
    sql_prompts = Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="SELECT COUNT(*) AS orders FROM orders WHERE order_date >= '2026-10-01' AND order_date < '2026-11-01'", rationale="October orders")])
    chat("October had 79 orders.")

    res = run_graph(new_graph(), {"question": "what about last month?"}, resumes=["How many orders were placed?"])

    assert len(res.interrupts) == 1
    assert res.interrupts[0] == {
        "ask": "Last month's what: orders, revenue or tickets?",
        "original_question": "what about last month?",
        "why": "because clarify",
        "clarification": 1,
        "of": 2,
    }
    assert len(cls) == 2, "classified once before and once after the clarification"
    assert "Clarification 1" not in cls.prompts[0]
    assert "Last month's what" in cls.prompts[1] and "How many orders were placed?" in cls.prompts[1]

    out = res.output
    assert out["route"] == "database" and out["clarifications"] == [{"ask": "Last month's what: orders, revenue or tickets?", "answer": "How many orders were placed?"}]
    assert out["effective_question"] == "what about last month? How many orders were placed?"
    assert "Question: what about last month? How many orders were placed?" in sql_prompts.prompts[0], "the specialist gets the question AND the user's answer"
    assert out["details"]["rows"] == [{"orders": 79}]


def test_paused_run_holds_until_resumed_and_the_resume_value_is_used():
    Seen(ex.RouteDecision, [decision("clarify", ask="Which measure?"), decision("small_talk")])
    chat("ok")
    g = new_graph()
    cfg = {"configurable": {"thread_id": "c1"}}
    paused = g.invoke({"question": "and the other one?"}, cfg)
    assert paused["__interrupt__"][0].value["ask"] == "Which measure?"
    assert not paused.get("answer"), "nothing was answered while waiting"
    done = g.invoke(Command(resume="the cameras"), cfg)
    assert done["clarifications"][0]["answer"] == "the cameras" and done["route"] == "small_talk"


def test_clarification_cap_forces_a_best_guess_with_a_schema_that_cannot_clarify():
    asked = Seen(ex.RouteDecision, [decision("clarify", ask="Can you be more specific?")])  # would clarify forever
    forced = Seen(ex.RouteGuess, [guess("database")])
    Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="SELECT COUNT(*) AS n FROM orders", rationale="r")])
    chat("There are 845 orders.")

    res = run_graph(new_graph(), {"question": "hmm, numbers?"}, resumes=["more", "stuff"])

    assert ex.MAX_CLARIFICATIONS == 2
    assert len(res.interrupts) == 2, "exactly two clarifications are asked"
    assert [i["clarification"] for i in res.interrupts] == [1, 2]
    assert len(asked) == 2 and len(forced) == 1, "the third classification uses RouteGuess, which has no 'clarify'"
    assert "may NOT ask another" in forced.prompts[0] and "2 clarifying question" in forced.prompts[0]
    assert "more" in forced.prompts[0] and "stuff" in forced.prompts[0]
    assert "may NOT ask another" not in asked.prompts[1]
    assert res.output["route"] == "database" and res.output["reason"] == "best guess database"
    assert len(res.output["clarifications"]) == 2
    assert res.output["effective_question"] == "hmm, numbers? more stuff"


def test_route_guess_schema_has_no_clarify_route():
    assert "clarify" not in ex.RouteGuess.model_fields["route"].annotation.__args__
    assert "clarify" in ex.RouteDecision.model_fields["route"].annotation.__args__


def test_clarify_without_a_question_still_asks_something():
    Seen(ex.RouteDecision, [decision("clarify", ask=None), decision("small_talk")])
    chat("ok")
    res = run_graph(new_graph(), {"question": "??"}, resumes=["hello"])
    assert res.interrupts[0]["ask"] == "Could you say a bit more about what you need?"


# --------------------------------------------------------------------------- composition details
def test_subgraph_nodes_appear_under_their_own_namespace_when_streaming():
    Seen(ex.RouteDecision, [decision("database")])
    Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="SELECT COUNT(*) AS n FROM orders", rationale="r")])
    chat("845")
    events = list(new_graph().stream({"question": "How many orders are there?"}, {"configurable": {"thread_id": "st1"}}, stream_mode="updates", subgraphs=True))
    top = [next(iter(u)) for ns, u in events if ns == ()]
    inner = {next(iter(u)) for ns, u in events if ns and ns[0].startswith("database:")}
    assert top == ["start_run", "classify", "database"]
    assert {"inspect_schema", "write_sql", "guard", "execute", "answer"} <= inner


def test_a_blank_form_runs_the_first_ready_made_question_and_an_unknown_case_is_refused():
    first = next(iter(ex.CASES.values()))["question"]
    for n, blank in enumerate(({}, {"question": "  "})):  # Studio's Submit on an empty form: no dead end
        Seen(ex.RouteDecision, [decision("clarify", ask="Last month's what?")])
        chat("ok")
        res = run_graph(new_graph(), blank, config={"configurable": {"thread_id": f"blank{n}"}})
        assert res.interrupts[0]["original_question"] == first
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        new_graph().invoke({"scenario": "not a case"}, {"configurable": {"thread_id": "e1"}})


def test_each_ready_made_label_leads_with_its_question_and_states_the_real_clarification_cap():
    for label, fields in ex.CASES.items():
        assert label.startswith(fields["question"]), f"{label}: lead with the question the case asks"
    assert ex.CASES and ex.MAX_CLARIFICATIONS == 2
    assert f"{ex.MAX_CLARIFICATIONS}-question cap" in next(k for k in ex.CASES if k.startswith("and the other one"))


def test_picking_the_vague_case_and_answering_vaguely_hits_the_cap_and_forces_a_route():
    label = next(k for k in ex.CASES if k.startswith("and the other one"))
    Seen(ex.RouteDecision, [decision("clarify", ask="The other what?")])  # would ask forever
    forced = Seen(ex.RouteGuess, [guess("small_talk")])
    chat("Happy to help: ask me about the handbook or the shop.")
    res = run_graph(new_graph(), {"scenario": label}, resumes=["the other one", "you know"])
    assert len(res.interrupts) == ex.MAX_CLARIFICATIONS and len(forced) == 1
    assert res.output["route"] == "small_talk" and res.output["effective_question"] == "and the other one? the other one you know"


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
def route_by_question():
    """A stateless scripted front door: "Hello" is small talk, "last month" is asked about once, the rest is the database."""

    def classify(text: str):
        question = re.search(r"Question: (.*)", text).group(1)
        if "Hello" in question:
            return decision("small_talk")
        if "last month" in question and "Clarification 1" not in text:
            return decision("clarify", ask="Last month's what?")
        return decision("database")

    set_structured_handler(ex.RouteDecision, classify)
    set_structured_handler(ex.RouteGuess, lambda text: guess("database"))
    Seen(sql_analyst.SqlDraft, [sql_analyst.SqlDraft(sql="SELECT COUNT(*) AS n FROM orders", rationale="r")])
    chat("scripted reply")


@pytest.mark.parametrize(
    "first, first_resumes, second",
    [
        ("what about last month, then?", ["revenue"], "Hello there"),  # a clarified question, then small talk
        ("Hello there", [], "what about last month, then?"),  # small talk, then a question that needs clarifying
        ("what about last month, then?", ["revenue"], "How many orders are there?"),
    ],
    ids=["clarified-then-hello", "hello-then-clarified", "clarified-then-database"],
)
def test_a_second_question_on_the_same_thread_is_the_same_as_on_a_fresh_thread(first, first_resumes, second):
    route_by_question()
    second_resumes = ["revenue"] if "last month" in second else []
    reused, fresh = run_twice(
        new_graph, {"question": first}, {"question": second}, first_resumes=first_resumes, second_resumes=second_resumes
    )
    assert normalise(reused.output) == normalise(fresh.output)


def test_an_old_clarification_is_not_glued_onto_the_next_question():
    route_by_question()
    g = new_graph()
    cfg = {"configurable": {"thread_id": "t"}}
    first = run_graph(g, {"question": "what about last month, then?"}, resumes=["revenue"], config=cfg)
    assert first.output["effective_question"] == "what about last month, then? revenue"
    second = run_graph(g, {"question": "Hello there"}, config=cfg)
    assert second.output["effective_question"] == "Hello there"
    assert second.output["clarifications"] == [] and second.interrupts == []


def test_a_run_paused_at_clarify_shows_nothing_from_the_previous_questions_answer():
    route_by_question()
    g = new_graph()
    cfg = {"configurable": {"thread_id": "stale"}}
    done = run_graph(g, {"question": "Hello there"}, config=cfg)
    assert done.output["answer"] == "scripted reply" and done.output["effective_question"] == "Hello there"
    paused = run_graph(g, {"question": "what about last month, then?"}, config=cfg)
    assert paused.interrupts, "the second question needs clarifying"
    assert not paused.output.get("answer") and not paused.output.get("details") and not paused.output.get("effective_question")

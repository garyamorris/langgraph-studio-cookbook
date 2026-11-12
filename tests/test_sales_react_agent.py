from __future__ import annotations

import itertools
import json
import re

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import set_chat_handler
from deepdive.common.testing import check_example
from deepdive.examples import sales_react_agent as ex

_ids = itertools.count(1)


def call(name, args=None, *, n=None):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": f"call_{n or next(_ids)}", "type": "tool_call"}])


class Script:
    """Scripted model: the k-th model call returns the k-th reply (the last repeats) and every call is recorded."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[list, list[str]]] = []

    def __call__(self, messages, tools):
        self.calls.append((list(messages), [t["function"]["name"] for t in tools]))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return reply(messages, tools) if callable(reply) else reply


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def tool_messages(out):
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def assert_history_is_valid(messages):
    """Every assistant tool call has its tool result, and no tool result is orphaned."""
    requested = {tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert requested == answered, f"unanswered={requested - answered} orphaned={answered - requested}"


def completed():
    df = read_csv("sales.csv")
    df["d"] = pd.to_datetime(df["order_date"])
    return df[df["status"] != "cancelled"]


def test_contract_and_offline_run():
    out = check_example(ex).output
    assert isinstance(out["messages"][-1], AIMessage) or "messages" in out


# --------------------------------------------------------------------------- the happy path
def test_tool_call_then_tool_result_then_answer_with_exact_numbers():
    args = {"metrics": ["return_rate"], "group_by": "region", "product": "Lumen Cam Pro", "period": "2026Q3"}

    def answer(messages, tools):
        rows = json.loads(messages[-1].content)["rows"]  # the model reads the tool result
        return AIMessage(f"{rows[0]['group']} had the highest return rate at {rows[0]['return_rate']}%.")

    script = Script(call("query_sales", args), answer)
    set_chat_handler(script)
    out = ex.graph.invoke(user("Which region had the highest return rate for Cam Pro last quarter?"))

    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    d = completed()
    q3 = d[(d["product"] == "Lumen Cam Pro") & (d["d"] >= "2026-07-01") & (d["d"] <= "2026-09-30")]
    rates = q3.groupby("region").apply(lambda g: (g["status"] == "returned").mean() * 100, include_groups=False)
    result = json.loads(out["messages"][2].content)
    assert result["rows"][0]["group"] == rates.idxmax()
    assert result["rows"][0]["return_rate"] == pytest.approx(rates.max(), abs=0.05)
    assert out["messages"][-1].content == f"{rates.idxmax()} had the highest return rate at {round(rates.max(), 1)}%."
    assert out["tool_rounds"] == 1 and out["messages"][2].status == "success" and out["mode"] == "offline"


def test_the_model_gets_a_system_prompt_with_resolved_dates_and_all_three_tools():
    script = Script(AIMessage("ok"))
    set_chat_handler(script)
    out = ex.graph.invoke(user("hi"))
    sent, tool_names = script.calls[0]
    assert isinstance(sent[0], SystemMessage) and not any(isinstance(m, SystemMessage) for m in out["messages"]), "the prompt is added per call, never stored"
    assert set(tool_names) == {"describe_dataset", "query_sales", "calculator"}
    text = sent[0].content
    assert "Every number you state must come from a tool result" in text
    assert "'last quarter' = 2026Q3" in text and "today is 2026-10-31" in text and "2026Q2" in text
    assert "Lumen Cam Pro" in text and "calculator" in text


def test_query_sales_tool_returns_numbers_equal_to_direct_pandas():
    d = completed()
    raw = json.loads(ex.query_sales.invoke({"metrics": ["revenue", "units"], "group_by": "product", "period": "2026Q2", "top_n": 3}))
    q2 = d[(d["d"] >= "2026-04-01") & (d["d"] <= "2026-06-30")]
    expected = q2.groupby("product").agg(revenue=("revenue", "sum"), units=("quantity", "sum")).sort_values("revenue", ascending=False).head(3)
    assert [r["group"] for r in raw["rows"]] == list(expected.index)
    for row, (_, e) in zip(raw["rows"], expected.iterrows()):
        assert row["revenue"] == pytest.approx(e["revenue"], abs=0.01) and row["units"] == e["units"]
    total = json.loads(ex.query_sales.invoke({"metrics": ["revenue"], "region": "europe", "status": "all"}))
    eu = read_csv("sales.csv").query("region == 'Europe'")
    assert total["total"]["revenue"] == pytest.approx(eu["revenue"].sum(), abs=0.01) and total["filters"]["region"] == "Europe"
    info = json.loads(ex.describe_dataset.invoke({}))
    assert info["last_full_quarter"]["label"] == "2026Q3" and "Lumen Therm" in info["products"]


def test_calculator_is_available_and_exact():
    script = Script(call("calculator", {"expression": "(129 * 3) * 0.9"}), AIMessage("It is 348.3."))
    set_chat_handler(script)
    out = ex.graph.invoke(user("What is 3 hubs with a 10% discount?"))
    assert tool_messages(out)[0].content == "348.3"


def test_parallel_tool_calls_in_one_reply_are_one_round():
    both = AIMessage(content="", tool_calls=[
        {"name": "describe_dataset", "args": {}, "id": "c1", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "c2", "type": "tool_call"},
    ])
    set_chat_handler(Script(both, AIMessage("done")))
    out = ex.graph.invoke(user("q"))
    assert len(tool_messages(out)) == 2 and out["tool_rounds"] == 1
    assert_history_is_valid(out["messages"])


# --------------------------------------------------------------------------- tool errors are conversation
def test_invalid_arguments_return_an_error_tool_message_and_the_loop_continues():
    bad = call("query_sales", {"metrics": ["revenue"], "product": "Lumen Thermm", "period": "2026Q2"})
    fixed = call("query_sales", {"metrics": ["units"], "product": "Lumen Therm", "period": "2026Q2"})
    script = Script(bad, fixed, AIMessage("Therm sold some units."))
    set_chat_handler(script)
    out = ex.graph.invoke(user("How many Therms did we sell in Q2?"))
    first, second = tool_messages(out)
    assert first.status == "error" and "Did you mean 'Lumen Therm'" in first.content and "call the tool again" in first.content
    assert second.status == "success" and json.loads(second.content)["total"]["units"] > 0
    assert len(script.calls) == 3, "the model was called again after the error and then once more after the fix"
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in script.calls[1][0]), "the model saw the error"
    assert out["tool_rounds"] == 2 and out["messages"][-1].content == "Therm sold some units."


@pytest.mark.parametrize(
    "args,fragment",
    [
        ({"group_by": "colour"}, "Invalid arguments for query_sales"),  # rejected by the argument schema
        ({"metrics": ["profit"]}, "Invalid arguments for query_sales"),
        ({"period": "2026-13-45"}, "not a real calendar date"),  # rejected by the query layer
        ({"region": "Mars"}, "Valid region values"),
        ({"period": "2024Q1"}, "outside the data"),
        ({"top_n": 3}, "needs a group_by"),
    ],
)
def test_each_kind_of_bad_call_is_reported_not_raised(args, fragment):
    set_chat_handler(Script(call("query_sales", args), AIMessage("sorry")))
    out = ex.graph.invoke(user("q"))
    message = tool_messages(out)[0]
    assert message.status == "error" and fragment in message.content
    assert out["messages"][-1].content == "sorry"


def test_unknown_tool_name_is_also_handled():
    set_chat_handler(Script(call("drop_table", {}), AIMessage("fine")))
    out = ex.graph.invoke(user("q"))
    assert out["messages"][-1].content == "fine" and tool_messages(out)[0].status == "error"


# --------------------------------------------------------------------------- the loop is bounded
def test_a_model_that_never_stops_calling_tools_is_stopped_at_the_cap():
    script = Script(lambda messages, tools: call("describe_dataset"))  # stubborn: asks for a tool on every call
    set_chat_handler(script)
    out = ex.graph.invoke({**user("q"), "max_tool_rounds": 3})
    assert len(tool_messages(out)) == 3 and out["tool_rounds"] == 3, "exactly the allowed number of tool rounds ran"
    assert len(script.calls) == 4, "three tool rounds, then one forced final answer"
    final = out["messages"][-1]
    assert isinstance(final, AIMessage) and not final.tool_calls and "used all 3 of my tool calls" in final.content
    assert_history_is_valid(out["messages"])  # a follow-up question on this thread would still be accepted by the API
    last_messages, last_tools = script.calls[-1]
    assert last_tools == [], "at the cap the model is called with no tools bound"
    assert "TOOL LIMIT REACHED" in last_messages[0].content and "all 3 tool-calling rounds" in last_messages[0].content
    assert all("TOOL LIMIT" not in c[0][0].content for c in script.calls[:-1])


def test_at_the_cap_a_well_behaved_model_gives_a_real_partial_answer():
    def at_limit(messages, tools):
        return AIMessage("From the results so far: Europe is ahead; I could not check APAC.") if not tools else call("describe_dataset")

    set_chat_handler(Script(at_limit))
    out = ex.graph.invoke({**user("q"), "max_tool_rounds": 2})
    assert len(tool_messages(out)) == 2
    assert out["messages"][-1].content == "From the results so far: Europe is ahead; I could not check APAC."


def test_the_cap_is_per_user_message_not_per_thread():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    fresh = lambda make: (lambda messages, tools: make())  # noqa: E731 - new message objects per call, as a real model returns
    turn = [fresh(lambda: call("describe_dataset")), fresh(lambda: call("describe_dataset")), fresh(lambda: AIMessage("answer"))]
    set_chat_handler(Script(*turn, *turn))
    first = g.invoke({**user("one"), "max_tool_rounds": 2}, cfg)
    second = g.invoke(user("two"), cfg)
    assert first["messages"][-1].content == "answer" and second["messages"][-1].content == "answer", "neither turn hit the cap"
    assert second["tool_rounds"] == 2, "the counter restarted for the second question (4 tool rounds in total)"


def test_max_tool_rounds_and_messages_are_validated():
    for bad in (0, 11, True):
        with pytest.raises(ValueError, match="max_tool_rounds must be"):
            ex.graph.invoke({**user("q"), "max_tool_rounds": bad})
    # a round limit with no message is real input that cannot run (a blank form, by contrast, runs the first ready-made case)
    with pytest.raises(ValueError, match="Send at least one user message"):
        ex.graph.invoke({"max_tool_rounds": 3})
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        ex.graph.invoke({"scenario": "not a case"})


# --------------------------------------------------------------------------- history is bounded
def test_trim_history_keeps_the_window_start_on_a_user_message_and_never_orphans_tool_results():
    msgs = [
        HumanMessage("q1 " * 20, id="h1"), call("describe_dataset", n=1), ToolMessage("x" * 400, tool_call_id="call_1", id="t1"), AIMessage("a1", id="a1"),
        HumanMessage("q2", id="h2"), AIMessage("a2", id="a2"),
    ]
    msgs[1].id = "ai1"
    removed = {r.id for r in ex.trim_history(msgs, budget=40)}
    assert removed == {"h1", "ai1", "t1", "a1"}, "the whole first exchange goes together"
    assert ex.trim_history(msgs, budget=10_000) == []
    huge = [HumanMessage("old", id="o"), AIMessage("a", id="oa"), HumanMessage("word " * 500, id="new")]
    assert {r.id for r in ex.trim_history(huge, budget=20)} == {"o", "oa"}, "the newest question is kept even if it alone exceeds the budget"


def test_long_conversations_stay_within_the_token_budget(monkeypatch):
    budget = 160
    monkeypatch.setattr(ex, "HISTORY_TOKEN_BUDGET", budget)
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "long-chat"}}
    seen_by_model: list[int] = []

    def model(messages, tools):
        window = [m for m in messages if not isinstance(m, SystemMessage)]
        if not seen_by_model or isinstance(window[-1], HumanMessage):
            seen_by_model.append(count_tokens_approximately(window))
        if isinstance(window[-1], HumanMessage) and "tool" in window[-1].content:
            return call("calculator", {"expression": "1+1"})
        return AIMessage("An answer that is about twenty words long so that every turn adds a noticeable amount of text to the chat history.")

    set_chat_handler(model)
    for i in range(10):
        text = f"Question number {i} about regional revenue, please answer briefly" + (" and use a tool" if i % 3 == 1 else "")
        out = g.invoke(user(text), cfg)
        stored = g.get_state(cfg).values["messages"]
        assert isinstance(stored[0], HumanMessage), "the kept window always starts at a user message"
        assert_history_is_valid(stored)
        assert text in [m.content for m in stored if isinstance(m, HumanMessage)], "the newest question is always kept"
    humans = [m.content for m in stored if isinstance(m, HumanMessage)]
    assert 1 < len(humans) < 10, f"old turns were dropped from state, recent ones kept: {humans}"
    assert "Question number 0" not in " ".join(humans)
    assert max(seen_by_model) <= budget, f"the model never saw more than the budget at the start of a turn: {seen_by_model}"
    assert out["messages"][-1].content.startswith("An answer")


def test_a_short_conversation_is_left_alone():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "short"}}
    set_chat_handler(Script(AIMessage("a1"), AIMessage("a2")))
    g.invoke(user("first"), cfg)
    out = g.invoke(user("second"), cfg)
    assert [m.content for m in out["messages"]] == ["first", "a1", "second", "a2"]


# --------------------------------------------------------------------------- streaming
def test_tokens_stream_from_the_agent_node_only():
    final = "Europe led last quarter with the highest return rate overall."
    set_chat_handler(Script(call("describe_dataset"), AIMessage(final)))
    chunks = []
    other_nodes = set()
    for message, meta in ex.graph.stream(user("which region?"), stream_mode="messages"):
        if meta["langgraph_node"] == "agent" and isinstance(message, AIMessageChunk):
            chunks.append(str(message.content))
        else:
            other_nodes.add(meta["langgraph_node"])
    assert len([c for c in chunks if c]) > 3, "the answer arrives as several token chunks"
    assert "".join(chunks) == final
    assert "tools" in other_nodes, "tool results are streamed too, from a different node, so a UI can filter them out"


# --------------------------------------------------------------------------- the ready-made opening messages
def label_for(start: str) -> str:
    return next(k for k in ex.CASES if k.startswith(start))


def test_a_blank_form_asks_the_first_ready_made_question():
    first = next(iter(ex.CASES.values()))["messages"][0]["content"]
    script = Script(AIMessage("ok"))
    set_chat_handler(script)
    for blank in ({}, {"messages": []}):  # Studio's Submit on an empty form: no dead end
        out = ex.graph.invoke(blank)
        assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"] and out["messages"][0].content == first
    sent = [m.content for m in script.calls[0][0] if isinstance(m, HumanMessage)]
    assert sent == [first], "the model was sent the opening question, and nothing else from the user"


def test_every_case_is_one_user_message_with_a_valid_round_limit_and_its_label_leads_with_it():
    for label, fields in ex.CASES.items():
        (message,) = fields["messages"]
        assert message["role"] == "user" and label.startswith(message["content"]), f"{label}: lead with the question the case asks"
        rounds = fields["max_tool_rounds"]
        assert rounds is None or 1 <= rounds <= ex.MAX_TOOL_ROUNDS_LIMIT
        stated = re.search(r"\((\d+) tool rounds?\)|\((\d+) rounds\)", label)
        assert (int(next(g for g in stated.groups() if g)) if stated else None) == rounds, f"{label}: the label must state the limit the case sets"


def test_picking_a_case_on_a_thread_with_history_adds_to_it_like_any_typed_message():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "history"}}
    set_chat_handler(Script(AIMessage("a1"), AIMessage("a2")))
    g.invoke(user("first question"), cfg)
    out = g.invoke({"scenario": label_for("What did customers say")}, cfg)
    assert [m.content for m in out["messages"]] == ["first question", "a1", "What did customers say about the Therm price?", "a2"]
    assert g.get_state(cfg).values.get("scenario") is None
    # and a blank submit on a thread that already has messages is not given a new opening message
    blank = g.invoke({}, cfg)
    assert [m.content for m in blank["messages"]][:4] == ["first question", "a1", "What did customers say about the Therm price?", "a2"]
    assert "Which region had the highest return rate" not in " ".join(str(m.content) for m in blank["messages"])


def test_a_picked_case_is_trimmed_with_the_history_so_the_newest_question_is_always_kept(monkeypatch):
    monkeypatch.setattr(ex, "HISTORY_TOKEN_BUDGET", 60)
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "trim"}}
    set_chat_handler(Script(AIMessage("An answer long enough to take real room in the history window of this chat.")))
    for i in range(3):
        g.invoke(user(f"Older question number {i}, please answer in some detail"), cfg)
    out = g.invoke({"scenario": label_for("What did customers say")}, cfg)
    stored = g.get_state(cfg).values["messages"]
    assert isinstance(stored[0], HumanMessage), "the kept window starts at a user message"
    assert "What did customers say about the Therm price?" in [m.content for m in stored], "the newest question is always kept"
    assert "Older question number 0, please answer in some detail" not in [m.content for m in stored], "old turns were trimmed"
    assert_history_is_valid(out["messages"])


def test_the_facts_in_the_ready_made_labels_are_true():
    d = completed()
    cam = d[(d["product"] == "Lumen Cam Pro") & (d["d"] >= "2026-07-01") & (d["d"] <= "2026-09-30")]
    rate = cam.groupby("region")["status"].apply(lambda x: (x == "returned").mean() * 100)
    top = rate.idxmax()
    label = label_for("Which region had the highest return rate")
    assert top in label and f"{round(rate[top])}%" in label and f"{(cam['region'] == top).sum()} lines" in label
    eu = d[d["region"] == "Europe"]
    october = eu[(eu["d"] >= "2026-10-01") & (eu["d"] <= "2026-10-31")]["revenue"].sum()
    july = eu[(eu["d"] >= "2026-07-01") & (eu["d"] <= "2026-07-31")]["revenue"].sum()
    assert f"${october:,.2f} vs ${july:,.2f}" in label_for("Compare Europe's October")
    assert not {"review", "comment", "feedback", "satisfaction", "text"} & {c.lower() for c in read_csv("sales.csv").columns}
    assert "Lumen Doorbell" not in set(d["product"]) and "'Lumen Bell'" in label_for("How many Lumen Doorbells")
    assert json.loads(ex.describe_dataset.invoke({}))["last_full_quarter"]["label"] in label_for("What products and regions")


def test_picking_the_doorbell_case_returns_an_error_message_the_model_recovers_from():
    bad = call("query_sales", {"metrics": ["units"], "product": "Lumen Doorbell", "period": "2026Q3"})
    fixed = call("query_sales", {"metrics": ["units"], "product": "Lumen Bell", "period": "2026Q3"})
    set_chat_handler(Script(bad, fixed, AIMessage("We sold some Lumen Bell units.")))
    out = ex.graph.invoke({"scenario": label_for("How many Lumen Doorbells")})
    first, second = tool_messages(out)
    assert first.status == "error" and "Did you mean 'Lumen Bell'" in first.content and second.status == "success"
    assert out["tool_rounds"] == 2


def test_picking_the_one_round_case_stops_the_loop_at_its_cap_and_still_answers():
    script = Script(lambda messages, tools: call("describe_dataset"))  # a model that never stops asking for tools
    set_chat_handler(script)
    out = ex.graph.invoke({"scenario": label_for("How did each product's revenue change")})
    assert len(tool_messages(out)) == 1 and out["tool_rounds"] == 1 and len(script.calls) == 2
    assert script.calls[-1][1] == [], "the second model call is made with no tools bound"
    assert "used all 1 of my tool calls" in out["messages"][-1].content
    assert_history_is_valid(out["messages"])

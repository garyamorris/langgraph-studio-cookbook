from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.config import data_dir
from deepdive.common.fakes import set_chat_handler
from deepdive.common.testing import check_example
from deepdive.examples import support_swarm as ex

ORDER = 1828  # Lumen Plug 4-pack, 2026-10-25, delivered: inside the return window on AS_OF


# --------------------------------------------------------------------------- scripted chat model
def which_agent(tools: list[dict]) -> str:
    names = {t["function"]["name"] for t in tools}
    if "lookup_order" in names:
        return "billing"
    return "technical" if "search_kb" in names else "triage"


def call(*calls) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{n}_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)])


def assert_valid_history(messages):
    """The OpenAI rule: every assistant tool call is answered by a tool message right after it, and vice versa."""
    open_calls: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in open_calls, f"tool message {m.tool_call_id} has no preceding tool call"
            open_calls.remove(m.tool_call_id)
        else:
            assert not open_calls, f"tool calls {open_calls} were never answered (next message: {type(m).__name__})"
            if isinstance(m, AIMessage):
                open_calls = [tc["id"] for tc in m.tool_calls]
    assert not open_calls, f"tool calls {open_calls} were never answered"


class Playbook:
    """A tiny customer-support script: who calls which tool depends on the agent and on the last message it sees."""

    def __init__(self):
        self.calls: list[tuple[str, set[str]]] = []  # (agent, tool names bound) per model call

    def __call__(self, messages, tools):
        agent = which_agent(tools)
        self.calls.append((agent, {t["function"]["name"] for t in tools}))
        last = messages[-1]
        text = str(last.content).lower() if isinstance(last, HumanMessage) else ""
        if agent == "triage":
            target = "technical" if any(w in text for w in ("offline", "wifi", "camera")) else "billing"
            return call((f"transfer_to_{target}", {"reason": f"customer needs {target}"}))
        if agent == "billing":
            if isinstance(last, HumanMessage) and "offline" in text:
                return call(("transfer_to_technical", {"reason": "camera dropping offline"}))
            if isinstance(last, HumanMessage) or (isinstance(last, ToolMessage) and last.name.startswith("transfer_to_")):
                return call(("lookup_order", {"order_id": ORDER}))
            if last.name == "lookup_order":
                return call(("check_refund_eligibility", {"order_id": ORDER}))
            return AIMessage(content="Order {order_id}: {reason}".format(**json.loads(last.content)))
        # technical
        if isinstance(last, HumanMessage) and "refund" in text:
            return call(("transfer_to_billing", {"reason": "refund question"}))
        if isinstance(last, HumanMessage) or last.name.startswith("transfer_to_"):
            return call(("search_kb", {"query": "camera drops offline wifi"}))
        return AIMessage(content="Try the 2.4 GHz steps in [device-setup-and-wifi].")

    def agents(self, since: int = 0) -> list[str]:
        return [a for a, _ in self.calls[since:]]


def new_graph():
    return ex.build_graph(checkpointer=InMemorySaver())


def cfg(thread="t1"):
    return {"configurable": {"thread_id": thread}, "recursion_limit": 60}


def say(g, text, thread="t1"):
    return g.invoke({"messages": [HumanMessage(text)]}, cfg(thread))


# --------------------------------------------------------------------------- tests
def test_contract():
    check_example(ex)


def test_triage_hands_off_to_billing_which_looks_up_the_order_and_answers():
    pb = Playbook()
    set_chat_handler(pb)
    out = say(new_graph(), "Can I return the plug from order 1828?")
    assert pb.agents() == ["triage", "billing", "billing", "billing"]
    assert out["active_agent"] == "billing" and out["mode"] == "offline"
    assert out["handoffs"] == [{"from": "triage", "to": "billing", "reason": "customer needs billing"}]
    tools_used = [m.name for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tools_used == ["transfer_to_billing", "lookup_order", "check_refund_eligibility"]
    lookup = json.loads(next(m.content for m in out["messages"] if isinstance(m, ToolMessage) and m.name == "lookup_order"))
    assert lookup["order_id"] == ORDER and lookup["items"][0]["product"] == "Lumen Plug 4-pack"
    final = out["messages"][-1]
    assert isinstance(final, AIMessage) and not final.tool_calls
    assert "inside the 30-day window" in final.content, "the answer comes from the deterministic eligibility tool"


def test_message_history_after_a_handoff_has_no_orphaned_tool_calls():
    set_chat_handler(Playbook())
    out = say(new_graph(), "Can I return the plug from order 1828?")
    assert_valid_history(out["messages"])
    kinds = [type(m).__name__ for m in out["messages"]]
    # Human, AI(transfer call), Tool(transferred) - adjacent - then billing's own calls
    assert kinds[:3] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert out["messages"][1].tool_calls[0]["id"] == out["messages"][2].tool_call_id
    assert len({m.id for m in out["messages"]}) == len(out["messages"]), "no message was duplicated by the handoff"


def test_the_next_turn_on_the_same_thread_resumes_with_the_agent_who_last_held_it():
    pb = Playbook()
    set_chat_handler(pb)
    g = new_graph()
    say(g, "Can I return the plug from order 1828?")
    n = len(pb.calls)
    out = say(g, "Thanks. Also my camera keeps dropping offline.")
    turn2 = pb.agents(n)
    assert turn2[0] == "billing", "turn 2 must start where turn 1 ended, not at triage"
    assert turn2 == ["billing", "technical", "technical"]
    assert out["active_agent"] == "technical"
    assert [h["to"] for h in out["handoffs"]] == ["billing", "technical"]
    assert_valid_history(out["messages"])
    assert "device-setup-and-wifi" in out["messages"][-1].content
    # and turn 3 starts with technical; a refund question sends it back to billing
    n = len(pb.calls)
    out = say(g, "Great. What about the refund?")
    assert pb.agents(n)[0] == "technical" and out["active_agent"] == "billing"
    assert [h["to"] for h in out["handoffs"]] == ["billing", "technical", "billing"]
    assert_valid_history(out["messages"])
    # the customer never had to repeat anything: the full history is what each agent was shown
    assert [m.content for m in out["messages"] if isinstance(m, HumanMessage)] == [
        "Can I return the plug from order 1828?", "Thanks. Also my camera keeps dropping offline.", "Great. What about the refund?"]


def test_a_new_thread_starts_at_triage_again():
    pb = Playbook()
    set_chat_handler(pb)
    g = new_graph()
    say(g, "Can I return the plug from order 1828?", thread="a")
    n = len(pb.calls)
    say(g, "Hello, my wifi is broken", thread="b")
    assert pb.agents(n)[0] == "triage"


def test_each_agent_gets_only_its_own_tools():
    pb = Playbook()
    set_chat_handler(pb)
    g = new_graph()
    say(g, "Can I return the plug from order 1828?")
    say(g, "Also my camera keeps dropping offline.")
    bound = {}
    for agent, names in pb.calls:
        bound.setdefault(agent, set()).update(names)
    assert bound == {
        "triage": {"transfer_to_billing", "transfer_to_technical"},
        "billing": {"lookup_order", "check_refund_eligibility", "transfer_to_technical"},
        "technical": {"search_kb", "transfer_to_billing"},
    }


def test_ping_pong_handoffs_are_capped_per_user_turn():
    calls = []

    def pingpong(messages, tools):
        agent = which_agent(tools)
        calls.append(agent)
        last = messages[-1]
        if isinstance(last, ToolMessage) and str(last.content).startswith("Transfer refused"):
            return AIMessage(content="I'll answer this myself.")
        other = {"triage": "billing", "billing": "technical", "technical": "billing"}[agent]
        return call((f"transfer_to_{other}", {"reason": "not mine"}))

    set_chat_handler(pingpong)
    out = say(new_graph(), "Something confusing")
    assert len(out["handoffs"]) == ex.MAX_HANDOFFS_PER_TURN
    # the 4th transfer attempt (by billing) is refused, then billing's next model call answers in text
    assert calls == ["triage", "billing", "technical", "billing", "billing"]
    refused = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.content.startswith("Transfer refused")]
    assert len(refused) == 1
    assert out["messages"][-1].content == "I'll answer this myself."
    assert out["active_agent"] == "billing"
    assert_valid_history(out["messages"])


def test_handoff_budget_resets_with_each_user_message():
    calls = []

    def chat(messages, tools):
        agent = which_agent(tools)
        calls.append(agent)
        last = messages[-1]
        if isinstance(last, HumanMessage) and agent != "triage":
            return call((f"transfer_to_{'technical' if agent == 'billing' else 'billing'}", {"reason": "swap"}))
        if agent == "triage":
            return call(("transfer_to_billing", {"reason": "r"}))
        return AIMessage(content="ok")

    set_chat_handler(chat)
    g = new_graph()
    for i in range(5):  # five user turns with one transfer each: more than the per-turn cap of 3 in total
        out = say(g, f"message {i}")
    assert len(out["handoffs"]) == 5 > ex.MAX_HANDOFFS_PER_TURN and not any(
        isinstance(m, ToolMessage) and m.content.startswith("Transfer refused") for m in out["messages"])
    assert_valid_history(out["messages"])


def test_runaway_tool_loop_is_cut_off_without_another_model_call():
    n_calls = []

    def loop(messages, tools):
        n_calls.append(which_agent(tools))
        return call(("lookup_order", {"order_id": ORDER})) if which_agent(tools) == "billing" else call(("transfer_to_billing", {"reason": "r"}))

    set_chat_handler(loop)
    out = say(new_graph(), "order help")
    lookups = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.name == "lookup_order"]
    assert len(lookups) == ex.MAX_TOOL_ROUNDS - 1  # the transfer's own tool message counts as one round
    assert out["messages"][-1].content == ex.FALLBACK_REPLY
    assert n_calls.count("billing") == ex.MAX_TOOL_ROUNDS - 1, "the fallback reply costs no model call"
    assert_valid_history(out["messages"])


def test_a_transfer_in_a_message_with_other_tool_calls_still_leaves_a_valid_history():
    state = {"first": True}

    def chat(messages, tools):
        agent = which_agent(tools)
        if agent == "triage":
            return call(("transfer_to_billing", {"reason": "r"}))
        if agent == "billing" and state["first"]:
            state["first"] = False
            return call(("lookup_order", {"order_id": ORDER}), ("transfer_to_technical", {"reason": "also a fault"}))
        return AIMessage(content="tech here")

    set_chat_handler(chat)
    out = say(new_graph(), "hi")
    assert_valid_history(out["messages"])
    placeholders = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.content.startswith("Not executed")]
    assert [m.name for m in placeholders] == ["lookup_order"]
    assert out["active_agent"] == "technical" and len(out["handoffs"]) == 2


def test_start_routing_follows_active_agent():
    assert ex.route_to_active_agent({}) == "triage"
    assert ex.route_to_active_agent({"active_agent": "technical"}) == "technical"
    assert ex.route_to_active_agent({"active_agent": "nonsense"}) == "triage"


# --------------------------------------------------------------------------- eligibility rule
def _order(order_id=1767):
    con = sqlite3.connect(f"file:{data_dir() / 'store.sqlite'}?mode=ro", uri=True)
    try:
        return con.execute("SELECT order_date, status FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()


@pytest.mark.parametrize("days,eligible", [(0, True), (1, True), (29, True), (30, True), (31, False), (32, False), (400, False)])
def test_thirty_day_window_boundary(days, eligible):
    order_date, status = _order(1767)
    assert status == "delivered"
    delivered = date.fromisoformat(order_date) + timedelta(days=ex.ASSUMED_TRANSIT_DAYS)
    r = ex.refund_eligibility(1767, as_of=delivered + timedelta(days=days))
    assert r["days_since_delivery"] == days
    assert r["eligible_for_return_refund"] is eligible and r["within_return_window"] is eligible
    assert (str(days) + " days ago") in r["reason"]
    assert ("outside the 30-day window" in r["reason"]) is (not eligible)


def test_the_default_as_of_date_puts_order_1767_exactly_on_day_30():
    r = ex.refund_eligibility(1767)
    assert r["as_of"] == "2026-11-06" and r["days_since_delivery"] == 30 and r["eligible_for_return_refund"] is True
    assert ex.refund_eligibility(ORDER)["eligible_for_return_refund"] is True
    assert ex.refund_eligibility(1677)["eligible_for_return_refund"] is False  # delivered long ago


def test_eligibility_before_delivery_and_for_cancelled_returned_and_unknown_orders():
    order_date, _ = _order(1767)
    day_before = date.fromisoformat(order_date) + timedelta(days=ex.ASSUMED_TRANSIT_DAYS - 2)
    r = ex.refund_eligibility(1767, as_of=day_before)
    assert r["eligible_for_return_refund"] is False and "Not delivered yet" in r["reason"]
    assert "cancelled" in ex.refund_eligibility(1840)["reason"] and not ex.refund_eligibility(1840)["eligible_for_return_refund"]
    returned = ex.refund_eligibility(1836)
    assert not returned["eligible_for_return_refund"] and "one refund per order" in returned["reason"]
    missing = ex.refund_eligibility(999_999)
    assert missing["found"] is False and not missing["eligible_for_return_refund"]


def test_warranty_lasts_longer_for_locks_than_for_plugs():
    lock_order = 1838  # Lumen Lock
    order_date, _ = _order(lock_order)
    delivered = date.fromisoformat(order_date) + timedelta(days=ex.ASSUMED_TRANSIT_DAYS)
    r = ex.refund_eligibility(lock_order, as_of=delivered + timedelta(days=400))
    assert r["warranty"] == [{"product": "Lumen Lock", "covered_by_warranty": True}]
    assert ex.refund_eligibility(1828, as_of=date(2028, 1, 1))["warranty"][0]["covered_by_warranty"] is False


def test_lookup_order_returns_database_facts_and_rejects_non_numeric_ids():
    data = json.loads(ex.lookup_order.invoke({"order_id": ORDER}))
    con = sqlite3.connect(f"file:{data_dir() / 'store.sqlite'}?mode=ro", uri=True)
    expect = con.execute("SELECT SUM(quantity*unit_price) FROM order_items WHERE order_id = ?", (ORDER,)).fetchone()[0]
    discount = con.execute("SELECT discount_pct FROM orders WHERE order_id = ?", (ORDER,)).fetchone()[0]
    con.close()
    assert data["subtotal"] == pytest.approx(expect) and data["total"] == pytest.approx(expect * (1 - discount / 100), abs=0.01)
    assert "customer" not in json.dumps(data).lower(), "no customer identity in the tool output"
    assert json.loads(ex.lookup_order.invoke({"order_id": 999_999}))["found"] is False
    with pytest.raises(Exception):  # the argument is an int: an SQL fragment never reaches the query
        ex.lookup_order.invoke({"order_id": "1 OR 1=1"})


def test_search_kb_cites_public_article_ids_and_never_returns_internal_articles():
    text = ex.search_kb.invoke({"query": "camera keeps dropping offline wifi 2.4 GHz"})
    assert "[device-setup-and-wifi]" in text
    internal = ex.search_kb.invoke({"query": "refund approval matrix manager approval fraud review authority levels"})
    assert "refund-approval-matrix" not in internal


# --------------------------------------------------------------------------- ready-made cases (the Studio dropdown)
def case(prefix: str) -> str:
    return next(k for k in ex.CASES if k.startswith(prefix))


def test_a_blank_form_runs_the_first_case_but_never_injects_a_message_into_a_thread_with_history():
    pb = Playbook()
    set_chat_handler(pb)
    g = new_graph()
    out = g.invoke({}, cfg())  # Studio's Submit on an empty form
    assert out["messages"][0].content == ex.EXAMPLE_INPUT["messages"][0]["content"] and pb.agents()[0] == "triage"
    humans = lambda: [m.content for m in g.get_state(cfg()).values["messages"] if isinstance(m, HumanMessage)]  # noqa: E731
    before = humans()
    set_chat_handler(lambda messages, tools: AIMessage(content="Anything else?"))
    g.invoke({}, cfg())  # the thread already has a chat: nothing may be typed on the customer's behalf
    assert humans() == before and len(before) == 1
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        g.invoke({"scenario": "not a case"}, cfg())


def test_the_input_schema_keeps_messages_for_studios_chat_view_and_offers_the_dropdown():
    schema = new_graph().get_input_jsonschema()
    assert {"messages", "scenario"} <= set(schema["properties"]) and not schema.get("required")


def test_a_picked_case_is_a_new_contact_that_starts_at_triage_even_on_a_thread_another_agent_holds():
    pb = Playbook()
    set_chat_handler(pb)
    g = new_graph()
    g.invoke({"messages": [HumanMessage("Hello, my camera keeps going offline")]}, cfg())
    assert g.get_state(cfg()).values["active_agent"] == "technical"
    n = len(pb.calls)
    out = g.invoke({"scenario": case("Order 1828 (Plug")}, cfg())
    assert pb.agents(n)[0] == "triage", "a case starts at the front desk"
    assert [m.content for m in out["messages"] if isinstance(m, HumanMessage)][-1] == ex.EXAMPLE_INPUT["messages"][0]["content"]
    assert len([m for m in out["messages"] if isinstance(m, HumanMessage)]) == 2, "added to the chat, not replacing it"

    n = len(pb.calls)  # but a typed message still resumes with whoever held the conversation
    g.invoke({"messages": [HumanMessage("Thanks. Also my camera keeps dropping offline.")]}, cfg())
    assert pb.agents(n)[0] == "billing"


def test_the_order_cases_state_the_policy_verdict_the_code_really_gives():
    seen = 0
    for label, fields in ex.CASES.items():
        text = fields["messages"][0]["content"]
        if not label.startswith("Order "):
            continue
        order_id = int(label.split()[1])
        assert str(order_id) in text, f"{label}: the message must mention the order the label leads with"
        order, verdict = ex._fetch_order(order_id), ex.refund_eligibility(order_id)
        assert order is not None, f"{label}: order {order_id} is not in store.sqlite"
        seen += 1
        if "policy check:" not in label:  # the safety case: no verdict is claimed
            continue
        eligible = verdict["eligible_for_return_refund"]
        assert label.endswith("policy check: ELIGIBLE") if eligible else label.endswith("policy check: NOT eligible"), label
        if "days since delivery" in label:
            assert f"{verdict['days_since_delivery']} days since delivery" in label, label
        for product in ("Plug 4-pack", "Bell", "Therm"):
            if product in label:
                assert any(product in i["product"] for i in order["items"]), f"{label}: {product} is not in order {order_id}"
    assert seen == 4, "the plug, the Bell + Plug, the refunded Therm order, and the hot plug in order 1828"


def test_the_mixed_case_needs_two_transfers_which_fits_under_the_cap_of_three():
    def chat(messages, tools):
        agent, last = which_agent(tools), messages[-1]
        if agent == "triage":
            return call(("transfer_to_billing", {"reason": "return from order 1828, plus a camera fault"}))
        if agent == "billing":
            if isinstance(last, ToolMessage) and last.name == "lookup_order":
                return call(("transfer_to_technical", {"reason": "Cam Pro drops off wifi"}))
            return call(("lookup_order", {"order_id": ORDER}))
        if isinstance(last, ToolMessage) and last.name == "search_kb":
            return AIMessage(content="See [device-setup-and-wifi].")
        return call(("search_kb", {"query": "camera drops offline wifi"}))

    set_chat_handler(chat)
    out = new_graph().invoke({"scenario": case("Return from order 1828 AND")}, cfg())
    assert [(h["from"], h["to"]) for h in out["handoffs"]] == [("triage", "billing"), ("billing", "technical")]
    assert len(out["handoffs"]) <= ex.MAX_HANDOFFS_PER_TURN and out["active_agent"] == "technical"
    assert_valid_history(out["messages"])


def test_the_technical_and_safety_cases_use_wording_from_the_ticket_data():
    from deepdive.common.data import read_csv

    bodies = " ".join(read_csv("support_tickets.csv").body)
    assert "drops off wifi every few hours. I rebooted the router twice and the app shows error E-102" in bodies
    assert "got extremely hot and there's a burning plastic smell" in bodies
    texts = [f["messages"][0]["content"] for f in ex.CASES.values()]
    assert any("drops off wifi every few hours. I rebooted the router twice and the app shows error E-102" in t for t in texts)
    assert any("got extremely hot and there's a burning plastic smell" in t for t in texts)

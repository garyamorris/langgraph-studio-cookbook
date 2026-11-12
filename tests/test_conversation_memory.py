from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.fakes import set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import conversation_memory as ex

COMMITMENT = "I will submit a refund request for order #1042 and you will hear back within 2 business days."


class ScriptedModel:
    """Chat stand-in that tells the three kinds of call apart and records exactly what each was shown."""

    def __init__(self, summary_text: str | None = None):
        self.replies: list[list] = []  # prompts of respond calls
        self.summaries: list[list] = []  # prompts of summarize calls
        self.summary_text = summary_text
        set_chat_handler(self.handler)
        set_structured_handler(ex.FactUpdate, self.facts)

    def handler(self, messages, tools):
        system = str(messages[0].content)
        if system.startswith(ex.SUMMARY_PROMPT[:40]):
            self.summaries.append(list(messages))
            text = f"SUMMARY v{len(self.summaries)}: Priya, order #1042 cracked; assistant promised a refund request (2 business days); refund still open."
            return AIMessage(text if self.summary_text is None else self.summary_text)
        self.replies.append(list(messages))
        n = len(self.replies)
        return AIMessage(COMMITMENT if "refund" in str(messages[-1].content).lower() and n < 5 else f"Answer {n}.")

    @staticmethod
    def facts(prompt_text: str) -> ex.FactUpdate:
        name = "Priya" if "I'm Priya" in prompt_text else "Sam" if "I'm Sam" in prompt_text else None
        return ex.FactUpdate(customer_name=name, order_ids=[], commitments=[COMMITMENT] if "Assistant: " + COMMITMENT in prompt_text else [])


def say(graph, config, text, **extra):
    return graph.invoke({"messages": [{"role": "user", "content": text}], **extra}, config)


def cfg(name="t"):
    return {"configurable": {"thread_id": name}}


def stored(graph, config):
    return graph.get_state(config).values["messages"]


def system_text(prompt):
    assert isinstance(prompt[0], SystemMessage)
    return str(prompt[0].content)


def run_turns(graph, config, n, first="Hi, I'm Priya. Order #1042 arrived with a cracked case.", **extra):
    """Turn 1 introduces the customer, turn 2 asks for a refund (the assistant promises), the rest is chatter."""
    outs = []
    for i in range(1, n + 1):
        text = first if i == 1 else "Please refund it." if i == 2 else f"Unrelated question number {i} about the hub."
        outs.append(say(graph, config, text, **(extra if i == 1 else {})))
    return outs


# --------------------------------------------------------------------------- contract
def test_example_contract():
    res = check_example(ex)
    assert [m.type for m in res.output["messages"]] == ["human", "ai"]
    assert "#1042" in res.output["facts"]["order_ids"] and res.output["turn"] == 1 and res.output["mode"] == "offline"


# --------------------------------------------------------------------------- the main behaviour: 20 turns on one thread
def test_twenty_turns_stay_bounded_and_the_first_order_number_is_still_in_the_last_prompt():
    model = ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    sizes = [len(out["messages"]) for out in run_turns(graph, c, 20)]
    assert sizes == [2, 4, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8], "grows to the limit, folds to keep_last, repeats"

    final = graph.get_state(c).values
    assert final["summary"].startswith("SUMMARY v4") and final["folded_messages"] == 32
    assert final["turn"] == 20 and len(model.replies) == 20 and len(model.summaries) == 4

    last = model.replies[-1]
    system = system_text(last)
    assert "#1042" in system and "Priya" in system and COMMITMENT in system, "pinned facts reach the prompt at turn 20"
    assert "SUMMARY v" in system, "so does the rolling summary"
    window_text = " ".join(str(m.content) for m in last[1:])
    assert "cracked" not in window_text and "#1042" not in window_text, "turn 1 is long gone from the verbatim window"
    assert len(last) - 1 <= ex.DEFAULT_MAX_MESSAGES + 1
    assert "#1042" not in " ".join(str(m.content) for m in stored(graph, c) if m.type == "human"), "naive truncation would have forgotten the order"


def test_the_summary_prompt_tells_the_model_what_must_survive():
    model = ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    run_turns(graph, c, 12)
    assert len(model.summaries) == 2
    first, second = model.summaries
    instructions = system_text(first).lower()
    for must in ("commitment", "open issues", "order number", "never invent"):
        assert must in instructions
    body = str(first[1].content)
    assert "PREVIOUS SUMMARY\n(none yet)" in body and "Customer: Hi, I'm Priya" in body, "the oldest messages are what gets folded"
    assert "PREVIOUS SUMMARY\nSUMMARY v1" in str(second[1].content), "later folds receive the previous summary"


def test_folding_removes_exactly_the_oldest_messages_and_keeps_a_window_that_starts_on_a_human_message():
    ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    outs = run_turns(graph, c, 7)  # 14 messages > 12 -> fold down to keep_last
    msgs = outs[-1]["messages"]
    assert len(msgs) == ex.DEFAULT_KEEP_LAST == 6
    assert msgs[0].type == "human" and msgs[-1].type == "ai"
    assert "Unrelated question number 7" in str(msgs[-2].content)  # the newest exchange survived
    assert outs[-1]["folded_messages"] == 8


def test_choose_fold_boundaries():
    def chat(n):
        return [HumanMessage(f"h{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(n)]

    assert ex.choose_fold(chat(14), 6) == chat(14)[:8]
    assert ex.choose_fold(chat(4), 6) == []
    odd = ex.choose_fold(chat(14), 5)  # cut at 9 would start on an AI message; move to the next human message (index 10)
    assert len(odd) == 10 and chat(14)[len(odd)].type == "human"
    assert ex.choose_fold([AIMessage("a")] * 5, 2) == [AIMessage("a")] * 3, "no human to land on: fall back to the plain cut"


def test_max_messages_and_keep_last_can_be_overridden_per_thread_and_persist():
    ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    sizes = [len(o["messages"]) for o in run_turns(graph, c, 8, max_messages=4, keep_last=2)]
    assert max(sizes) <= 4 and graph.get_state(c).values["max_messages"] == 4, sizes
    assert ex.limits({"max_messages": 1, "keep_last": 50}) == (4, 2), "nonsense limits are clamped"
    assert ex.limits({}) == (12, 6)


# --------------------------------------------------------------------------- pinned facts
def test_facts_are_pinned_from_both_sides_of_the_conversation():
    ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    run_turns(graph, c, 2)
    facts = graph.get_state(c).values["facts"]
    assert facts["customer_name"] == "Priya" and facts["order_ids"] == ["#1042"] and facts["commitments"] == [COMMITMENT]


def test_merge_never_overwrites_with_empty_and_never_duplicates():
    base = {"customer_name": "Priya", "order_ids": ["#1042"], "commitments": ["Refund request for #1042"]}
    assert ex.merge_facts(base, {"customer_name": "", "order_ids": [], "commitments": []}) == base
    assert ex.merge_facts(base, {"customer_name": None}) == base
    assert ex.merge_facts(base, None) == base
    merged = ex.merge_facts(base, {"customer_name": "Priya K.", "order_ids": ["#1042", "#2077"], "commitments": ["  refund request  for #1042", "Call back Friday"]})
    assert merged == {"customer_name": "Priya K.", "order_ids": ["#1042", "#2077"], "commitments": ["Refund request for #1042", "Call back Friday"]}
    assert ex.merge_facts(None, {"order_ids": ["#1"]})["customer_name"] == ""


def test_pinned_lists_are_capped_so_the_prompt_stays_bounded():
    merged = ex.merge_facts({}, {"order_ids": [f"#{1000 + i}" for i in range(ex.MAX_PINNED + 5)]})
    assert len(merged["order_ids"]) == ex.MAX_PINNED and merged["order_ids"][-1] == f"#{1000 + ex.MAX_PINNED + 4}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my order #1042 is broken", ["#1042"]),
        ("Order 1042 and ORDER NUMBER 2077", ["#1042", "#2077"]),
        ("order no. 3141, then #3141 again", ["#3141"]),
        ("Order ID: 98765", ["#98765"]),
        ("ticket #55 and 12 orders", []),
        ("it arrived 5 days after I placed the order", []),
        ("call me on 07700 900123", []),
    ],
)
def test_order_number_regex(text, expected):
    assert ex.extract_order_ids(text) == expected


def test_the_regex_pins_order_numbers_even_when_the_extractor_misses_or_fails():
    ScriptedModel()
    set_structured_handler(ex.FactUpdate, lambda text: ex.FactUpdate(customer_name=None, order_ids=[], commitments=[]))
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    say(graph, c, "Order #7788 never arrived")
    assert graph.get_state(c).values["facts"]["order_ids"] == ["#7788"]

    def boom(text):
        raise RuntimeError("structured output failed")

    set_structured_handler(ex.FactUpdate, boom)
    out = say(graph, c, "Also order 9911 is late")
    assert out["facts"]["order_ids"] == ["#7788", "#9911"] and out["turn"] == 2, "the chat carried on"


def test_a_model_that_formats_the_order_number_differently_is_normalised():
    ScriptedModel()
    set_structured_handler(ex.FactUpdate, lambda text: ex.FactUpdate(customer_name="Sam", order_ids=["order 4455"], commitments=[]))
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    assert say(graph, c, "hello")["facts"]["order_ids"] == ["#4455"]


# --------------------------------------------------------------------------- failure rules of the summariser
def test_an_empty_summary_removes_nothing_and_is_retried_next_turn():
    model = ScriptedModel(summary_text="   ")
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    outs = run_turns(graph, c, 8)
    assert len(outs[-1]["messages"]) == 16, "nothing may be deleted without a summary to replace it"
    assert not graph.get_state(c).values.get("summary") and len(model.summaries) >= 2, "it tried again on later turns"
    model.summary_text = "SUMMARY recovered: order #1042 refund pending."
    out = say(graph, c, "anything new?")
    assert out["summary"].startswith("SUMMARY recovered") and len(out["messages"]) <= ex.DEFAULT_MAX_MESSAGES


def test_a_runaway_summary_is_cut_to_the_guard_length():
    ScriptedModel(summary_text="x" * 10_000)
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    run_turns(graph, c, 8)
    assert len(graph.get_state(c).values["summary"]) == ex.SUMMARY_MAX_CHARS


# --------------------------------------------------------------------------- threads
def test_two_threads_are_isolated():
    model = ScriptedModel()
    graph = ex.build_graph(checkpointer=InMemorySaver())
    a, b = cfg("alice"), cfg("sam")
    for i in range(1, 5):
        say(graph, a, "Hi, I'm Priya. Order #1042 is cracked." if i == 1 else f"alice question {i}")
        say(graph, b, "Hi, I'm Sam. Order #2077 never came." if i == 1 else f"sam question {i}")
    fa, fb = graph.get_state(a).values["facts"], graph.get_state(b).values["facts"]
    assert (fa["customer_name"], fa["order_ids"]) == ("Priya", ["#1042"])
    assert (fb["customer_name"], fb["order_ids"]) == ("Sam", ["#2077"])
    sam_prompts = " ".join(str(m.content) for p in model.replies[1::2] for m in p)
    assert "Priya" not in sam_prompts and "#1042" not in sam_prompts, "no memory leaks across threads"
    alice_prompts = " ".join(str(m.content) for p in model.replies[0::2] for m in p)
    assert "Sam" not in alice_prompts.replace("Sample", "") and "#2077" not in alice_prompts


def test_a_resumed_thread_continues_from_its_saved_state():
    model = ScriptedModel()
    saver = InMemorySaver()
    c = cfg("long-chat")
    run_turns(ex.build_graph(checkpointer=saver), c, 10)
    before = ex.build_graph(checkpointer=saver).get_state(c).values
    assert before["turn"] == 10 and before["summary"]

    restarted = ex.build_graph(checkpointer=saver)  # a brand-new graph object: think "server restarted"
    out = say(restarted, c, "Sorry, which order was this about again?")
    assert out["turn"] == 11 and out["folded_messages"] >= before["folded_messages"]
    system = system_text(model.replies[-1])
    assert "#1042" in system and "Priya" in system and "SUMMARY v" in system
    assert [m.type for m in out["messages"]][-2:] == ["human", "ai"]


# --------------------------------------------------------------------------- streaming
def test_only_the_reply_is_streamed_to_the_customer_not_the_extractor_or_summariser():
    model = ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    run_turns(graph, c, 3, max_messages=4, keep_last=2)  # the fourth message pushes the window over the limit
    chunks = list(graph.stream({"messages": [{"role": "user", "content": "one more question please"}]}, c, stream_mode="messages"))
    assert chunks and len(model.summaries) >= 1, "this turn really did run the summariser"
    assert {meta["langgraph_node"] for _, meta in chunks} == {"respond"}
    streamed = "".join(chunk.content for chunk, _ in chunks)
    assert streamed.strip() == graph.get_state(c).values["messages"][-1].content.strip()


def test_memory_changes_are_emitted_as_custom_events():
    ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    first = list(graph.stream({"messages": [{"role": "user", "content": "Hi, I'm Priya. Order #1042 is cracked."}], "max_messages": 4, "keep_last": 2}, c, stream_mode="custom"))
    assert first and first[0]["event"] == "memory_updated"
    assert first[0]["added"]["customer_name"] == "Priya" and first[0]["added"]["order_ids"] == ["#1042"]
    events = []
    for i in range(3):
        events += list(graph.stream({"messages": [{"role": "user", "content": f"chatter {i}"}]}, c, stream_mode="custom"))
    folded = [e for e in events if e["event"] == "summarized"]
    assert folded and folded[0]["folded"] >= 2 and folded[0]["kept"] == 2


# --------------------------------------------------------------------------- ready-made cases (the Studio dropdown)
CASE_EXPECT = {
    # label prefix: (order numbers pinned after the first Submit, folded_messages, messages left in the window)
    "Pasted 5-message": (["#1042"], 4, 2),
    "Priya + order #1042": (["#1042"], 0, 2),
    "Long message": (["#1042", "#1828", "#1838"], 0, 2),
    "Order-number spellings": (["#1828", "#1677", "#1767"], 0, 2),
    "Nonsense settings": (["#1767"], 4, 2),
}


def test_a_blank_form_runs_the_first_case_which_folds_at_once_but_never_injects_into_a_thread_with_history():
    model = ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    out = graph.invoke({}, c)  # Studio's Submit on an empty form
    assert len(model.summaries) == 1 and out["folded_messages"] == 4 and [m.type for m in out["messages"]] == ["human", "ai"]
    assert out["summary"].startswith("SUMMARY v1") and out["facts"]["order_ids"] == ["#1042"] and out["facts"]["customer_name"] == "Priya"
    assert out["turn"] == 1 and out["mode"] == "offline"

    humans = lambda: [m.content for m in graph.get_state(c).values["messages"] if m.type == "human"]  # noqa: E731
    before = humans()
    graph.invoke({}, c)  # a blank form on a thread that already has a chat must not add an opening message of its own
    assert humans() == before
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        graph.invoke({"scenario": "not a case"}, c)


def test_every_ready_made_case_does_what_its_label_claims():
    assert len(ex.CASES) == len(CASE_EXPECT), "a new case needs an entry in CASE_EXPECT"
    for label, fields in ex.CASES.items():
        order_ids, folded, kept = next(v for k, v in CASE_EXPECT.items() if label.startswith(k))
        ScriptedModel()
        graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
        out = graph.invoke({"scenario": label}, c)
        assert out["facts"]["order_ids"] == order_ids, label
        assert out.get("folded_messages", 0) == folded and len(out["messages"]) == kept, label
        assert bool(out.get("summary")) is bool(folded), label
        assert graph.get_state(c).values.get("scenario") is None
        for number in ("max_messages", "keep_last"):  # numbers quoted in a label are the numbers the case sets
            if f"{number}=" in label:
                assert f"{number}={fields[number]}" in label, label
        if label.startswith("Nonsense"):
            assert ex.limits(graph.get_state(c).values) == (4, 2)


def test_the_long_message_case_folds_two_messages_later_with_all_three_orders_still_pinned():
    model = ScriptedModel()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    label = next(k for k in ex.CASES if k.startswith("Long message"))
    graph.invoke({"scenario": label}, c)
    say(graph, c, "ok, one more thing: the hub light")
    assert not model.summaries
    out = say(graph, c, "and the app crashes at login")  # 6 stored messages > max_messages 4
    assert len(model.summaries) == 1 and out["folded_messages"] == 4 and out["summary"]
    assert out["facts"]["order_ids"] == ["#1042", "#1828", "#1838"]


def test_the_orders_in_the_cases_exist_in_the_shop_database():
    from deepdive.common.data import db_connect

    con = db_connect()
    try:
        known = {f"#{r[0]}" for r in con.execute("SELECT order_id FROM orders")}
    finally:
        con.close()
    mentioned = {oid for fields in ex.CASES.values() for m in fields["messages"] for oid in ex.extract_order_ids(m["content"])}
    assert mentioned and mentioned <= known, mentioned - known

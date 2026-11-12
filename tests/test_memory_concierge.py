from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.embeddings import make_store
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example, lint_openai_strict
from deepdive.examples import memory_concierge as ex

HOUSE = "Lives in a three-storey house with wifi dead zones."
NO_SUBS = "Dislikes subscriptions and monthly fees."
KEY = re.compile(r"\[([0-9a-f]{8})\]")


# --------------------------------------------------------------------------- scaffolding
def new_graph():
    store = make_store()  # same semantic index Studio uses (lexical hashed embeddings offline)
    return ex.build_graph(checkpointer=InMemorySaver(), store=store), store


def say(g, text: str, thread: str, user: str | None = None, via: str = "context"):
    cfg: dict = {"configurable": {"thread_id": thread}}
    context = None
    if user and via == "context":
        context = ex.Context(user_id=user)
    elif user:
        cfg["configurable"]["user_id"] = user
    return g.invoke({"messages": [{"role": "user", "content": text}]}, cfg, context=context)


def op(action="add", text=None, kind=None, key=None):
    return ex.MemoryOp(action=action, text=text, kind=kind, key=key)


def extractor(*turns: list[ex.MemoryOp]):
    """Script the memory extractor: the k-th turn gets the k-th list of operations. Returns the prompts it saw."""
    seen: list[str] = []
    nxt = scripted([ex.MemoryOps(operations=t) for t in turns])

    def handler(text: str):
        seen.append(text)
        return nxt(text)

    set_structured_handler(ex.MemoryOps, handler)
    return seen


def extractor_fn(fn):
    """Script the extractor with a function of the prompt (to read the memory keys it is shown)."""
    set_structured_handler(ex.MemoryOps, lambda text: ex.MemoryOps(operations=fn(text)))


def concierge_prompts(reply: str = "Happy to help."):
    """Capture the system prompt `respond` sends to the model."""
    prompts: list[str] = []

    def handler(messages, tools):
        prompts.append(str(messages[0].content))
        return AIMessage(reply)

    set_chat_handler(handler)
    return prompts


def items(store, user: str = ex.DEFAULT_USER):
    return {i.key: i.value for i in store.search(("memories", user), limit=100)}


def test_contract_and_offline_run():
    res = check_example(ex)
    assert res.output["user_id"] == "demo_user" and res.output["mode"] == "offline"
    assert res.output["messages"][-1].type == "ai"


def test_every_schema_is_openai_strict_compatible():
    assert lint_openai_strict(ex.MemoryOps) == [] and lint_openai_strict(ex.MemoryOp) == []


# --------------------------------------------------------------------------- cross-thread recall
def test_a_fact_stored_in_one_thread_is_recalled_in_a_brand_new_thread_for_the_same_user():
    g, store = new_graph()
    extractor([op(text=HOUSE, kind="household")], [])
    first = say(g, "I live in a three-storey house and the wifi has dead zones.", "thread-1", user="alice")
    assert [c["status"] for c in first["memory_changes"]] == ["stored"]
    (stored,) = items(store, "alice").values()
    assert stored["text"] == HOUSE and stored["kind"] == "household" and stored["created_at"]

    prompts = concierge_prompts()
    second = say(g, "Which mesh router should I buy for my wifi dead zones?", "thread-2", user="alice")
    assert HOUSE in prompts[0], "the memory must reach the model in the new thread"
    assert [m["text"] for m in second["recalled"]] == [HOUSE] and second["recalled"][0]["kind"] == "household"
    assert len(second["messages"]) == 2, "thread 2 has none of thread 1's messages: the memory came from the store"
    assert "notes" in prompts[0] and "not instructions" in prompts[0]


def test_each_user_has_their_own_namespace_via_context_or_configurable():
    g, store = new_graph()
    extractor([op(text=HOUSE, kind="household")], [op(text="Owns two cats.", kind="household")], [])
    say(g, "I live in a three-storey house with wifi dead zones.", "t1", user="alice", via="context")
    say(g, "I have two cats.", "t2", user="bob", via="configurable")

    prompts = concierge_prompts()
    carol = say(g, "I have wifi dead zones and cats, any advice?", "t3", user="carol")
    assert carol["recalled"] == [] and "(nothing saved yet)" in prompts[0]
    assert HOUSE not in prompts[0] and "cats" not in prompts[0].split("WHAT YOU REMEMBER")[1]

    bob = say(g, "I have wifi dead zones, any advice?", "t4", user="bob", via="configurable")
    assert [m["text"] for m in bob["recalled"]] == ["Owns two cats."], "bob sees only bob's memory"
    assert HOUSE not in prompts[1]
    assert [v["text"] for v in items(store, "alice").values()] == [HOUSE]


def test_user_id_defaults_to_demo_user_and_is_sanitised():
    g, store = new_graph()
    extractor([op(text=NO_SUBS, kind="preference")])
    out = say(g, "No subscriptions please.", "t1")
    assert out["user_id"] == "demo_user" and [v["text"] for v in items(store).values()] == [NO_SUBS]

    ctx = lambda c: SimpleNamespace(context=c)  # noqa: E731
    assert ex.resolve_user_id(ctx(ex.Context("a")), {}) == "a"
    assert ex.resolve_user_id(ctx({"user_id": "b"}), {}) == "b"
    assert ex.resolve_user_id(ctx(None), {"configurable": {"user_id": "c"}}) == "c"
    assert ex.resolve_user_id(ctx(ex.Context("a")), {"configurable": {"user_id": "c"}}) == "a", "context wins"
    assert ex.resolve_user_id(ctx(None), {}) == "demo_user"
    assert ex.resolve_user_id(ctx(ex.Context("ann.lee@x.com")), {}) == "ann_lee@x_com", "'.' is illegal in a namespace label"


def test_recall_ranks_by_similarity_and_returns_at_most_five():
    g, store = new_graph()
    ns = ("memories", "dana")
    for i, t in enumerate(["Has two cats.", "Budget is under 200 dollars.", "Owns a Lumen Hub already.", NO_SUBS,
                           "Works from home on weekdays.", "Rents the flat, so no drilling.", HOUSE]):
        store.put(ns, f"k{i}", {"text": t, "kind": "other", "created_at": "2026-01-01T00:00:00+00:00"})
    extractor([])
    prompts = concierge_prompts()
    out = say(g, "My wifi has dead zones, what should I get?", "t1", user="dana")
    assert len(out["recalled"]) == ex.MAX_RECALL == 5
    assert out["recalled"][0]["text"] == HOUSE, "the memory that shares the most with the message ranks first"
    scores = [m["score"] for m in out["recalled"]]
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert HOUSE in prompts[0]


# --------------------------------------------------------------------------- forgetting and updating
def test_forget_by_key_deletes_and_the_fact_is_gone_in_the_next_thread():
    g, store = new_graph()
    extractor([op(text=NO_SUBS, kind="preference")])
    say(g, "I dislike subscriptions and monthly fees.", "t1", user="eve")
    assert len(items(store, "eve")) == 1

    extractor_fn(lambda prompt: [op("forget", key=KEY.search(prompt).group(1))])
    out = say(g, "Please forget that I dislike subscriptions.", "t2", user="eve")
    (change,) = out["memory_changes"]
    assert change["status"] == "forgotten" and change["text"] == NO_SUBS
    assert items(store, "eve") == {}

    extractor([])
    prompts = concierge_prompts()
    say(g, "Any smart plug with no subscriptions or monthly fees?", "t3", user="eve")
    assert NO_SUBS not in prompts[0] and "(nothing saved yet)" in prompts[0]


def test_forget_by_description_only_deletes_a_confident_match():
    g, store = new_graph()
    ns = ("memories", "fay")
    store.put(ns, "keep", {"text": "Owns two cats named Pixel and Dot.", "kind": "household", "created_at": "t"})
    store.put(ns, "gone", {"text": NO_SUBS, "kind": "preference", "created_at": "t"})

    extractor([op("forget", text="dislikes subscriptions and monthly fees")], [op("forget", text="quantum chromodynamics")])
    first = say(g, "Forget my subscription preference.", "t1", user="fay")
    assert first["memory_changes"][0]["status"] == "forgotten" and "matched by description" in first["memory_changes"][0]["reason"]
    assert set(items(store, "fay")) == {"keep"}

    second = say(g, "Forget the thing about physics.", "t2", user="fay")
    assert second["memory_changes"][0]["status"] == "not_found"
    assert set(items(store, "fay")) == {"keep"}, "no confident match -> nothing is deleted"


def test_update_replaces_the_fact_instead_of_duplicating_it():
    g, store = new_graph()
    extractor([op(text="Has a budget of 100 dollars.", kind="constraint")])
    say(g, "My budget is 100 dollars.", "t1", user="gus")
    ((key, before),) = items(store, "gus").items()

    extractor_fn(lambda prompt: [op("update", key=KEY.search(prompt).group(1), text="Has a budget of 250 dollars.", kind="constraint")])
    out = say(g, "Actually my budget went up to 250 dollars.", "t2", user="gus")
    after = items(store, "gus")
    assert set(after) == {key} and len(after) == 1, "one memory, same key"
    assert after[key]["text"] == "Has a budget of 250 dollars."
    assert after[key]["created_at"] == before["created_at"] and after[key]["updated_at"]
    assert out["memory_changes"][0]["status"] == "updated" and out["memory_changes"][0]["previous"] == before["text"]

    # the extraction model was shown the existing memory and its key, which is what makes updating possible
    extractor([])
    seen = extractor([])
    say(g, "Hello again", "t3", user="gus")
    assert f"[{key}]" in seen[0] and "budget of 250" in seen[0]


def test_update_of_an_unknown_key_becomes_an_add_and_duplicates_are_skipped():
    store = make_store()
    ns = ("memories", "hal")
    out = ex.apply_ops(store, ns, [op("update", key="deadbeef", text=HOUSE, kind="household")])
    assert out[0]["status"] == "stored" and "unknown key" in out[0]["reason"]
    out = ex.apply_ops(store, ns, [op("add", text="  lives in a THREE-storey house, with wifi dead zones ", kind="household")])
    assert out[0]["status"] == "skipped" and out[0]["reason"] == "already remembered"
    assert len(items(store, "hal")) == 1


def test_limits_are_enforced_in_code():
    store = make_store()
    ns = ("memories", "ivy")
    too_long = "Likes " + "very " * 60 + "much."
    out = ex.apply_ops(store, ns, [op(text=too_long), op(text=""), op("add", text=None)])
    assert [c["status"] for c in out] == ["refused", "skipped", "skipped"]

    many = [op(text=f"Owns smart bulb number {n}.", kind="device") for n in range(ex.MAX_OPS_PER_TURN + 3)]
    out = ex.apply_ops(store, ns, many)
    assert len(out) == ex.MAX_OPS_PER_TURN == len(items(store, "ivy")), "extra operations are dropped"

    for n in range(ex.MAX_MEMORIES):
        store.put(ns, f"fill{n}", {"text": f"Filler fact {n}", "kind": "other", "created_at": "t"})
    full = ex.apply_ops(store, ns, [op(text="One more fact.", kind="other")])
    assert full[0]["status"] == "refused" and "memory is full" in full[0]["reason"]


# --------------------------------------------------------------------------- privacy
def test_card_number_is_never_stored_even_if_the_extractor_tries_and_never_reaches_it():
    g, store = new_graph()
    seen = extractor([op(text="Customer's card number is 4242 4242 4242 4242", kind="other"), op(text="Prefers Matter-compatible gear.", kind="preference")])
    out = say(g, "My card number is 4242 4242 4242 4242 and I prefer Matter-compatible gear.", "t1", user="jo")

    statuses = {c["text"]: c["status"] for c in out["memory_changes"]}
    assert statuses == {"[withheld]": "refused", "Prefers Matter-compatible gear.": "stored"}
    assert [v["text"] for v in items(store, "jo").values()] == ["Prefers Matter-compatible gear."]
    assert "4242" not in json.dumps(out["memory_changes"]), "refused text is never echoed back"
    assert "4242" not in seen[0] and "[REDACTED]" in seen[0], "the extraction model never sees the number"
    assert "payment card number" in next(c for c in out["memory_changes"] if c["status"] == "refused")["reason"]


@pytest.mark.parametrize(
    "text, label",
    [
        ("my card number is 4242 4242 4242 4242", "payment card number"),
        ("card 4242-4242-4242-4242", "payment card number"),
        ("4242424242424242", "payment card number"),
        ("my password is hunter2", "credential"),
        ("wifi passcode: sunshine99", "credential"),
        ("the PIN is 4321", "credential"),
        ("cvv 123", "credential"),
        ("API key sk-abc123", "credential"),
        ("SSN 123-45-6789", "government ID"),
        ("national insurance number QQ 12 34 56 C", "government ID"),
        ("passport number 123456789", "government ID"),
        ("IBAN GB82 WEST 1234 5698 7654 32", "bank details|payment card number"),  # either label: what matters is the refusal
        ("email me at jo.smith@example.com", "e-mail address"),
        ("call me on +44 20 7946 0958", "phone number"),
        ("lives at 12 Oak Street", "street address"),
        ("221B Baker Street, London", "street address"),
        ("my address is 5 Elm Drive", "street address"),
        ("delivery address is the back gate", "street address"),
        ("postcode SW1A 1AA", "postcode"),
        ("Dislikes password resets", "credential"),  # a documented false positive: the filter errs toward refusing
    ],
)
def test_privacy_filter_refuses_sensitive_text(text, label):
    found = ex.privacy_violation(text)
    assert found and re.search(label, found), found
    assert "[REDACTED]" in ex.redact(text)


@pytest.mark.parametrize(
    "text",
    [
        HOUSE,
        NO_SUBS,
        "Prefers Matter-compatible gear.",
        "Budget is under 200 dollars.",
        "Moved house on 2026-11-01.",
        "Has 3 cameras on the drive and 2 plugs near the road.",
        "Has 2 smart plugs in the way of the sofa.",
        "Wants Wi-Fi 6E and 4K video at 60 fps.",
        "Rents a 2 bedroom flat, so no drilling.",
        "Dislikes passwordless setups that need a phone.",
        "Owns a Lumen Hub and a Lumen Lock.",
    ],
)
def test_privacy_filter_lets_ordinary_facts_through(text):
    assert ex.privacy_violation(text) is None
    assert ex.redact(text) == text


def test_redaction_replaces_only_the_sensitive_span():
    assert ex.redact("I live in a flat. My card is 4242 4242 4242 4242, thanks!") == "I live in a flat. My card is [REDACTED], thanks!"


# --------------------------------------------------------------------------- streaming and offline behaviour
def test_reply_tokens_stream_from_respond_only():
    g, _ = new_graph()
    extractor([])
    concierge_prompts("Here are three mesh routers that suit a tall house.")
    cfg = {"configurable": {"thread_id": "s1"}}
    chunks = list(g.stream({"messages": [{"role": "user", "content": "Which mesh router?"}]}, cfg, stream_mode="messages"))
    assert len(chunks) > 3, "the reply arrives as several token chunks"
    assert {meta["langgraph_node"] for _, meta in chunks} == {"respond"}, "memory extraction must not leak into the token stream"
    assert "".join(c.content for c, _ in chunks).strip() == "Here are three mesh routers that suit a tall house."


def test_nodes_run_load_case_then_recall_then_respond_then_remember():
    g, _ = new_graph()
    extractor([])
    updates = list(g.stream({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "s2"}}, stream_mode="updates"))
    assert [next(iter(u)) for u in updates] == ["load_case", "recall", "respond", "remember"]


def test_offline_placeholder_operations_never_pollute_the_store():
    g, store = new_graph()  # no handlers: the stand-in model emits "(offline stub)" operations
    out = say(g, "I prefer quiet fans.", "t1", user="kim")
    assert out["memory_changes"] and all(c["status"] == "ignored" for c in out["memory_changes"])
    assert items(store, "kim") == {}


def test_missing_store_is_a_clear_error():
    with pytest.raises(RuntimeError, match="needs a store"):
        ex.graph.invoke({"messages": [{"role": "user", "content": "hi"}]})


# --------------------------------------------------------------------------- ready-made cases (the Studio dropdown)
def case(prefix: str) -> str:
    return next(k for k in ex.CASES if k.startswith(prefix))


def test_a_blank_form_runs_the_first_case_but_never_injects_a_message_into_a_thread_with_history():
    g, _ = new_graph()
    extractor([], [])
    cfg = {"configurable": {"thread_id": "blank"}}
    first = g.invoke({}, cfg)
    assert [m.type for m in first["messages"]] == ["human", "ai"] and first["messages"][0].content == ex.EXAMPLE_INPUT["messages"][0]["content"]
    again = g.invoke({}, cfg)  # the thread already has a chat: nothing new may be typed on the user's behalf
    assert [m.type for m in again["messages"]].count("human") == 1
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        g.invoke({"scenario": "not a case"}, cfg)


def test_the_recall_case_brings_its_own_memories_so_it_needs_no_earlier_run():
    g, store = new_graph()  # a brand-new, empty store: nothing was saved by any other case
    extractor([])
    prompts = concierge_prompts()
    out = g.invoke({"scenario": case("Customer 'ria'")}, {"configurable": {"thread_id": "t"}})
    wanted = {n["text"] for n in ex.CASES[case("Customer 'ria'")]["saved_notes"]}
    assert {m["text"] for m in out["recalled"]} == wanted and len(wanted) == 2 and out["user_id"] == "ria"
    assert all(t in prompts[0] for t in wanted), "the saved notes reached the model"
    assert {v["text"] for v in items(store, "ria").values()} == wanted
    assert items(store) == {}, "nothing leaked into demo_user"


def test_a_customer_without_notes_recalls_nothing_even_when_another_case_saved_notes():
    g, store = new_graph()
    extractor([], [])
    g.invoke({"scenario": case("Customer 'ria'")}, {"configurable": {"thread_id": "a"}})
    out = g.invoke({"scenario": case("Customer 'zed'")}, {"configurable": {"thread_id": "b"}})
    assert out["recalled"] == [] and out["user_id"] == "zed" and items(store, "zed") == {}


def test_repeating_the_forget_case_restores_its_notes_and_a_follow_up_turn_does_not_resurrect_them():
    g, store = new_graph()
    label, cfg = case("Customer 'ben'"), {"configurable": {"thread_id": "ben"}}
    notes = {n["text"] for n in ex.CASES[label]["saved_notes"]}
    # the extractor sees "[key] (kind) text" lines; forget the one about subscriptions, as a good model would
    extractor_fn(lambda prompt: [op("forget", key=next(m.group(1) for m in re.finditer(r"\[([0-9a-f]{8})\] \(\w+\) Dislikes subscriptions", prompt)))])
    out = g.invoke({"scenario": label}, cfg)
    assert [c["status"] for c in out["memory_changes"]] == ["forgotten"]
    assert {v["text"] for v in items(store, "ben").values()} == notes - {"Dislikes subscriptions and monthly fees."}

    extractor([])
    g.invoke({"messages": [{"role": "user", "content": "What do you remember about me?"}]}, cfg)
    assert "Dislikes subscriptions and monthly fees." not in {v["text"] for v in items(store, "ben").values()}, "notes are written once, not on every turn"
    assert g.get_state(cfg).values["saved_notes"] is None and g.get_state(cfg).values["as_user"] == "ben", "ben stays the customer of this thread"

    extractor([])
    g.invoke({"scenario": label}, {"configurable": {"thread_id": "ben-again"}})  # picking the case again restores the demo
    assert {v["text"] for v in items(store, "ben").values()} == notes
    assert len(items(store, "ben")) == 2, "same keys: no copies"


def test_the_card_number_case_never_stores_the_number_even_if_the_extractor_proposes_it():
    g, store = new_graph()
    label = case("Card number")
    seen = extractor([op(text="Customer's card number is 4242 4242 4242 4242", kind="other"), op(text="Prefers Matter-compatible gear.", kind="preference")])
    out = g.invoke({"scenario": label}, {"configurable": {"thread_id": "dee"}})
    assert "4242" not in json.dumps(items(store, "dee")) and "4242" not in json.dumps(out["memory_changes"])
    assert "4242" not in seen[0] and "[REDACTED]" in seen[0]
    assert [c["status"] for c in out["memory_changes"]] == ["refused", "stored"]


def test_as_user_beats_the_assistants_user_id_setting_and_saved_notes_obey_the_privacy_filter():
    ctx = lambda c: SimpleNamespace(context=c)  # noqa: E731
    assert ex.resolve_user_id(ctx(ex.Context("ctx")), {"configurable": {"user_id": "cfg"}}, "form") == "form"
    assert ex.resolve_user_id(ctx(ex.Context("ctx")), {}, None) == "ctx" and ex.resolve_user_id(ctx(ex.Context("ctx")), {}, "") == "ctx"

    store = make_store()
    ns = ("memories", "kai")
    n = ex.seed_notes(store, ns, [{"text": "Prefers quiet fans.", "kind": "preference"}, {"text": "My card is 4242 4242 4242 4242"}, "Owns a Lumen Hub.", {"text": ""}, {"text": None}, None])
    assert n == 2 and {v["text"] for v in items(store, "kai").values()} == {"Prefers quiet fans.", "Owns a Lumen Hub."}
    assert ex.seed_notes(store, ns, ["Owns a Lumen Hub."]) == 1 and len(items(store, "kai")) == 2, "idempotent: same key, no copy"


def test_every_case_is_self_contained_and_its_saved_notes_are_ordinary_facts():
    for label, fields in ex.CASES.items():
        notes = fields["saved_notes"] or []
        assert all(ex.privacy_violation(n["text"]) is None and n["kind"] in ex.KINDS for n in notes), label
        assert (fields["as_user"] is None) <= (not notes), f"{label}: saved notes need their own customer, not the shared demo_user"
        assert fields["as_user"] is None or f"'{fields['as_user']}'" in label or label.startswith("Card number"), label
        assert [m["role"] for m in fields["messages"]] == ["user"]

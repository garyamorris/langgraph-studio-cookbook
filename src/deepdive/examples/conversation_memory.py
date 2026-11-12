"""Short-term memory for a long support chat: a bounded window, a rolling summary, and facts that are never forgotten.

PROBLEM
    A support chat can run for 100 turns. Sending the whole history every turn costs more with every message and
    eventually overflows the context window. Cutting it off at the last N messages is cheap but makes the assistant
    forget the very things that matter: the customer's name, which order they are talking about, and what the
    assistant itself promised ("I'll email a return label by Friday"). A support assistant that forgets its own
    commitments is worse than one that never made them.

DESIGN
    Memory is split by how much it hurts to lose, and each kind lives in its own state field, persisted per
    ``thread_id`` by the checkpointer (with a durable one, the same chat can continue tomorrow in another process):

      * ``messages``  the recent verbatim window. Bounded: once it exceeds ``max_messages`` (default 12) the oldest
                      messages are folded away until ``keep_last`` (default 6) remain. Both are state fields you can
                      override in the input, per thread.
      * ``summary``   a rolling summary of everything that was folded away. A chat model rewrites it each time
                      ("previous summary + the messages being removed -> updated summary"); the prompt tells it to
                      keep commitments, open issues, amounts, dates and stated preferences and to drop pleasantries.
                      It is lossy by nature, which is why it is not the only safety net:
      * ``facts``     PINNED memory that is never dropped and never summarised: customer name, order numbers, and
                      the assistant's commitments. Each turn a small structured-output call extracts them from the
                      new messages, and a deterministic regex also catches order numbers ('#1042', 'order 1042')
                      whatever the model returns, because a missed order number is the costliest miss. The ``facts``
                      field has a reducer (``merge_facts``) so the merge rule lives with the state, not in whichever
                      node writes it: lists are unioned and de-duplicated, and an empty value can never overwrite
                      a real one. Lists are capped at 30 entries per kind to keep the prompt bounded; past that the
                      oldest entry is the one that goes.

    Every ``respond`` prompt is built from: the instructions, then the pinned facts, then the summary (if any), then
    the messages still in the window. The window passed to the model is also trimmed to ``max_messages + 1`` as a
    safety net (for example if an input pastes in a 40-message history); the usual state is already bounded by
    folding.

    Order of nodes: ``respond`` first, then ``remember`` (it can see what the assistant just promised), then
    a conditional edge ``maybe_summarize`` that goes to ``summarize`` only when the window is over its limit.
    The reply streams before the summariser runs, so the customer never waits for it. After every turn the stored
    window holds at most ``max_messages`` messages; when the next user message arrives that is at most
    ``max_messages + 1``, which is exactly what ``respond`` sends, so nothing is ever left out of a prompt before it
    has been summarised. The cost is two model calls per turn (the reply, and fact extraction on the cheaper model)
    plus a third, the summary, every ``(max_messages - keep_last) / 2 + 1`` turns (every 4th turn with the defaults).

    Failure rules: if fact extraction raises, the turn still completes and the regex net still pins order numbers; if
    the summariser returns nothing, NO messages are removed (they would be lost) and it simply tries again next turn.

    Streaming: ``respond`` is the only node whose model output is meant for the customer. The extractor and summariser
    calls are tagged ``nostream`` so ``stream_mode="messages"`` shows only the reply tokens (filtering on
    ``metadata["langgraph_node"] == "respond"`` works too). Memory changes are emitted as ``stream_mode="custom"``
    events (``memory_updated``, ``summarized``) so a UI can show "remembered order #1042".

GRAPH
    START -> load_case -> respond -> remember -> maybe_summarize --(window > max_messages)--> summarize -> END
                                                        |
                                                        +--(otherwise)-----------------------------> END

    ``load_case`` only turns a picked ready-made case (the ``scenario`` dropdown) into the chat input; it does nothing
    for a message you type. It is a separate node because the first real node, ``respond``, is already the model call.

LANGGRAPH FEATURES
    * checkpointer-backed thread memory (state persists per ``thread_id``; threads are isolated)
    * ``add_messages`` plus ``RemoveMessage`` returned from a node to delete old messages from state
    * a rolling-summary field and a trimming policy (``trim_messages`` as a prompt-side safety net)
    * pinned structured memory with a custom reducer on a state key
    * a conditional edge that decides whether the summariser runs
    * token streaming (``stream_mode="messages"``) with internal calls hidden, and ``get_stream_writer`` custom events
    * ``input_schema`` / ``output_schema`` with a ``messages`` key so Studio's chat view works

MODELS AND DATA
    OpenAI chat: the reply model (default tier), and the fast tier for fact extraction (structured output) and
    summarising. No dataset, no retrieval: this example is about memory, so the assistant answers from the
    conversation alone and the prompt forbids inventing order data it cannot see. Offline mode uses placeholder
    replies and a stub extractor, so the pinned facts you see offline are stub text; the mechanics (folding, bounds,
    ordering, isolation) are exactly the same and are what the tests check.

TRY IT IN STUDIO
    Pick a case from the **Ready-made case** dropdown and press Submit (a blank form on a new thread runs the first):
      * "Pasted 5-message chat ... max_messages=4, keep_last=2": folds on this very run. Look at ``summary``,
        ``folded_messages`` (4) and ``messages`` (2 left), and at ``facts`` (the name, #1042 and the refund promise).
      * "Priya + order #1042": the plain one-message start (default limits 12 / 6). Keep chatting in the chat view
        ("can you refund it?", then a dozen or so unrelated questions) and watch the first fold happen on your 7th message.
      * "Long message ... three orders": ``facts.order_ids`` holds all three; limits are 4 / 2, so send two more
        short messages (any text) and the summary appears while the order numbers stay pinned.
      * "Order-number spellings": three spellings, all pinned by the regex alone. "Nonsense settings": ``max_messages=1,
        keep_last=50`` are clamped to 4 / 2 and it still folds.
    A picked case ADDS its message(s) to the thread and replaces ``max_messages`` / ``keep_last``; the summary and facts
    of earlier turns stay (that is the point of the example). Start a New Thread for a clean slate.
    To use your own input instead, leave the dropdown empty and fill ``messages`` (or just type in the chat view):
    1. Open the graph in chat mode (or graph mode with a ``messages`` list) and send, as separate turns:
       "Hi, I'm Priya. My order #1042 arrived with a cracked case." / "Can you refund it?" / then ten or so
       unrelated questions ("how do I reset the hub?", "what is the warranty on the lock?", ...).
    2. To see folding fast, use graph mode and put ``"max_messages": 6, "keep_last": 2`` in the first input.
       Watch the state panel after each turn: ``messages`` stops growing, ``summary`` appears and is rewritten,
       ``folded_messages`` climbs, and ``facts`` keeps the name, the order number and the assistant's promise.
    3. Ask "which order were we talking about?" long after the first message is gone. The reply is possible because
       the order number is in ``facts``, not in the window. If LangSmith tracing is on, the trace of ``respond`` shows the
       exact prompt (instructions, pinned facts, summary, last messages) the model was given; the tests capture it too.
    4. Start a second thread: it has its own messages, summary and facts.

EXTEND IT
    Move ``summarize`` into a background run so the reply never shares a run with it. Count tokens instead of
    messages (``trim_messages(token_counter=model)``). Promote the pinned facts to the LangGraph Store so they follow
    the customer across threads: that is long-term memory, a different mechanism from this per-thread state.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="conversation_memory",
    title="Short-term memory: window, rolling summary and pinned facts",
    category="Memory & streaming",
    problem="Keep a 100-turn support chat cheap and within context without forgetting the customer's name, order "
    "numbers or what the assistant promised, by combining a bounded window, a rolling summary and pinned facts.",
    features=[
        "checkpointer thread memory",
        "add_messages + RemoveMessage",
        "rolling summary",
        "trim_messages window",
        "pinned facts with a reducer",
        "conditional edge",
        "token streaming + custom events",
    ],
    models=["OpenAI chat"],
    datasets=[],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"messages": [{"role": "user", "content": "Hi, I'm Priya. My order #1042 arrived with a cracked case. Can you help?"}]}

# Ready-made cases for Studio's dropdown (see deepdive.common.scenarios). A chat case is its message(s) plus the two
# limits, and every case sets all three keys (None = the default) so a pick replaces the previous limits. The first
# case is the best first-time demo: a pasted chat that is already over its limit, so the summary appears on the first
# Submit instead of after a dozen turns. Order numbers 1042 / 1828 / 1838 / 1767 are real orders in store.sqlite, but
# this example never reads the database: they are only text in the chat.
CASES: dict[str, dict] = {
    "Pasted 5-message chat, max_messages=4 keep_last=2 -> folds 4 into a summary on this run, #1042 stays pinned": {
        "messages": [
            {"role": "user", "content": "Hi, I'm Priya. My order #1042 arrived with a cracked case."},
            {"role": "assistant", "content": "Sorry about that, Priya. I will submit a refund request for order #1042 and you will hear back within 2 business days."},
            {"role": "user", "content": "Thanks. Separately, the light on my Hub has been solid red since this morning."},
            {"role": "assistant", "content": "A solid red Hub light means no network. Power cycle the router, then the Hub, and wait two minutes."},
            {"role": "user", "content": "Did that and it is still red. By the way, which order was my refund request for?"},
        ],
        "max_messages": 4, "keep_last": 2},
    "Priya + order #1042 in one message, default limits -> facts pin #1042 (regex), no summary yet": {
        "messages": list(EXAMPLE_INPUT["messages"]), "max_messages": None, "keep_last": None},
    "Long message: Marcus, 3 orders (1042, 1828, 1838), limits 4/2 -> all 3 pinned; send 2 more to fold": {
        "messages": [{"role": "user", "content": (
            "Hi, I'm Marcus. I have three problems and I would like them all written down. First, order #1042: one of the two "
            "Lumen Cam Pro cameras arrived with a cracked case. Second, order 1828: the Lumen Plug 4-pack never connected to "
            "wifi. Third, order No. 1838: the Lumen Lock keeps saying low battery after a week. Please contact me by phone, "
            "not by email, and I need an answer before Friday because I am moving house.")}],
        "max_messages": 4, "keep_last": 2},
    "Order-number spellings: 'order 1828', 'Order No. 1677', '#1767' -> the regex pins all three": {
        "messages": [{"role": "user", "content": (
            "Three parcels, one question: order 1828, Order No. 1677 and #1767 all arrived this month. Which of them can I "
            "still return?")}],
        "max_messages": None, "keep_last": None},
    "Nonsense settings (max_messages=1, keep_last=50) on a pasted chat -> clamped to 4 / 2, still folds 4": {
        "messages": [
            {"role": "user", "content": "Hi, I'm Sam. My order #1767 has not been delivered yet."},
            {"role": "assistant", "content": "Sorry to hear that, Sam. I will escalate order #1767 to the shipping team and you will hear back by tomorrow."},
            {"role": "user", "content": "Ok. Can you also tell me how long the warranty is on the Lumen Lock?"},
            {"role": "assistant", "content": "The Lumen Lock has a 24-month warranty."},
            {"role": "user", "content": "Thanks. Remind me, what did you promise about my order?"},
        ],
        "max_messages": 1, "keep_last": 50},
}
SCENARIO = scenario_field(CASES)

DEFAULT_MAX_MESSAGES = 12  # fold once the stored window is longer than this ...
DEFAULT_KEEP_LAST = 6  # ... down to this many verbatim messages
SUMMARY_MAX_CHARS = 2400  # hard guard on a runaway summary (the pinned facts are the safety net, not the summary)
MAX_PINNED = 30  # per list in ``facts``
NO_STREAM = ["nostream"]  # tag that keeps a model call out of stream_mode="messages"

# 'order #1042', 'order number 1042', 'Order No. 1042', '#1042' - three to eight digits
ORDER_RE = re.compile(r"(?:#\s?|\border(?:\s+(?:number|no\.?|id))?\s*:?\s*#?\s*)(\d{3,8})\b", re.IGNORECASE)


# --------------------------------------------------------------------------- pinned facts
def extract_order_ids(text: str) -> list[str]:
    """Deterministic safety net: every order number mentioned in ``text``, normalised to '#1042', in order of appearance."""
    out: list[str] = []
    for m in ORDER_RE.finditer(text):
        oid = f"#{m.group(1)}"
        if oid not in out:
            out.append(oid)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def merge_facts(old: dict | None, new: dict | None) -> dict:
    """Reducer for the pinned ``facts``: union the lists (de-duplicated, capped), replace the name only by a non-empty one."""
    merged = {
        "customer_name": str((old or {}).get("customer_name") or ""),
        "order_ids": list((old or {}).get("order_ids") or []),
        "commitments": list((old or {}).get("commitments") or []),
    }
    new = new or {}
    name = str(new.get("customer_name") or "").strip()
    if name:
        merged["customer_name"] = name
    for key in ("order_ids", "commitments"):
        seen = {_norm(x) for x in merged[key]}
        for item in new.get(key) or []:
            item = str(item).strip()
            if item and _norm(item) not in seen:
                merged[key].append(item)
                seen.add(_norm(item))
        merged[key] = merged[key][-MAX_PINNED:]
    return merged


class FactUpdate(BaseModel):
    """What the extractor may pin from the newest messages."""

    customer_name: str | None = Field(description="The name the customer gave for themselves, or null if none was stated.")
    order_ids: list[str] = Field(description="Every order number mentioned, written like #1042. Empty if none.")
    commitments: list[str] = Field(
        description="Things the ASSISTANT promised to do or provide (refund, replacement, callback, deadline), one short "
        "sentence each, including order number, amount and date when stated. Never suggestions, questions or customer actions."
    )


# --------------------------------------------------------------------------- state
class ChatIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made case; adds its message(s) and replaces the two limits below
    messages: Annotated[list[AnyMessage], add_messages]
    max_messages: int  # fold when more than this many messages are stored (default 12)
    keep_last: int  # ... leaving this many (default 6)


class ChatOut(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    facts: dict
    turn: int
    folded_messages: int
    mode: str


class ChatState(ChatIn, total=False):
    summary: str
    facts: Annotated[dict, merge_facts]
    facts_cursor: str  # id of the last message the extractor has seen
    turn: int
    folded_messages: int
    mode: str


def limits(state: ChatState) -> tuple[int, int]:
    """(max_messages, keep_last) with sane clamps, so a bad input cannot disable folding or fold everything."""
    mx = max(4, int(state.get("max_messages") or DEFAULT_MAX_MESSAGES))
    kl = max(2, min(int(state.get("keep_last") or DEFAULT_KEEP_LAST), mx - 2))
    return mx, kl


# --------------------------------------------------------------------------- prompts
RESPOND_PROMPT = (
    "You are Lumen Home's customer-support assistant for smart-home devices. Reply in under 90 words, friendly and "
    "concrete. You cannot see order or account data in this chat, so never invent order status; say what you can do "
    "(explain policy, submit a request to the team, escalate to a human) and what happens next. Only commit to things "
    "within support's power, and state any commitment as one clear sentence with the order number and timeframe so it "
    "can be remembered, for example 'I will submit a refund request for order #<number> and you will hear back "
    "within 2 business days.' Use the pinned facts and the conversation summary below as ground truth about earlier parts of "
    "the chat, and never ask the customer for something they are already recorded as having told you."
)

SUMMARY_PROMPT = (
    "You maintain the running memory of a customer-support chat between a customer and the Lumen Home assistant. "
    "You are given the PREVIOUS SUMMARY and the OLDEST MESSAGES, which are about to be removed from the chat. Write "
    "the UPDATED SUMMARY that replaces both. Rules: (1) keep what the customer wants, plus every order number, product, "
    "amount, date and deadline; (2) keep every COMMITMENT the assistant made - who promised what, by when - in "
    "its own words; (3) keep OPEN ISSUES: anything asked or reported and not yet resolved, and note briefly what was "
    "resolved; (4) keep stated customer preferences and how the customer feels; (5) drop greetings, thanks, repetition "
    "and troubleshooting steps beyond one phrase; (6) never invent or speculate. At most 160 words, in third person, "
    "plain sentences or short bullets."
)

EXTRACT_PROMPT = (
    "You maintain a support chat's pinned memory. From the NEW MESSAGES extract only what is explicitly stated: "
    "customer_name (the name the customer gave for themselves; null if not stated), order_ids (every order number "
    "mentioned, written like #1042), and commitments (things the ASSISTANT promised to do or provide, such as a refund "
    "request, replacement, callback or deadline: one short sentence each with the order number, amount and date if "
    "stated). Never include suggestions, questions, things the customer will do, or anything you are guessing. Return "
    "empty lists when there is nothing."
)


def pinned_block(facts: dict | None) -> str:
    facts = facts or {}
    lines = ["PINNED FACTS (recorded earlier in this conversation; do not ask the customer for them again)"]
    lines.append(f"- Customer name: {facts.get('customer_name') or '(not given yet)'}")
    lines.append(f"- Order numbers: {', '.join(facts.get('order_ids') or []) or '(none mentioned yet)'}")
    commitments = facts.get("commitments") or []
    if commitments:
        lines.append("- Commitments you (the assistant) have made:")
        lines += [f"  {i}. {c}" for i, c in enumerate(commitments, 1)]
    else:
        lines.append("- Commitments you (the assistant) have made: (none yet)")
    return "\n".join(lines)


def build_system(state: ChatState) -> str:
    parts = [RESPOND_PROMPT, pinned_block(state.get("facts"))]
    if state.get("summary"):
        parts.append("SUMMARY OF EARLIER MESSAGES (no longer shown verbatim)\n" + state["summary"])
    return "\n\n".join(parts)


def window(state: ChatState) -> list[BaseMessage]:
    """The verbatim messages sent to the model: normally all stored ones, never more than max_messages + 1."""
    mx, _ = limits(state)
    messages = state["messages"]
    kept = trim_messages(messages, max_tokens=mx + 1, token_counter=len, strategy="last", start_on="human", include_system=False)
    return kept or messages[-1:]


def choose_fold(messages: list[BaseMessage], keep_last: int) -> list[BaseMessage]:
    """The oldest messages to fold away, leaving ``keep_last`` (moved forward so the kept window starts on a human message)."""
    cut = len(messages) - keep_last
    if cut <= 0:
        return []
    adjusted = cut
    while adjusted < len(messages) and not isinstance(messages[adjusted], HumanMessage):
        adjusted += 1
    return messages[: adjusted if adjusted < len(messages) else cut]


def transcript(messages: list[BaseMessage]) -> str:
    return "\n".join(f"{'Customer' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text}" for m in messages)


# --------------------------------------------------------------------------- nodes
def load_case(state: ChatState) -> dict:
    """Turn a picked ready-made case into the chat input; leave a message you typed alone.

    ``messages`` has the ``add_messages`` reducer, so the case's messages are APPENDED to whatever the thread already
    holds. That is why "real input" here means "the state already has messages": a blank form on a thread with history
    must not inject a new opening message, while a picked case always does.
    """
    return expand(state, CASES, has_input=lambda s: bool(s.get("messages")))


def respond(state: ChatState) -> dict:
    """Reply from: instructions + pinned facts + summary + the window. This is the only node the customer sees."""
    prompt = [SystemMessage(build_system(state)), *window(state)]
    reply = get_llm().invoke(prompt)
    return {"messages": [reply], "turn": state.get("turn", 0) + 1, "mode": mode_label()}


def remember(state: ChatState) -> dict:
    """Pin name, order numbers and the assistant's commitments from the messages added since the last look."""
    messages = state["messages"]
    ids = [m.id for m in messages]
    start = ids.index(state["facts_cursor"]) + 1 if state.get("facts_cursor") in ids else 0
    new = messages[start:]
    if not new:
        return {}
    text = transcript(new)
    update: FactUpdate | None = None
    try:
        update = structured_llm(FactUpdate, "fast").with_config(tags=NO_STREAM).invoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(f"NEW MESSAGES\n{text}")])
    except Exception:  # noqa: BLE001 - memory extraction must never break the conversation; the regex net below still runs
        update = None
    model_ids = [f"#{d}" for oid in (update.order_ids if update else []) for d in re.findall(r"\d{3,8}", oid)]
    fresh = {
        "customer_name": (update.customer_name or "") if update else "",
        "order_ids": model_ids + extract_order_ids(text),
        "commitments": list(update.commitments) if update else [],
    }
    before = merge_facts(state.get("facts"), None)
    added = {k: [x for x in fresh[k] if _norm(x) not in {_norm(y) for y in before[k]}] for k in ("order_ids", "commitments")}
    if fresh["customer_name"] and fresh["customer_name"] != before["customer_name"]:
        added["customer_name"] = fresh["customer_name"]
    if any(added.values()):
        get_stream_writer()({"event": "memory_updated", "added": added})
    return {"facts": fresh, "facts_cursor": messages[-1].id}


def maybe_summarize(state: ChatState) -> Literal["summarize", "__end__"]:
    """Conditional edge: only pay for a summary when the window is over its limit."""
    mx, _ = limits(state)
    return "summarize" if len(state["messages"]) > mx else END


def summarize(state: ChatState) -> dict:
    """Fold the oldest messages into ``summary`` and remove them from state with ``RemoveMessage``."""
    _, keep_last = limits(state)
    old = choose_fold(state["messages"], keep_last)
    if not old:
        return {}
    previous = state.get("summary") or "(none yet)"
    reply = (
        get_llm("fast")
        .with_config(tags=NO_STREAM)
        .invoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(f"PREVIOUS SUMMARY\n{previous}\n\nOLDEST MESSAGES\n{transcript(old)}")])
    )
    text = str(reply.text).strip()
    if not text:  # never delete messages we failed to summarise: try again next turn
        return {}
    get_stream_writer()({"event": "summarized", "folded": len(old), "kept": len(state["messages"]) - len(old)})
    return {
        "summary": text[:SUMMARY_MAX_CHARS],
        "messages": [RemoveMessage(id=m.id) for m in old],
        "folded_messages": state.get("folded_messages", 0) + len(old),
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(ChatState, input_schema=ChatIn, output_schema=ChatOut)
    g.add_node("load_case", load_case)
    g.add_node("respond", respond)
    g.add_node("remember", remember)
    g.add_node("summarize", summarize)
    g.add_edge(START, "load_case")
    g.add_edge("load_case", "respond")
    g.add_edge("respond", "remember")
    g.add_conditional_edges("remember", maybe_summarize, ["summarize", END])
    g.add_edge("summarize", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

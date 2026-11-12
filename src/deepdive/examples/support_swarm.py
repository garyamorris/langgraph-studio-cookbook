"""Support swarm: peer agents that hand a live conversation to each other.

PROBLEM
    In a customer chat the right specialist changes mid-conversation. It starts as a billing question ("can I return
    the plug from order 1828?") and turns into a device fault ("also, my camera keeps going offline"). A router that
    decides once, at the start, keeps the customer with the wrong specialist; sending them back to the front desk makes
    them repeat themselves. Customers hate both.

DESIGN
    A *swarm*: three peer agents - ``triage``, ``billing``, ``technical`` - and no central supervisor. Whoever holds
    the conversation decides when to pass it on, by calling a transfer tool. Each agent is a small ReAct loop
    (model <-> tools) that is given ONLY its own tools, so it cannot answer outside its remit:

      * triage    transfer_to_billing, transfer_to_technical. It never solves anything.
      * billing   lookup_order, check_refund_eligibility, transfer_to_technical. Order facts come from tools reading
                  ``store.sqlite``; the 30-day return rule is plain code, not model judgement, and is evaluated against a
                  fixed ``AS_OF`` date so every run gives the same answer. It cannot issue refunds (no write tool).
      * technical search_kb (public BM25 support articles, with ids to cite), transfer_to_billing.

    The conversation lives in one shared ``messages`` list, so a specialist sees everything said so far. The parent
    graph also keeps ``active_agent`` in state and routes ``START`` to it, so the *next* user message on the same
    thread resumes with whoever last held the conversation - not with triage.

    Things a swarm gets wrong if you do not design for them, and what is done here:
      * Message history must stay valid for OpenAI: a tool call needs a matching tool message right after it. A transfer
        leaves the agent's subgraph and the subgraph's own pending messages are dropped (checked: returning only the
        ``ToolMessage`` leaves an orphaned tool message in the parent), so the transfer tool returns the whole history plus
        its ``ToolMessage`` (``add_messages`` de-duplicates by id). Models are bound with ``parallel_tool_calls=False`` and any sibling tool
        call in the same message is answered with "not executed", so nothing is ever orphaned.
      * Ping-pong: with no supervisor, nothing stops billing -> technical -> billing -> ... forever. Transfers are capped
        per user turn (``MAX_HANDOFFS_PER_TURN``); beyond it the tool refuses and the agent must answer. The count is
        derived from the message history, so it needs no counter to reset.
      * Runaway tool loops: an agent that keeps calling tools is cut off after ``MAX_TOOL_ROUNDS`` rounds with a fixed
        "a human will follow up" reply (no model call).

GRAPH
    START -> load_case --(route on active_agent)--> triage | billing | technical --> END
              each agent is a subgraph:  agent <-> tools   (ToolNode)
              a transfer tool returns Command(goto=<peer>, graph=Command.PARENT) and jumps across to the peer

    ``load_case`` only turns a picked ready-made case (the ``scenario`` dropdown) into the chat input; it does nothing
    for a message you type. The routing edge reads ``active_agent`` AFTER it, so it can send a picked case to the front desk.

LANGGRAPH FEATURES
    * ``Command(goto=..., graph=Command.PARENT, update=...)`` returned from a tool: peer-to-peer handoff from inside a
      subgraph, updating ``active_agent`` and the shared messages in the same step
    * agents as subgraphs sharing the parent's state keys (``messages``), each with its own tool set
    * ``add_conditional_edges(START, ...)`` reading persisted state, so a thread continues with the right agent;
      needs a checkpointer (Studio threads supply one; tests pass ``InMemorySaver``)
    * ``ToolNode`` with ``InjectedState`` / ``InjectedToolCallId`` so a tool can see the history and answer its own call
    * explicit bounds on handoffs and tool rounds (cheap defences against loops with no supervisor)

MODELS AND DATA
    OpenAI chat with tool calling (default tier, one model call per agent step). Data: ``store.sqlite`` (orders,
    order_items, products - read-only) and the public ``support`` knowledge base (``audience="public"``: the internal
    refund approval matrix is never retrievable here).
    Simulated / not production-grade: there is NO customer authentication - ``lookup_order`` returns any order by id,
    so a real deployment must verify the customer first. Delivery dates are not in the database: delivery is assumed to
    be ``ASSUMED_TRANSIT_DAYS`` (5) days after the order date, and the return window is evaluated on ``AS_OF``
    (2026-11-06: five days after the last order, so every "delivered" order counts as delivered). Nobody can actually escalate to a human: the agents say a human will
    follow up, and a real system would need an ``escalate_to_human`` tool.

TRY IT IN STUDIO
    Pick a case from the **Ready-made case** dropdown and press Submit (a blank form on a new thread runs the first).
    The order ids, products and day counts in the labels come from ``store.sqlite`` and the fixed ``AS_OF`` date:
      * "Order 1828 ... ELIGIBLE", "Order 1677 ... NOT eligible", "Order 1836 ... NOT eligible": triage should hand the
        customer to billing, which calls ``lookup_order`` and ``check_refund_eligibility``. That verdict is plain code,
        so it is the same on every run (inside the window; 62 days, outside; already refunded).
      * "Cam Pro offline": triage should hand to technical, which searches the knowledge base and cites article ids.
      * "Return from order 1828 AND Cam Pro offline": one message, two specialists. Open ``handoffs`` and
        ``active_agent`` in the state: triage -> billing -> technical, inside the cap of 3 transfers per message.
      * "Order 1828 plug 'extremely hot', burning smell": technical should tell the customer to unplug and promise a human
        follow-up. That is the prompt's job only (this example has no safety lexicon; compare ``ticket_router``).
    Which agent and which tools the MODEL picks is not guaranteed, so the labels say "watch". A picked case is a new
    contact: it starts at triage even on a thread that already has a chat, and its message is added to that chat. Then
    type a follow-up in the chat view (or put it in ``messages``) to see the swarm resume with whoever held the thread
    last, for example after the first case::

        {"messages": [{"role": "user", "content": "Thanks. Separately my Cam Pro keeps dropping offline after the update."}]}

    The run starts at ``billing`` (not triage), which transfers to ``technical``; technical searches the knowledge base
    and cites article ids. A third message about the refund returns to billing. Other orders to ask about: 1767
    (delivered exactly 30 days ago: still eligible), 1840 (cancelled). To type a first message yourself, leave the
    dropdown empty and send::

        {"messages": [{"role": "user", "content": "Hi, can I still return the Lumen Plug 4-pack from order 1828?"}]}

EXTEND IT
    Add an ``escalate_to_human`` tool that ``interrupt()``s for an agent to take over. Add identity verification before
    ``lookup_order``. Give each handoff a structured summary (order id, device, what was tried) instead of free text.
"""

from __future__ import annotations

import json
import operator
from datetime import date, timedelta
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import db_connect
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm
from deepdive.common.retrieval import get_retriever
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="support_swarm",
    title="Support swarm (peer handoffs)",
    category="Multi-agent",
    problem="Let a live support conversation move between billing, technical and front-desk agents without making the "
    "customer repeat themselves, resuming with whoever last held it on the next message.",
    features=[
        "Command.PARENT handoffs from tools",
        "agents as subgraphs",
        "persistent active_agent routing from START",
        "scoped tool sets",
        "handoff and tool-round bounds",
    ],
    models=["OpenAI chat"],
    datasets=["store.sqlite", "kb/support"],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"messages": [{"role": "user", "content": "Hi, can I still return the Lumen Plug 4-pack from order 1828?"}]}

# Ready-made cases for Studio's dropdown (see deepdive.common.scenarios): one first message from a customer each. The orders
# are real rows of store.sqlite; "days since delivery" is evaluated on AS_OF with the assumed 5-day transit (the policy
# verdict is code, see refund_eligibility). Which specialist the MODEL hands the customer to is never promised ("watch").
CASES: dict[str, dict] = {
    "Order 1828 (Plug 4-pack, 7 days since delivery): return it? -> watch triage > billing; policy check: ELIGIBLE": {
        "messages": list(EXAMPLE_INPUT["messages"])},
    "Cam Pro drops off wifi, error E-102, no order given (T-2084's words) -> watch triage > technical and a KB citation": {
        "messages": [{"role": "user", "content": "My Lumen Cam Pro drops off wifi every few hours. I rebooted the router twice and the app shows error E-102."}]},
    "Return from order 1828 AND Cam Pro offline in ONE message -> watch handoffs: triage > billing > technical": {
        "messages": [{"role": "user", "content": "Can I return the Lumen Plug 4-pack from order 1828? Also, separately, my Lumen Cam Pro keeps dropping off wifi, error E-102."}]},
    "Order 1677 (Bell x3 + Plug, 62 days since delivery): 'changed my mind' -> watch billing; policy check: NOT eligible": {
        "messages": [{"role": "user", "content": "Hello, I would like to send back the Lumen Bell and the plug from order 1677. I changed my mind. Is that still possible?"}]},
    "Order 1836 (Therm) is already refunded; customer asks again -> watch billing; policy check: NOT eligible": {
        "messages": [{"role": "user", "content": "Hi, please refund my Lumen Therm from order 1836."}]},
    "Order 1828 plug 'extremely hot', burning smell (T-2301 words) -> watch: unplug + human follow-up (prompt only)": {
        "messages": [{"role": "user", "content": "One of the plugs in my Lumen Plug 4-pack (order 1828) got extremely hot and there's a burning plastic smell. I've unplugged it. This is a fire risk!!"}]},
}
SCENARIO = scenario_field(CASES)

AGENTS = ("triage", "billing", "technical")
AS_OF = date(2026, 11, 6)  # the fixed "today" (last order 2026-10-31 + 5 days): results never depend on the real clock
RETURN_WINDOW_DAYS = 30  # data/kb/support/returns-and-refunds.md: "within 30 days of delivery"
ASSUMED_TRANSIT_DAYS = 5  # the database has no delivery date; shipping-and-delivery.md says 3-6 business days
WARRANTY_DAYS = {"default": 365, "Lumen Lock": 730, "Lumen Therm": 730}  # warranty-and-repairs.md: 12 months; locks/thermostats 24
MAX_HANDOFFS_PER_TURN = 3
MAX_TOOL_ROUNDS = 6
TRANSFER_OK = "Transferred to "
FALLBACK_REPLY = (
    "I'm sorry - I wasn't able to finish this myself. I've kept everything you told me, and a human agent will "
    "follow up so you don't need to repeat it."
)


# --------------------------------------------------------------------------- state
class AgentState(MessagesState):
    """What an agent subgraph sees. ``messages`` comes from MessagesState (add_messages reducer, idempotent by id)."""

    active_agent: str  # who holds the conversation; START routes here on the next user message
    mode: str  # "live" or "offline": which kind of model produced the replies


class SwarmIn(TypedDict, total=False):
    """What Studio's form (and chat view) sends: the chat message(s), or a ready-made case. Nothing is required."""

    scenario: SCENARIO  # a ready-made case; adds its first message and starts the conversation at triage
    messages: Annotated[list[AnyMessage], add_messages]


class SwarmState(AgentState):
    """Parent state = the shared keys plus the audit log.

    ``handoffs`` is deliberately NOT part of ``AgentState``: when a subgraph that shares a key finishes, it writes its
    whole state back to the parent, and an append reducer (``operator.add``) would then append the log to itself.
    Only the transfer tool writes it, straight to the parent via ``Command.PARENT``.
    """

    handoffs: Annotated[list[dict], operator.add]  # {"from", "to", "reason"} per transfer
    scenario: SCENARIO | None  # the picked ready-made case; load_case clears it once used


class SwarmOut(TypedDict, total=False):
    messages: list[AnyMessage]
    active_agent: str
    handoffs: list[dict]
    mode: str


# --------------------------------------------------------------------------- deterministic business logic
def _fetch_order(order_id: int) -> dict | None:
    con = db_connect()  # read-only connection
    try:
        o = con.execute(
            "SELECT order_id, order_date, channel, status, discount_pct FROM orders WHERE order_id = ?", (int(order_id),)
        ).fetchone()
        if o is None:
            return None
        items = con.execute(
            "SELECT p.name AS product, i.quantity, i.unit_price FROM order_items i "
            "JOIN products p ON p.product_id = i.product_id WHERE i.order_id = ? ORDER BY i.item_id",
            (int(order_id),),
        ).fetchall()
    finally:
        con.close()
    subtotal = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)
    return {
        "order_id": o["order_id"],
        "order_date": o["order_date"],
        "channel": o["channel"],
        "status": o["status"],
        "discount_pct": o["discount_pct"],
        "items": [{"product": i["product"], "quantity": i["quantity"], "unit_price": i["unit_price"]} for i in items],
        "subtotal": subtotal,
        "total": round(subtotal * (1 - o["discount_pct"] / 100), 2),
    }


def refund_eligibility(order_id: int, as_of: date = AS_OF) -> dict:
    """Apply the return policy to one order. Pure function of the database row and ``as_of`` - no model involved.

    Rules (returns-and-refunds.md): a full refund within 30 days of delivery (day 30 counts, day 31 does not), for
    unused items in original packaging; one refund per order; after the window a "changed my mind" return is not
    refundable, but a defective product is covered by the warranty. The packaging condition cannot be seen in the
    data, so it is returned as a condition for the agent to confirm with the customer.
    """
    order = _fetch_order(order_id)
    if order is None:
        return {"order_id": order_id, "found": False, "eligible_for_return_refund": False, "reason": "No order with this id exists."}
    base = {"order_id": order["order_id"], "found": True, "status": order["status"], "total": order["total"]}
    if order["status"] == "cancelled":
        return {**base, "eligible_for_return_refund": False, "reason": "The order was cancelled, so there is nothing to return."}
    if order["status"] == "returned":
        return {**base, "eligible_for_return_refund": False,
                "reason": "This order has already been refunded; policy allows only one refund per order."}
    delivered_on = date.fromisoformat(order["order_date"]) + timedelta(days=ASSUMED_TRANSIT_DAYS)
    days = (as_of - delivered_on).days
    warranty = [
        {"product": i["product"], "covered_by_warranty": days <= WARRANTY_DAYS.get(i["product"], WARRANTY_DAYS["default"])}
        for i in order["items"]
    ]
    common = {**base, "assumed_delivery_date": delivered_on.isoformat(), "days_since_delivery": days,
              "return_window_days": RETURN_WINDOW_DAYS, "warranty": warranty, "as_of": as_of.isoformat()}
    if days < 0:
        return {**common, "within_return_window": False, "eligible_for_return_refund": False,
                "reason": f"Not delivered yet (assumed delivery {delivered_on.isoformat()}); a return can start after delivery."}
    within = days <= RETURN_WINDOW_DAYS
    if within:
        reason = (f"Delivered {days} days ago, inside the {RETURN_WINDOW_DAYS}-day window. Eligible for a full refund to the "
                  "original payment method (5-7 business days) if the item is unused and in its original packaging.")
    else:
        reason = (f"Delivered {days} days ago, outside the {RETURN_WINDOW_DAYS}-day window. A 'changed my mind' return is not "
                  "refundable (store credit is at an agent's discretion). A defective product may still be covered by the warranty.")
    return {**common, "within_return_window": within, "eligible_for_return_refund": within, "reason": reason,
            "condition_to_confirm": "item unused and in original packaging" if within else None}


def handoffs_this_turn(messages: list[BaseMessage]) -> int:
    """Successful transfers since the last user message, counted from the history itself."""
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and str(m.content).startswith(TRANSFER_OK):
            n += 1
    return n


def tool_rounds(messages: list[BaseMessage]) -> int:
    """Tool results since the last user message (one per tool call)."""
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            n += 1
    return n


# --------------------------------------------------------------------------- tools
@tool
def lookup_order(order_id: int) -> str:
    """Look up one Lumen Home order by its numeric id: date, status, items and total. Returns JSON."""
    order = _fetch_order(order_id)
    return json.dumps(order if order else {"order_id": order_id, "found": False, "error": "No order with this id exists."})


@tool
def check_refund_eligibility(order_id: int) -> str:
    """Apply the 30-day return policy to an order and say whether a refund is possible, and why. Returns JSON."""
    return json.dumps(refund_eligibility(order_id))


@tool
def search_kb(query: str) -> str:
    """Search the public Lumen Home support articles (setup, wifi, firmware, warranty, safety...). Returns the top matches with ids."""
    hits = get_retriever("support").search(query, k=3)  # audience="public": never the internal refund matrix
    if not hits:
        return "No matching article found."
    return "\n\n".join(f"[{h.doc.id}] {h.doc.title}\n{h.doc.text}" for h in hits)


def make_handoff_tool(source: str, target: str, when: str):
    """Build ``transfer_to_<target>`` for agent ``source``.

    The tool runs inside the source agent's subgraph, so it returns ``Command(graph=Command.PARENT)`` to jump to the peer
    in the parent graph. A parent-level jump drops the subgraph's own pending state, so the update carries the whole
    message history plus the answer to this tool call; ``add_messages`` merges by id, so nothing is duplicated.
    """

    @tool(
        f"transfer_to_{target}",
        description=f"Hand the conversation to the {target} specialist when {when}. "
        "Argument 'reason': one sentence saying what the customer needs and what is already known, so they never have to repeat it.",
    )
    def handoff(reason: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Any:
        messages: list[BaseMessage] = list(state["messages"])
        if handoffs_this_turn(messages) >= MAX_HANDOFFS_PER_TURN:
            return (f"Transfer refused: the conversation has already been transferred {MAX_HANDOFFS_PER_TURN} times since the "
                    "customer's last message. Answer the customer yourself, or tell them a human agent will follow up.")
        answer = ToolMessage(content=f"{TRANSFER_OK}{target}. Reason: {reason}", name=f"transfer_to_{target}", tool_call_id=tool_call_id)
        # Any sibling tool calls in the same AI message must also be answered, or the history is invalid for OpenAI.
        calling = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls), None)
        siblings = [
            ToolMessage(content="Not executed: the conversation was transferred in the same step.", name=tc["name"], tool_call_id=tc["id"])
            for tc in (calling.tool_calls if calling else [])
            if tc["id"] != tool_call_id
        ]
        return Command(
            goto=target,
            graph=Command.PARENT,
            update={
                "active_agent": target,
                "messages": messages + siblings + [answer],
                "handoffs": [{"from": source, "to": target, "reason": reason}],
            },
        )

    return handoff


# --------------------------------------------------------------------------- agents
COMMON_RULES = (
    "Be brief (under 120 words) and warm. The whole conversation so far is visible to you, so never ask the customer to "
    "repeat anything they already said. If the conversation was just transferred to you, carry on where the previous "
    "agent stopped - do not greet again."
)

PROMPTS = {
    "triage": (
        "You are the front desk of Lumen Home customer support. You do NOT solve problems. Work out what the customer "
        "needs and transfer them immediately to the right specialist: billing (orders, charges, returns, refunds, "
        "subscriptions) or technical (devices, wifi, setup, firmware, error codes, anything that smells of burning or "
        "overheats). Ask one short clarifying question only if you truly cannot tell. " + COMMON_RULES
    ),
    "billing": (
        "You are a billing and returns specialist at Lumen Home. Use lookup_order and check_refund_eligibility for every "
        "fact about an order - never guess a date, status, price or whether a refund is possible. If the customer has not "
        "given an order number, ask for it. You cannot issue refunds or change orders: say what the policy result is and "
        "what happens next, and confirm the item must be unused and in its original packaging. If the conversation turns to "
        "a device fault, wifi, setup or firmware, transfer to technical instead of trying to answer. " + COMMON_RULES
    ),
    "technical": (
        "You are a device-support engineer at Lumen Home. Call search_kb before answering any technical question and answer "
        "ONLY from the articles it returns, citing their ids in square brackets, for example [firmware-updates]. Give "
        "numbered steps, most likely fix first. If the customer mentions smoke, a burning smell, sparks or overheating, tell "
        "them to unplug the device now and that a human safety specialist must follow up (you cannot escalate yourself). If "
        "the question is about orders, charges or refunds, transfer to billing. If the articles do not cover the problem, say "
        "a human agent will follow up. " + COMMON_RULES
    ),
}


def make_agent(name: str, tools: list):
    """One peer: a small ReAct loop (agent <-> tools) compiled as a subgraph that shares ``messages`` with the parent."""
    tool_node = ToolNode(tools)

    def agent(state: AgentState) -> dict:
        history = state["messages"]
        if tool_rounds(history) >= MAX_TOOL_ROUNDS:  # cut a runaway tool loop without another model call
            return {"messages": [AIMessage(content=FALLBACK_REPLY)], "active_agent": name, "mode": mode_label()}
        # One tool call per step: a transfer must be the only thing the model does in its message.
        model = get_llm().bind_tools(tools, parallel_tool_calls=False)
        reply = model.invoke([SystemMessage(PROMPTS[name]), *history])
        return {"messages": [reply], "active_agent": name, "mode": mode_label()}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools"; plain text -> END
    g.add_edge("tools", "agent")
    return g.compile(name=name)


def load_case(state: SwarmState) -> dict:
    """Turn a picked ready-made case into the first chat message; leave a message you typed alone.

    ``messages`` has the ``add_messages`` reducer, so a case's message is APPENDED to the chat. "Real input" therefore
    means "the state already has messages": a blank form on a thread with history must not inject a new opening
    message, while a picked case always does. A picked case is a new customer contact, so it also resets
    ``active_agent`` to the front desk (a typed message still resumes with whoever held the conversation last).
    """
    update = expand(state, CASES, has_input=lambda s: bool(s.get("messages")))
    return {**update, "active_agent": "triage"} if update.get("messages") else update


def route_to_active_agent(state: SwarmState) -> Literal["triage", "billing", "technical"]:
    """Where a new user message goes: to whoever held the conversation last (triage for a brand-new thread)."""
    active = state.get("active_agent")
    return active if active in AGENTS else "triage"  # type: ignore[return-value]


# Who may hand the conversation to whom, and when. One table drives the tools the model sees AND the edges Studio draws.
HANDOFF_WHEN: dict[tuple[str, str], str] = {
    ("triage", "billing"): "the customer asks about orders, charges, returns or refunds",
    ("triage", "technical"): "the customer has a device, wifi, setup, firmware or safety problem",
    ("billing", "technical"): "the customer raises a device fault, wifi, setup or firmware problem",
    ("technical", "billing"): "the customer asks about an order, a charge, a return or a refund",
}
OWN_TOOLS = {"triage": [], "billing": [lookup_order, check_refund_eligibility], "technical": [search_kb]}


def build_graph(**compile_kwargs):
    g = StateGraph(SwarmState, input_schema=SwarmIn, output_schema=SwarmOut)
    g.add_node("load_case", load_case)
    for name in AGENTS:
        peers = [dst for (src, dst) in HANDOFF_WHEN if src == name]
        handoffs = [make_handoff_tool(name, dst, HANDOFF_WHEN[(name, dst)]) for dst in peers]
        # destinations tells Studio which peers an agent can jump to (the jumps themselves come from the transfer tools)
        g.add_node(name, make_agent(name, OWN_TOOLS[name] + handoffs), destinations=(*peers, END))
    g.add_edge(START, "load_case")
    g.add_conditional_edges("load_case", route_to_active_agent, list(AGENTS))
    for name in AGENTS:
        g.add_edge(name, END)
    return g.compile(**compile_kwargs)


graph = build_graph()

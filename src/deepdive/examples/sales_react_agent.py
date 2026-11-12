"""A chat analyst that answers questions about the sales data with tools, never from memory.

PROBLEM
    An operations analyst wants to chat with the sales data: "Which region had the highest return rate for Cam Pro last
    quarter?", "How did Europe do in October compared with July?". The answers feed decisions, so the numbers must be
    exact. A model asked for a figure will happily produce a plausible one, and a model left in a tool loop can call
    tools forever or drag an ever-growing chat history into every request.

DESIGN
    A hand-built ReAct loop (the same shape as ``create_react_agent``, written out so each part is visible and
    adjustable), with four guard rails:

      * Numbers come from tools. The system prompt requires a tool result behind every figure and gives the model a
        ``query_sales`` tool (group by product / region / month ..., with filters, over exact pandas code in
        ``_sales_tools``), a ``describe_dataset`` tool, and the shared ``calculator`` for arithmetic. The prompt also
        resolves relative dates for it ("last quarter" is 2026Q3 because the dataset's "today" is 2026-10-31), a classic
        place for a model to be quietly wrong.
      * Tool errors are conversation, not crashes. ``ToolNode`` is created with a ``handle_tool_errors`` function, so
        an unknown product or a malformed period comes back as an error ``ToolMessage`` (with the list of valid
        values) and the model corrects itself on the next turn.
      * The tool loop is bounded by a counter in state. ``prepare_turn`` resets ``tool_rounds`` to 0 for each user
        message; every model reply that asks for tools increments it. When it reaches ``max_tool_rounds`` (default 5)
        the next model call is made WITHOUT tools and with a note to answer from what it has, so a model that never
        stops asking cannot loop, and the user still gets a partial answer instead of an error.
      * The history is bounded. ``prepare_turn`` trims the stored conversation to a token budget before the model
        sees it: whole older messages are removed from state with ``RemoveMessage`` (so the checkpoint stays small, not
        only the prompt), the window always starts at a user message so tool calls are never separated from their
        results, and the newest question is always kept. Trimming once per user message is enough: inside a turn the
        history can only grow by the tool rounds, which the previous guard rail caps.

GRAPH
    START -> prepare_turn -> agent --(no tool calls)--> END
                              ^ |
                              | +--(tool calls)--> tools
                              +---------------------+

LANGGRAPH FEATURES
    * ``messages`` state with the ``add_messages`` reducer, plus a ``RemoveMessage`` trim step
    * ``ToolNode`` with ``handle_tool_errors`` and ``tools_condition`` (the prebuilt pieces, wired by hand)
    * ``model.bind_tools`` inside a node, and a tool-less call at the loop limit
    * an explicit loop counter in state instead of the recursion limit
    * token streaming: run with ``stream_mode="messages"`` and filter on ``metadata["langgraph_node"] == "agent"`` to
      show only the analyst's words (tool results are separate messages from the ``tools`` node)

MODELS AND DATA
    OpenAI chat model with tool calling. Data: ``sales.csv``. Offline, the stand-in model picks one tool by keyword
    overlap, fills its arguments with placeholders, and then repeats the tool output as its "answer", so you see the
    ReAct loop run once but not a real analysis: run live for that. Limits: the model can still misread a question or
    pick a poor grouping, and nothing in this graph verifies that the final text only quotes tool results (the prompt
    requires it and the tool calls are visible in the trace; ``report_orchestrator`` shows a deterministic number
    check you could add after the ``agent`` node).

TRY IT IN STUDIO
    Open the ``Ready-made case`` dropdown, pick one and press Submit (or leave the form blank on a new thread: that runs
    the first one). Each case is an opening question plus a tool-round limit; the tool results show how many order lines
    each figure rests on:
      * ``Which region had the highest return rate for Lumen Cam Pro last quarter?`` - APAC leads at 20%, on only 10
        order lines; a good answer says the rate is shaky. Then follow up in the same thread (type in the chat) with
        "And in the quarter before?" to see history and the quarter arithmetic at work.
      * ``Compare Europe's October revenue with July's`` (3 rounds) - two figures, then the calculator for the change.
      * ``What did customers say about the Therm price?`` - the sales data has no such field, so a good run says so.
      * ``How many Lumen Doorbells did we sell last quarter?`` - the product is called ``Lumen Bell``; a wrong name
        comes back as an error ``ToolMessage`` listing the valid ones, and the model corrects itself.
      * ``What products and regions does the data cover, and what is 'last quarter'?`` - the ``describe_dataset`` tool.
      * ``How did each product's revenue change from Q2 to Q3?`` (1 tool round) - more work than one round allows; watch
        the next model call be made without tools and answer from what it has.
    To ask your own, paste ``{"messages": [{"role": "user", "content": "Compare Europe's October revenue with July's"}],
    "max_tool_rounds": 3}``.

EXTEND IT
    Add a ``plot_series`` tool that returns chart data for the UI to render. Persist per-user preferences (default
    region, currency) in the store and put them in the system prompt. Replace the token-budget trim with a summary of
    the dropped messages.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm
from deepdive.common.scenarios import expand, scenario_field
from deepdive.common.tools import calculator
from deepdive.examples import _sales_tools as st

META = ExampleMeta(
    id="sales_react_agent",
    title="Chat with the sales data (bounded ReAct agent)",
    category="Agents & tools",
    problem="Let an analyst chat with the sales data and get exact numbers from tools, with tool errors recovered, "
    "the tool loop capped and the chat history trimmed.",
    features=["add_messages + RemoveMessage trimming", "ToolNode + handle_tool_errors", "tools_condition loop", "bounded agent loop", "token streaming"],
    models=["OpenAI chat (tool calling)"],
    datasets=["sales.csv"],
    difficulty="starter",
)

EXAMPLE_INPUT = {"messages": [{"role": "user", "content": "Which region had the highest return rate for Lumen Cam Pro last quarter?"}]}

# Ready-made opening messages for Studio's dropdown (see deepdive.common.scenarios). The first one runs on a blank form.
# Every case sets the same two keys (None = the default round limit). The facts in the labels come from data/sales.csv and
# the tool layer and are pinned by tests/test_sales_react_agent.py; what the model does with each is only ever "watch for".
CASES: dict[str, dict] = {
    "Which region had the highest return rate for Lumen Cam Pro last quarter? -> APAC 20% on just 10 lines: watch the caveat": {
        "messages": [{"role": "user", "content": "Which region had the highest return rate for Lumen Cam Pro last quarter?"}],
        "max_tool_rounds": None},
    "Compare Europe's October revenue with July's (3 rounds) -> gross $5,254.70 vs $3,446.30: watch the calculator": {
        "messages": [{"role": "user", "content": "Compare Europe's October revenue with July's"}],
        "max_tool_rounds": 3},
    "What did customers say about the Therm price? -> the sales data has no review text: does a good run say so?": {
        "messages": [{"role": "user", "content": "What did customers say about the Therm price?"}],
        "max_tool_rounds": None},
    "How many Lumen Doorbells did we sell last quarter? -> no such product ('Lumen Bell'): watch for an error, then a retry": {
        "messages": [{"role": "user", "content": "How many Lumen Doorbells did we sell last quarter?"}],
        "max_tool_rounds": None},
    "What products and regions does the data cover, and what is 'last quarter'? -> watch describe_dataset: it says 2026Q3": {
        "messages": [{"role": "user", "content": "What products and regions does the data cover, and what is 'last quarter'?"}],
        "max_tool_rounds": None},
    "How did each product's revenue change from Q2 to Q3, in percent? (1 tool round) -> watch the cap stop the loop": {
        "messages": [{"role": "user", "content": "How did each product's revenue change from Q2 to Q3, in percent?"}],
        "max_tool_rounds": 1},
}
SCENARIO = scenario_field(CASES)

DEFAULT_MAX_TOOL_ROUNDS = 5
MAX_TOOL_ROUNDS_LIMIT = 10
HISTORY_TOKEN_BUDGET = 3000  # approximate tokens of chat history kept in state (read at call time, so tests can lower it)


# --------------------------------------------------------------------------- tools
class QuerySalesArgs(BaseModel):
    """Arguments of ``query_sales``. Everything is optional; defaults give the company-wide total revenue."""

    metrics: list[Literal["revenue", "units", "orders", "margin", "return_rate", "avg_discount"]] = Field(
        default=["revenue"], description="What to compute. return_rate and avg_discount are percentages."
    )
    group_by: Literal["product", "category", "region", "segment", "channel", "month", "quarter", "status"] | None = Field(
        default=None, description="Break the result down by this field. Omit for a single total."
    )
    product: str | None = Field(default=None, description="Filter, e.g. 'Lumen Cam Pro' (the 'Lumen ' prefix is optional)")
    category: str | None = Field(default=None, description="Filter: cameras, climate, doorbells, hubs, lighting, locks, plugs, sensors")
    region: str | None = Field(default=None, description="Filter: North America, Europe, UK & Ireland, APAC")
    segment: str | None = Field(default=None, description="Filter: consumer, prosumer, small_business")
    channel: str | None = Field(default=None, description="Filter: web, app, retail_partner, marketplace")
    status: Literal["completed", "delivered", "returned", "cancelled", "all"] = Field(
        default="completed", description="completed = delivered + returned lines (the default); cancelled lines are excluded unless 'all' or 'cancelled'"
    )
    period: str | None = Field(
        default=None, description="Time window: a quarter (2026Q3), month (2026-05), year (2026), date, or range (2026-04-01..2026-06-30, 2026-02..2026-04). Omit for all data"
    )
    top_n: int | None = Field(default=None, description="Keep only the first n groups after ranking (needs group_by)")
    sort_by: Literal["revenue", "units", "orders", "margin", "return_rate", "avg_discount"] | None = Field(
        default=None, description="Ranking metric (one of the requested metrics). Default: the first metric"
    )
    ascending: bool = Field(default=False, description="Rank lowest first")


@tool(args_schema=QuerySalesArgs)
def query_sales(
    metrics: list[str] | None = None,
    group_by: str | None = None,
    product: str | None = None,
    category: str | None = None,
    region: str | None = None,
    segment: str | None = None,
    channel: str | None = None,
    status: str = "completed",
    period: str | None = None,
    top_n: int | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
) -> str:
    """Query the Lumen Home sales table (order lines, 2025-11-01 to 2026-10-31) and get EXACT aggregates as JSON.

    Returns `total` (all matched order lines) and, with group_by, `rows` (one per group, each with the number of order
    `lines` behind it). Use it for every figure you report. A group's rate rests on few lines if `lines` is small.
    """
    result = st.query_sales(
        metrics=metrics, group_by=group_by, product=product, category=category, region=region, segment=segment,
        channel=channel, status=status, period=period, top_n=top_n, sort_by=sort_by, ascending=ascending,
    )
    return json.dumps(result, separators=(",", ":"))


@tool
def describe_dataset() -> str:
    """Describe the sales data: valid products, regions, segments, channels, metric definitions, today's date, and what
    'last quarter', 'this quarter' and 'last month' mean. Call it when unsure which values are valid."""
    return json.dumps(st.describe_dataset(), separators=(",", ":"))


TOOLS = [describe_dataset, query_sales, calculator]


def explain_tool_error(error: Exception) -> str:
    """Turn any tool failure into text the model can act on (this becomes the content of an error ToolMessage)."""
    if isinstance(error, ToolInvocationError):
        return f"Invalid arguments for {error.tool_name}: {error.message} Fix the arguments and call the tool again."
    return f"Tool error: {error} Fix the arguments and call the tool again, or answer without it if the data cannot help."


# --------------------------------------------------------------------------- state
class AgentIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made opening message (and round limit); added to `messages` below
    messages: Annotated[list[AnyMessage], add_messages]
    max_tool_rounds: int  # tool-calling rounds allowed per user message (default 5, at most 10)


class AgentOut(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    mode: str  # "live" or "offline"


class AgentState(AgentIn, total=False):
    tool_rounds: int  # model replies that requested tools during the current user message
    mode: str


# --------------------------------------------------------------------------- history trimming (pure, unit-tested)
def trim_history(messages: list[AnyMessage], budget: int) -> list[RemoveMessage]:
    """``RemoveMessage`` for every message outside the newest ``budget`` tokens of conversation.

    ``strategy="last"`` keeps the most recent messages; ``start_on="human"`` makes the kept window begin at a user
    message, so an assistant tool call is never separated from its tool result. The latest user message and everything
    after it are always kept, even if that alone exceeds the budget.
    """
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    kept = trim_messages(
        messages, max_tokens=budget, token_counter=count_tokens_approximately, strategy="last", start_on="human", allow_partial=False
    )
    if len(kept) < len(messages) - last_human:  # the newest question did not fit: keep it anyway
        kept = messages[last_human:]
    kept_ids = {m.id for m in kept}
    return [RemoveMessage(id=m.id) for m in messages if m.id not in kept_ids]


# --------------------------------------------------------------------------- nodes
def prepare_turn(state: AgentState) -> dict:
    """Start of a user turn: reset the tool-round counter and trim old history.

    This is the entry node, so it first resolves a picked ready-made case. In a chat the messages ARE the thread's
    memory, so "the caller typed input" means the thread already has messages (or a round limit was typed with none): a
    case is then never added unless it was picked. A picked case appends its opening message to the history, like any
    message the user types; on a fresh thread that is the whole conversation.
    """
    picked = expand(state, CASES, has_input=lambda s: bool(s.get("messages")) or s.get("max_tool_rounds") is not None)
    opening = add_messages([], picked["messages"]) if picked.get("messages") else []  # dicts -> messages, with ids
    messages = add_messages(state.get("messages") or [], opening)  # what the reducer will make of the history
    max_rounds = picked["max_tool_rounds"] if "max_tool_rounds" in picked else state.get("max_tool_rounds")
    if not messages:
        raise ValueError("Send at least one user message, e.g. 'Which region had the highest return rate for Cam Pro last quarter?'")
    cap = DEFAULT_MAX_TOOL_ROUNDS if max_rounds is None else max_rounds
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= MAX_TOOL_ROUNDS_LIMIT:
        raise ValueError(f"max_tool_rounds must be a whole number between 1 and {MAX_TOOL_ROUNDS_LIMIT}; got {cap!r}.")
    # The opening message goes first, then the removals: both are one update, so the trim sees the new question.
    return {
        **picked,
        "tool_rounds": 0,
        "max_tool_rounds": cap,
        "mode": mode_label(),
        "messages": [*opening, *trim_history(messages, HISTORY_TOKEN_BUDGET)],
    }


SYSTEM_PROMPT = """You are the sales analyst assistant for Lumen Home, a smart-home hardware company. You answer questions about its order data using tools.

RULES
1. Every number you state must come from a tool result in this conversation. Never use memory and never estimate. For any arithmetic (differences, ratios, break-even) use the calculator tool.
2. Resolve relative dates yourself and pass them as `period`: {relative}
3. Prefer one query_sales call with group_by over many separate calls. Percent metrics (return_rate, avg_discount) are already percentages. Note: a rate based on fewer than 20 order lines is shaky; say so when it matters.
4. If a tool returns an error, read it, correct the arguments, and call again. Do not repeat a call that failed the same way.
5. If the data cannot answer the question (it has no customer-satisfaction, forecast or competitor fields), say so plainly instead of guessing.
6. Answer briefly. Name the period and filters you used, give units (USD, %, units), and for ranking questions name the winner first.

DATA
{brief}"""

LIMIT_NOTE = (
    "\n\nTOOL LIMIT REACHED: you have used all {n} tool-calling rounds allowed for this question and can no longer call tools. "
    "Answer now using only the tool results above, and say clearly what you could not establish."
)
LIMIT_FALLBACK = (
    "I used all {n} of my tool calls for this question without reaching a complete answer. "
    "Try a narrower question (one product, one period) and I will pick it up from the results so far."
)


def system_prompt() -> str:
    d = st.describe_dataset()
    relative = (
        f"today is {d['today']}; 'last quarter' = {d['last_full_quarter']['label']}; 'this quarter' = {d['current_quarter']['label']} "
        f"(October only so far); 'last month' = {d['last_month']}; 'the quarter before last quarter' = {st.last_full_quarter().previous().label}."
    )
    return SYSTEM_PROMPT.format(relative=relative, brief=st.dataset_brief())


def agent(state: AgentState) -> dict:
    """The model step. Within the round limit it can call tools; at the limit it must answer."""
    rounds, cap = state["tool_rounds"], state["max_tool_rounds"]
    at_limit = rounds >= cap
    prompt = system_prompt() + (LIMIT_NOTE.format(n=cap) if at_limit else "")
    model = get_llm() if at_limit else get_llm().bind_tools(TOOLS)  # no tools bound at the limit: the model cannot call any
    reply = model.invoke([SystemMessage(prompt), *state["messages"]])
    if at_limit and reply.tool_calls:  # a real model cannot do this; a scripted one can. Keep the history valid either way.
        reply = AIMessage(content=str(reply.text) or LIMIT_FALLBACK.format(n=cap), id=reply.id)
    return {"messages": [reply], "tool_rounds": rounds + (1 if reply.tool_calls else 0)}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(AgentState, input_schema=AgentIn, output_schema=AgentOut)
    g.add_node("prepare_turn", prepare_turn)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=explain_tool_error))
    g.add_edge(START, "prepare_turn")
    g.add_edge("prepare_turn", "agent")
    g.add_conditional_edges("agent", tools_condition, ["tools", END])  # tool calls -> tools, otherwise END
    g.add_edge("tools", "agent")
    return g.compile(**compile_kwargs)


graph = build_graph()

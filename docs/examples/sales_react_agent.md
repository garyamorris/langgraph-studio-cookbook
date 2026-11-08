# Chat with the sales data (bounded ReAct agent)

> Let an analyst chat with the sales data and get exact numbers from tools, with tool errors recovered, the tool loop capped and the chat history trimmed.

| | |
|---|---|
| **Example id** | `sales_react_agent` |
| **Category** | Agents & tools |
| **Difficulty** | starter |
| **Models** | OpenAI chat (tool calling) |
| **Data** | `sales.csv` |
| **LangGraph features** | add_messages + RemoveMessage trimming, ToolNode + handle_tool_errors, tools_condition loop, bounded agent loop, token streaming |
| **Source** | [`src/deepdive/examples/sales_react_agent.py`](../../src/deepdive/examples/sales_react_agent.py) |

---

## Problem

An operations analyst wants to chat with the sales data: "Which region had the highest return rate for Cam Pro last
quarter?", "How did Europe do in October compared with July?". The answers feed decisions, so the numbers must be
exact. A model asked for a figure will happily produce a plausible one, and a model left in a tool loop can call
tools forever or drag an ever-growing chat history into every request.

## Design

A hand-built ReAct loop (the same shape as `create_react_agent`, written out so each part is visible and
adjustable), with four guard rails:

  * Numbers come from tools. The system prompt requires a tool result behind every figure and gives the model a
    `query_sales` tool (group by product / region / month ..., with filters, over exact pandas code in
    `_sales_tools`), a `describe_dataset` tool, and the shared `calculator` for arithmetic. The prompt also
    resolves relative dates for it ("last quarter" is 2026Q3 because the dataset's "today" is 2026-10-31), a classic
    place for a model to be quietly wrong.
  * Tool errors are conversation, not crashes. `ToolNode` is created with a `handle_tool_errors` function, so
    an unknown product or a malformed period comes back as an error `ToolMessage` (with the list of valid
    values) and the model corrects itself on the next turn.
  * The tool loop is bounded by a counter in state. `prepare_turn` resets `tool_rounds` to 0 for each user
    message; every model reply that asks for tools increments it. When it reaches `max_tool_rounds` (default 5)
    the next model call is made WITHOUT tools and with a note to answer from what it has, so a model that never
    stops asking cannot loop, and the user still gets a partial answer instead of an error.
  * The history is bounded. `prepare_turn` trims the stored conversation to a token budget before the model
    sees it: whole older messages are removed from state with `RemoveMessage` (so the checkpoint stays small, not
    only the prompt), the window always starts at a user message so tool calls are never separated from their
    results, and the newest question is always kept. Trimming once per user message is enough: inside a turn the
    history can only grow by the tool rounds, which the previous guard rail caps.

## Graph

```text
START -> prepare_turn -> agent --(no tool calls)--> END
                          ^ |
                          | +--(tool calls)--> tools
                          +---------------------+
```

## LangGraph features

* `messages` state with the `add_messages` reducer, plus a `RemoveMessage` trim step
* `ToolNode` with `handle_tool_errors` and `tools_condition` (the prebuilt pieces, wired by hand)
* `model.bind_tools` inside a node, and a tool-less call at the loop limit
* an explicit loop counter in state instead of the recursion limit
* token streaming: run with `stream_mode="messages"` and filter on `metadata["langgraph_node"] == "agent"` to
  show only the analyst's words (tool results are separate messages from the `tools` node)

## Models and data

OpenAI chat model with tool calling. Data: `sales.csv`. Offline, the stand-in model picks one tool by keyword
overlap, fills its arguments with placeholders, and then repeats the tool output as its "answer", so you see the
ReAct loop run once but not a real analysis: run live for that. Limits: the model can still misread a question or
pick a poor grouping, and nothing in this graph verifies that the final text only quotes tool results (the prompt
requires it and the tool calls are visible in the trace; `report_orchestrator` shows a deterministic number
check you could add after the `agent` node).

## Try it in Studio

Open the graph in chat mode and ask "Which region had the highest return rate for Lumen Cam Pro last quarter?"
(the tool result also shows how many order lines each rate rests on). Then follow up in the same thread with "And
in the quarter before?" to see history and the quarter arithmetic at work. Or paste the input
`{"messages": [{"role": "user", "content": "Compare Europe's October revenue with July's"}], "max_tool_rounds": 3}`.
Ask "What did customers say about the Therm price?": the sales data has no such field, so a good run says so.

## Extend it

Add a `plot_series` tool that returns chart data for the UI to render. Persist per-user preferences (default
region, currency) in the store and put them in the system prompt. Replace the token-budget trim with a summary of
the dropped messages.

## Compiled graph

Drawn from the compiled graph by `scripts/gen_docs.py` (one level of subgraphs expanded).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	prepare_turn(prepare_turn)
	agent(agent)
	tools(tools<hr/><small><em>__tools = describe_dataset, query_sales, calculator</em></small>)
	__end__([<p>__end__</p>]):::last
	__start__ --> prepare_turn;
	agent -.-> __end__;
	agent -.-> tools;
	prepare_turn --> agent;
	tools --> agent;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Example input

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Which region had the highest return rate for Lumen Cam Pro last quarter?"
    }
  ]
}
```

Run it: `uv run langgraph dev`, open Studio, choose `sales_react_agent`, paste the input above.

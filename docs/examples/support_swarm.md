# Support swarm (peer handoffs)

> Let a live support conversation move between billing, technical and front-desk agents without making the customer repeat themselves, resuming with whoever last held it on the next message.

| | |
|---|---|
| **Example id** | `support_swarm` |
| **Category** | Multi-agent |
| **Difficulty** | advanced |
| **Models** | OpenAI chat |
| **Data** | `store.sqlite`, `kb/support` |
| **LangGraph features** | Command.PARENT handoffs from tools, agents as subgraphs, persistent active_agent routing from START, scoped tool sets, handoff and tool-round bounds |
| **Source** | [`src/deepdive/examples/support_swarm.py`](../../src/deepdive/examples/support_swarm.py) |

---

## Problem

In a customer chat the right specialist changes mid-conversation. It starts as a billing question ("can I return
the plug from order 1828?") and turns into a device fault ("also, my camera keeps going offline"). A router that
decides once, at the start, keeps the customer with the wrong specialist; sending them back to the front desk makes
them repeat themselves. Customers hate both.

## Design

A *swarm*: three peer agents - `triage`, `billing`, `technical` - and no central supervisor. Whoever holds
the conversation decides when to pass it on, by calling a transfer tool. Each agent is a small ReAct loop
(model <-> tools) that is given ONLY its own tools, so it cannot answer outside its remit:

  * triage    transfer_to_billing, transfer_to_technical. It never solves anything.
  * billing   lookup_order, check_refund_eligibility, transfer_to_technical. Order facts come from tools reading
              `store.sqlite`; the 30-day return rule is plain code, not model judgement, and is evaluated against a
              fixed `AS_OF` date so every run gives the same answer. It cannot issue refunds (no write tool).
  * technical search_kb (public BM25 support articles, with ids to cite), transfer_to_billing.

The conversation lives in one shared `messages` list, so a specialist sees everything said so far. The parent
graph also keeps `active_agent` in state and routes `START` to it, so the *next* user message on the same
thread resumes with whoever last held the conversation - not with triage.

Things a swarm gets wrong if you do not design for them, and what is done here:
  * Message history must stay valid for OpenAI: a tool call needs a matching tool message right after it. A transfer
    leaves the agent's subgraph and the subgraph's own pending messages are dropped (checked: returning only the
    `ToolMessage` leaves an orphaned tool message in the parent), so the transfer tool returns the whole history plus
    its `ToolMessage` (`add_messages` de-duplicates by id). Models are bound with `parallel_tool_calls=False` and any sibling tool
    call in the same message is answered with "not executed", so nothing is ever orphaned.
  * Ping-pong: with no supervisor, nothing stops billing -> technical -> billing -> ... forever. Transfers are capped
    per user turn (`MAX_HANDOFFS_PER_TURN`); beyond it the tool refuses and the agent must answer. The count is
    derived from the message history, so it needs no counter to reset.
  * Runaway tool loops: an agent that keeps calling tools is cut off after `MAX_TOOL_ROUNDS` rounds with a fixed
    "a human will follow up" reply (no model call).

## Graph

```text
START -> load_case --(route on active_agent)--> triage | billing | technical --> END
          each agent is a subgraph:  agent <-> tools   (ToolNode)
          a transfer tool returns Command(goto=<peer>, graph=Command.PARENT) and jumps across to the peer

``load_case`` only turns a picked ready-made case (the ``scenario`` dropdown) into the chat input; it does nothing
for a message you type. The routing edge reads ``active_agent`` AFTER it, so it can send a picked case to the front desk.
```

## LangGraph features

* `Command(goto=..., graph=Command.PARENT, update=...)` returned from a tool: peer-to-peer handoff from inside a
  subgraph, updating `active_agent` and the shared messages in the same step
* agents as subgraphs sharing the parent's state keys (`messages`), each with its own tool set
* `add_conditional_edges(START, ...)` reading persisted state, so a thread continues with the right agent;
  needs a checkpointer (Studio threads supply one; tests pass `InMemorySaver`)
* `ToolNode` with `InjectedState` / `InjectedToolCallId` so a tool can see the history and answer its own call
* explicit bounds on handoffs and tool rounds (cheap defences against loops with no supervisor)

## Models and data

OpenAI chat with tool calling (default tier, one model call per agent step). Data: `store.sqlite` (orders,
order_items, products - read-only) and the public `support` knowledge base (`audience="public"`: the internal
refund approval matrix is never retrievable here).
Simulated / not production-grade: there is NO customer authentication - `lookup_order` returns any order by id,
so a real deployment must verify the customer first. Delivery dates are not in the database: delivery is assumed to
be `ASSUMED_TRANSIT_DAYS` (5) days after the order date, and the return window is evaluated on `AS_OF`
(2026-11-06: five days after the last order, so every "delivered" order counts as delivered). Nobody can actually escalate to a human: the agents say a human will
follow up, and a real system would need an `escalate_to_human` tool.

## Try it in Studio

Pick a case from the **Ready-made case** dropdown and press Submit (a blank form on a new thread runs the first).
The order ids, products and day counts in the labels come from `store.sqlite` and the fixed `AS_OF` date:
  * "Order 1828 ... ELIGIBLE", "Order 1677 ... NOT eligible", "Order 1836 ... NOT eligible": triage should hand the
    customer to billing, which calls `lookup_order` and `check_refund_eligibility`. That verdict is plain code,
    so it is the same on every run (inside the window; 62 days, outside; already refunded).
  * "Cam Pro offline": triage should hand to technical, which searches the knowledge base and cites article ids.
  * "Return from order 1828 AND Cam Pro offline": one message, two specialists. Open `handoffs` and
    `active_agent` in the state: triage -> billing -> technical, inside the cap of 3 transfers per message.
  * "Order 1828 plug 'extremely hot', burning smell": technical should tell the customer to unplug and promise a human
    follow-up. That is the prompt's job only (this example has no safety lexicon; compare `ticket_router`).
Which agent and which tools the MODEL picks is not guaranteed, so the labels say "watch". A picked case is a new
contact: it starts at triage even on a thread that already has a chat, and its message is added to that chat. Then
type a follow-up in the chat view (or put it in `messages`) to see the swarm resume with whoever held the thread
last, for example after the first case::

    {"messages": [{"role": "user", "content": "Thanks. Separately my Cam Pro keeps dropping offline after the update."}]}

The run starts at `billing` (not triage), which transfers to `technical`; technical searches the knowledge base
and cites article ids. A third message about the refund returns to billing. Other orders to ask about: 1767
(delivered exactly 30 days ago: still eligible), 1840 (cancelled). To type a first message yourself, leave the
dropdown empty and send::

    {"messages": [{"role": "user", "content": "Hi, can I still return the Lumen Plug 4-pack from order 1828?"}]}

## Extend it

Add an `escalate_to_human` tool that `interrupt()`s for an agent to take over. Add identity verification before
`lookup_order`. Give each handoff a structured summary (order id, device, what was tried) instead of free text.

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
	load_case(load_case)
	__end__([<p>__end__</p>]):::last
	__start__ --> load_case;
	billing\3a__end__ -.-> __end__;
	billing\3a__end__ -.-> technical\3a__start__;
	load_case -.-> billing\3a__start__;
	load_case -.-> technical\3a__start__;
	load_case -.-> triage\3a__start__;
	technical\3a__end__ -.-> __end__;
	technical\3a__end__ -.-> billing\3a__start__;
	triage\3a__end__ -.-> __end__;
	triage\3a__end__ -.-> billing\3a__start__;
	triage\3a__end__ -.-> technical\3a__start__;
	subgraph triage
	triage\3a__start__(<p>__start__</p>)
	triage\3aagent(agent)
	triage\3atools(tools<hr/><small><em>__tools = transfer_to_billing, transfer_to_technical</em></small>)
	triage\3a__end__(<p>__end__</p>)
	triage\3a__start__ --> triage\3aagent;
	triage\3aagent -.-> triage\3a__end__;
	triage\3aagent -.-> triage\3atools;
	triage\3atools --> triage\3aagent;
	end
	subgraph billing
	billing\3a__start__(<p>__start__</p>)
	billing\3aagent(agent)
	billing\3atools(tools<hr/><small><em>__tools = lookup_order, check_refund_eligibility, transfer_to_technical</em></small>)
	billing\3a__end__(<p>__end__</p>)
	billing\3a__start__ --> billing\3aagent;
	billing\3aagent -.-> billing\3a__end__;
	billing\3aagent -.-> billing\3atools;
	billing\3atools --> billing\3aagent;
	end
	subgraph technical
	technical\3a__start__(<p>__start__</p>)
	technical\3aagent(agent)
	technical\3atools(tools<hr/><small><em>__tools = search_kb, transfer_to_billing</em></small>)
	technical\3a__end__(<p>__end__</p>)
	technical\3a__start__ --> technical\3aagent;
	technical\3aagent -.-> technical\3a__end__;
	technical\3aagent -.-> technical\3atools;
	technical\3atools --> technical\3aagent;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Ready-made cases in Studio

The input form has a **Ready-made case** dropdown, so you do not need to know the data. Pick one and press Submit; a blank form on a new thread runs the first.

| # | Case (as shown in the dropdown) | Input it stands for |
|---|---|---|
| 1 | Order 1828 (Plug 4-pack, 7 days since delivery): return it? -> watch triage > billing; policy check: ELIGIBLE | `{"messages": [{"role": "user", "content": "Hi, can I still return the Lumen Plug 4-pack from order 1828?"}]}` |
| 2 | Cam Pro drops off wifi, error E-102, no order given (T-2084's words) -> watch triage > technical and a KB citation | `{"messages": [{"role": "user", "content": "My Lumen Cam Pro drops off wifi every few hours. I rebooted the router twice and the app shows error E-102."}]}` |
| 3 | Return from order 1828 AND Cam Pro offline in ONE message -> watch handoffs: triage > billing > technical | `{"messages": [{"role": "user", "content": "Can I return the Lumen Plug 4-pack from order 1828? Also, separately, my Lumen Cam Pro keeps dropping off wifi, error E-102."}]}` |
| 4 | Order 1677 (Bell x3 + Plug, 62 days since delivery): 'changed my mind' -> watch billing; policy check: NOT eligible | `{"messages": [{"role": "user", "content": "Hello, I would like to send back the Lumen Bell and the plug from order 1677. I changed my mind. Is that still possible?"}]}` |
| 5 | Order 1836 (Therm) is already refunded; customer asks again -> watch billing; policy check: NOT eligible | `{"messages": [{"role": "user", "content": "Hi, please refund my Lumen Therm from order 1836."}]}` |
| 6 | Order 1828 plug 'extremely hot', burning smell (T-2301 words) -> watch: unplug + human follow-up (prompt only) | `{"messages": [{"role": "user", "content": "One of the plugs in my Lumen Plug 4-pack (order 1828) got extremely hot and there's a burning plastic smell. I've unplugged it. This is a fire risk!!"}]}` |

## Example input

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hi, can I still return the Lumen Plug 4-pack from order 1828?"
    }
  ]
}
```

Run it: `uv run langgraph dev`, open Studio, choose `support_swarm`, pick a case from the **Ready-made case** dropdown, or paste the input above.

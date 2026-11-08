"""Markdown knowledge-base content, written to data/kb/ by make_datasets.py.

Two corpora, each ``{slug: (title, body)}``:

* LANGGRAPH_KB - a compact handbook about LangGraph (verified against langgraph 1.2.x).
  Used by the RAG examples ("ask the LangGraph handbook").
* SUPPORT_KB   - fictional Lumen Home support/policy articles. Used by the support,
  refund and routing examples. The refund thresholds here are the *actual rules* the
  refund-approval example enforces, so keep them in sync.
"""

LANGGRAPH_KB: dict[str, tuple[str, str]] = {
    "state-and-reducers": (
        "State and reducers",
        """
A graph's state schema (a TypedDict, dataclass or Pydantic model) defines its channels: one per key.
Nodes read the current state and return a *partial update*; LangGraph merges that update into state.
By default a key is overwritten by the latest write. To combine writes, annotate the key with a reducer,
for example `Annotated[list, operator.add]` to append, or `add_messages` to merge chat messages by id.
Reducers matter most with parallel branches: if two nodes write the same key in one step and the key has
no reducer, LangGraph raises an InvalidUpdateError. You can also give a graph separate `input_schema` and
`output_schema` so callers only see the keys you intend, while internal keys stay private.
""",
    ),
    "nodes-edges-and-routing": (
        "Nodes, edges and routing",
        """
A StateGraph is built from nodes (plain functions) and edges. `add_edge(a, b)` is a fixed transition;
`add_conditional_edges(a, router)` calls a function after `a` and routes to the node name it returns.
A node can also route itself by returning `Command(goto="next_node", update={...})`, which fuses the state
update and the routing decision in one place; declare `destinations=` on `add_node` (or type the return as a
Literal) so Studio can draw the possible edges. START and END are the special entry and exit.
Cycles are allowed and are how loops and agents are built. The default recursion limit in langgraph 1.2 is very
high (10007, overridable with the LANGGRAPH_DEFAULT_RECURSION_LIMIT environment variable), so always put an
explicit iteration cap in state or pass `recursion_limit` in the run config for any loop.
""",
    ),
    "send-and-map-reduce": (
        "Send and map-reduce fan-out",
        """
When the number of parallel tasks is only known at runtime, return a list of `Send(node_name, payload)` objects
from a conditional edge. Each Send starts one instance of the target node with its *own* input payload, which
does not have to match the graph state. All instances run in the same super-step, and their outputs are merged
through the state reducers, so the collecting key needs a reducer such as `operator.add`. This is the map-reduce
pattern: orchestrator decides the work, workers run in parallel, a downstream node reduces the results.
Use `defer=True` on the reduce node if branches can finish in different super-steps and you need to wait for all.
""",
    ),
    "subgraphs": (
        "Subgraphs",
        """
A compiled graph can be used as a node in another graph. If parent and subgraph share state keys, add the
compiled subgraph directly with `add_node("team", team_graph)`. If their schemas differ, wrap the call in a
function that maps parent state to subgraph input and the subgraph output back to a parent update. Subgraphs
let teams own a self-contained workflow, be tested alone, and be reused. Pass `subgraphs=True` when streaming
to see events from inside them. The checkpointer of the parent is propagated, an interrupt raised inside a
subgraph pauses the whole run, and policies set with `set_node_defaults` are not inherited by subgraphs.
""",
    ),
    "checkpointers-and-threads": (
        "Checkpointers and threads",
        """
A checkpointer saves a snapshot of graph state after every super-step. Snapshots are grouped into a *thread*,
identified by `config={"configurable": {"thread_id": "..."}}`. Threads give you conversation memory, fault
tolerance, human-in-the-loop pauses and time travel. InMemorySaver is fine for development and tests; use a
database-backed saver in production. When you run under `langgraph dev` or LangGraph Studio the platform
supplies persistence for you, so do not compile exported graphs with their own checkpointer. In code, expose a
`build_graph(**kwargs)` helper that accepts a checkpointer for tests and scripts.
""",
    ),
    "human-in-the-loop-interrupts": (
        "Human-in-the-loop with interrupt()",
        """
Call `interrupt(payload)` inside a node to pause the run and surface `payload` to the caller (it appears under
`__interrupt__` in results, and as a prompt in Studio). Resume by invoking the same thread with
`Command(resume=value)`; `interrupt()` then returns `value`. Important: on resume the node restarts from its first
line, so put any side effects (sending email, charging cards) *after* the interrupt or make them idempotent.
When a node calls interrupt more than once, resume values are matched by order. Static breakpoints are also
available via `interrupt_before` / `interrupt_after` at compile time; they are handy for debugging in Studio,
where you can edit state before continuing. Interrupts require a checkpointer.
""",
    ),
    "time-travel": (
        "Time travel: replay and fork",
        """
Every checkpoint of a thread is addressable. `graph.get_state_history(config)` lists them newest first. To
*replay*, invoke with a config that includes an earlier `checkpoint_id`: nodes before that checkpoint are not
re-run, nodes after it are executed again (so LLM calls are repeated). To *fork*, call `graph.update_state(config,
values)` on a past checkpoint, which creates a new checkpoint on a new branch of history, then continue from it.
In Studio this is the "fork" action on a step. Time travel is ideal for what-if analysis and for debugging why an
agent took a wrong turn without re-running the entire expensive prefix.
""",
    ),
    "long-term-memory-store": (
        "Long-term memory with the Store",
        """
Checkpoints are scoped to one thread. For memory that spans threads, such as user preferences, use a Store (a
BaseStore implementation such as InMemoryStore). Items live under a namespace tuple like `("memories", user_id)`
with a key and a JSON value. Use `store.put`, `store.get` and `store.search`; semantic search additionally needs an
index configured with an embeddings model. Nodes receive the store by declaring a `store: BaseStore` parameter,
or through the runtime object. LangGraph Studio and `langgraph dev` provide a store automatically, so the same
graph code works locally and deployed.
""",
    ),
    "streaming": (
        "Streaming modes",
        """
`graph.stream(input, stream_mode=...)` supports several modes: `values` (full state after each step), `updates`
(only what each node returned), `messages` (LLM tokens as they are generated), `custom` (anything you emit with
`get_stream_writer()` inside a node) and `debug`. You can pass a list of modes to receive several at once. Custom
events are a lightweight way to report progress from long-running nodes, like "processed 40 of 200 rows".
With `subgraphs=True`, events from nested graphs are included and tagged with their namespace. Studio uses these
streams to animate execution live.
""",
    ),
    "functional-api": (
        "The Functional API",
        """
Besides StateGraph, LangGraph offers a functional style. Decorate a function with `@entrypoint()` to make a
workflow and use `@task` for units of work. Calling a task returns a future, so you get parallelism by starting
several tasks and then collecting `.result()`. Task results are checkpointed: if the workflow is interrupted and
resumed, finished tasks are not re-executed. `interrupt()` works inside entrypoints, and an entrypoint can accept
a `previous` argument to read the value saved by the prior run on the same thread; return `entrypoint.final(value=...,
save=...)` to return one thing and persist another. It is the most direct way to turn existing Python control flow
into a durable workflow.
""",
    ),
    "retries-caching-and-errors": (
        "Retries, caching, timeouts and error handlers",
        """
Node-level policies are passed to `add_node`. `retry_policy=RetryPolicy(max_attempts, initial_interval,
backoff_factor, retry_on)` retries a failing node with backoff. `cache_policy=CachePolicy(ttl=..., key_func=...)`
skips re-running a node for identical input when you compile with `cache=InMemoryCache()` (or another cache).
`timeout=TimeoutPolicy(run_timeout=..., idle_timeout=...)` bounds an attempt; timeouts rely on asyncio
cancellation, so they cannot interrupt blocking synchronous code. `error_handler=fn` runs after retries are
exhausted: declare `def fn(state, error: NodeError) -> Command` and return a `Command` with an update and a `goto`
to degrade gracefully. `StateGraph.set_node_defaults(...)` applies policies to every node at once.
""",
    ),
    "studio-and-configuration": (
        "LangGraph Studio and langgraph.json",
        """
`langgraph.json` maps graph ids to Python objects, for example `"ticket_router": "./src/pkg/mod.py:graph"`, lists
`dependencies` (usually `["."]`) and an optional `env` file. Run `uv run langgraph dev` to start a local server
(default port 2024) and open Studio against it. Studio lets you pick a graph, submit input, watch nodes execute,
inspect and edit state at any step, fork from a past checkpoint, and answer interrupts. A graph can declare a
`context_schema` (or read `config["configurable"]`) so runs can be parameterised from the Studio configuration
panel, for instance to choose a model or a threshold per run.
""",
    ),
    "multi-agent-patterns": (
        "Multi-agent patterns",
        """
Common architectures: a *supervisor* node decides which worker runs next, usually through structured output and
`Command(goto=...)`; a *swarm* lets peer agents hand off to one another with tools that return `Command(goto=
other_agent, graph=Command.PARENT)`; *hierarchical teams* nest supervisors, each team being a subgraph;
*debate* runs opposing agents in parallel and has a judge decide; *reflection* alternates a generator and a critic.
Prefer the simplest structure that works: a router plus specialists is cheaper and easier to debug than free-form
agent chatter. Always bound loops with an iteration counter, and give each agent only the tools it needs.
""",
    ),
    "prebuilt-agents": (
        "Prebuilt agent building blocks",
        """
The classic tool-calling agent is a loop: an LLM node bound to tools, a `ToolNode` that executes requested tool calls,
and `tools_condition` to route back to the model or to END. You can assemble it by hand, which is instructive, or use
`langgraph.prebuilt.create_react_agent(model, tools, prompt=..., checkpointer=..., store=...)`. Tools can read graph
state or the store through injected arguments (`InjectedState`, `InjectedStore`) without exposing them to the model.
Keep tool descriptions precise, return compact results, and trim long message histories before each model call.
""",
    ),
    "openai-decisions-api": (
        "OpenAI Decisions API",
        """
`client.decisions.create(model="gpt-6-luna", input=..., questions=[...])` answers many classification or scoring
questions about one input in a single call. Each question is a `predicate` (returns a probability that a
statement is true), a `choice` (the chosen option, a confidence and a probability per option) or a `score` (a
rating against ordered levels, with a probability per level). A request may contain 1 to 400 questions and answers
come back in order, each possibly a `refusal`. Because outputs are calibrated probabilities rather than prose, they
are well suited to routing, gating and thresholds, for example auto-approve above 0.9, escalate between 0.5 and
0.9, decline below. It classifies; it does not write text, so pair it with a chat model for generation.
""",
    ),
}


SUPPORT_KB: dict[str, tuple[str, str]] = {
    "returns-and-refunds": (
        "Returns and refunds policy",
        """
Customers may return any Lumen Home product within 30 days of delivery for a full refund if it is unused and in
its original packaging. Defective products are covered for 12 months under warranty (see warranty-and-repairs).
Approved refunds are issued to the original payment method within 5 to 7 business days. Items marked
"changed my mind" after 30 days are not refundable but may be eligible for store credit at the agent's discretion.
Shipping charges are refunded only when the product arrived damaged or defective, or the wrong item was sent.
A customer who has already received one refund for the same order cannot receive a second one.
""",
    ),
    "refund-approval-matrix": (
        "Refund approval matrix (internal)",
        """
Authority levels for refunds. AUTO-APPROVE: amount of $100 or less, within 30 days of delivery, reason is defective,
damaged in transit, wrong item or duplicate purchase, and the customer has fewer than 3 refunds in the past 12
months. MANAGER APPROVAL REQUIRED: amount above $100, or any refund outside the 30-day window but inside warranty,
or a customer with 3 or more refunds in the past 12 months. ALWAYS ESCALATE TO FRAUD REVIEW: 5 or more refunds in 12
months, or a request beyond 90 days with no defect evidence. Photo evidence is required for damage claims above $50.
Managers may approve, partially approve (store credit for the remainder) or reject, and must record a reason.
""",
    ),
    "shipping-and-delivery": (
        "Shipping and delivery",
        """
Standard shipping takes 3 to 6 business days in North America, 4 to 8 in Europe and the UK, and 7 to 14 for APAC.
Orders ship within 1 business day of payment. Tracking is emailed at dispatch. If tracking shows "delivered" but you
cannot find the parcel, check with neighbours and wait 24 hours, then contact support: we open a carrier
investigation and replace or refund the order once it is confirmed lost. Address changes are possible only before
the order ships. Customs duties on international orders are the buyer's responsibility.
""",
    ),
    "warranty-and-repairs": (
        "Warranty and repairs",
        """
All Lumen Home hardware carries a 12-month limited warranty against manufacturing defects. Locks and thermostats
carry 24 months. Warranty does not cover physical damage, water ingress outside rated conditions, or modification.
To make a claim, provide the order number and a short description or video of the fault. We offer a replacement
unit by advance shipment for the Hub, Lock and Therm, or a refund if no replacement is available. Replacement
units are covered for the remainder of the original warranty period.
""",
    ),
    "device-setup-and-wifi": (
        "Device setup and wifi troubleshooting",
        """
Lumen devices connect over 2.4 GHz wifi only. If a device repeatedly drops offline, make sure the router has 2.4 GHz
enabled, keep the device within 10 metres of the Hub or access point, and avoid band-steering features that move it
to 5 GHz. Error E-102 means the device could not reach the cloud; E-317 means pairing timed out; E-504 means the
Hub could not authenticate. A solid red Hub light means no network: power cycle the router, then the Hub, wait two
minutes. If the light stays red after that, perform a factory reset by holding the button for 10 seconds.
""",
    ),
    "firmware-updates": (
        "Firmware updates",
        """
Devices update automatically overnight and the app shows the version under Settings > Device > About. Never unplug
a device during an update: a progress bar that appears stuck can take up to 20 minutes on slow connections. If an
update has not progressed for 30 minutes, leave the device powered, check it is on 2.4 GHz wifi, and contact support
with the device serial number so we can push a recovery image. Firmware 3.0.2 fixes cloud reconnection after power
cuts on the Hub and a pairing timeout (E-317) on Cam Pro units.
""",
    ),
    "account-and-security": (
        "Account, password and two-factor security",
        """
Reset a password from the login screen using the email on the account. If you no longer have access to that email,
support can change it after verifying your identity with the last four digits of your payment card and the shipping
postcode. Two-factor codes are sent by SMS or an authenticator app; if SMS codes do not arrive, use a backup code
or ask support to temporarily disable two-factor after identity verification. Invite family members from Home >
Members instead of sharing credentials. Account deletion requests are processed within 30 days.
""",
    ),
    "billing-and-subscriptions": (
        "Billing and Lumen Care subscriptions",
        """
Lumen Care is an optional subscription at $4.99 per month or $49 per year that adds 30 days of camera clip history
and priority support. It renews automatically; cancellations take effect at the end of the current period, and
renewals can be refunded within 7 days of the charge if the service was unused. Duplicate charges are refunded in
full as soon as they are verified. Invoices with a company VAT number can be generated from the Orders page or by
support. Promotional codes cannot be applied after an order is placed unless support confirms the code was valid
at checkout.
""",
    ),
    "privacy-and-data": (
        "Privacy and data retention",
        """
Camera clips are stored encrypted and kept for 3 days without a subscription or 30 days with Lumen Care. Deleting your
account removes recordings, device history and personal data within 30 days, except records we must keep for tax
purposes. Any report that a customer saw another household's video is treated as a priority security incident:
support must escalate it immediately to the security team and must not discuss technical details with the
customer until security confirms. Customers can request a copy of their data under GDPR from account settings.
""",
    ),
    "safety-incidents": (
        "Product safety incidents",
        """
If a customer reports overheating, burning smell, smoke, sparks, a cracked bulb or any risk of injury, treat it as a
priority safety incident. Advise the customer to stop using and unplug the product immediately. Do not wait for
return logistics: escalate to the Product Safety team within 1 hour with the order number, product and photos if
available, and offer a free replacement or full refund once the unit is returned. Never dismiss a safety report as a
usage issue. Lock failures that leave a customer unable to enter or exit a home are also treated as urgent and
require a phone call, not an email.
""",
    ),
}

"""On-call ops copilot: reads freely, writes only with a human's approval, and never past a hard limit.

PROBLEM
    An on-call engineer chats with a copilot about service health ("the api is slow and throwing 503s") and wants
    it to ACT: restart a service, scale replicas, roll back a deploy. Looking is safe; changing production is not.
    So every change must be shown to the engineer, who can approve it, EDIT its arguments (scale to 4, not the 10 the
    model asked for) or reject it with a reason. And even an approved change must not break a hard rule (never more
    than 20 replicas, never restart the database by accident), because an approval is a human clicking quickly at
    3am, not a safety proof.

DESIGN
    A small simulated environment lives in graph STATE (``env``: four services api / worker / db / cache with status,
    CPU, memory, error rate, replicas and version, plus recent deploys and error logs; everything is deterministic).
    Keeping it in state means a change is visible in Studio's state panel, is checkpointed with the conversation,
    and is isolated per thread, so tests and parallel Studio threads never share an "environment".

    Tools are split by what they can do, and the split is enforced by the graph, not by the prompt:

      * read tools (``get_service_status``, ``get_error_logs``, ``list_recent_deploys``) run in a prebuilt
        ``ToolNode`` and read ``env`` through ``InjectedState`` (the model never sees or sets that argument);
      * write tools (``restart_service``, ``scale_service``, ``rollback_deploy``) are declared so the model can call
        them, but their bodies raise: they are never given to the ``ToolNode``. The only code that changes ``env`` is
        ``apply_write``, reached only from ``approval_gate``.

    One AI message may contain both kinds of call ("show me the logs and restart the worker"). ``route_after_agent``
    splits the message: the read calls go to the ``ToolNode`` as a ``Send`` of just those calls, and when they have
    finished the unanswered write calls go to ``approval_gate``. Reads therefore run first and the engineer sees
    the results while deciding. Every tool call gets exactly one matching ``ToolMessage`` - results, rejections,
    refusals and invalid arguments included - because OpenAI rejects a history in which a tool call has no answer.

    ``approval_gate`` works in two phases. Phase 1 asks: for each pending write it calls ``interrupt()`` with the
    proposed call, a one-line summary, the current state of the service and a preview of the hard-limit check, and
    accepts ``{"action": "approve"}``, ``{"action": "edit", "args": {...}}`` (merged over the proposal and validated)
    or ``{"action": "reject", "reason": "..."}``. An invalid answer is re-asked at most 2 times, then the call is
    rejected (fail closed). Phase 2 acts: only after ALL answers are in does it run anything. That ordering is the
    interrupt rule in action: when a node resumes it restarts from its first line and replays earlier answers, so
    a change applied between two ``interrupt()`` calls would be applied again on the next resume.

    Defence in depth: ``check_limits`` is plain code that runs in phase 2 on the arguments that will actually be used
    (so after an edit too), whatever the human clicked. The gate shows the same check as a preview so the engineer
    knows an approval will be refused, but the preview is advisory and the phase-2 check is the one that counts.
    The model is also told the limits in its prompt so it does not waste approvals; prompts are a courtesy, code is
    the guarantee. The agent loop is bounded by ``max_steps`` (default 6 model calls per user message).

    WHY A DYNAMIC ``interrupt()`` AND NOT A STATIC BREAKPOINT. The same behaviour is possible with a static
    breakpoint in a variant of this graph where the write calls run in a plain node called ``write_tools`` and the
    graph is compiled with ``interrupt_before=["write_tools"]``. The run then stops before that node; you read the
    proposal with ``graph.get_state(config).values["messages"][-1].tool_calls``; to edit, you call
    ``graph.update_state(config, {"messages": [AIMessage(content="", id=<same id>, tool_calls=[edited])]})`` (the same
    id makes ``add_messages`` replace the message); to reject, you write ``ToolMessage`` answers with
    ``update_state(config, {...}, as_node="write_tools")`` so the node is skipped; then ``graph.invoke(None, config)``
    continues. That works, and it is the right tool for debugging, but it is a poor approval flow: the breakpoint is
    unconditional (it stops even when there is nothing to approve, unless you split nodes); the human gets raw state,
    not a tailored case; an edit is state surgery with no validation; and the decision is not data the node can
    reason about, so nothing can re-ask on bad input or run the limit check against what the human chose. A dynamic
    ``interrupt()`` inside the gate asks only when there is something to approve, carries a purpose-built payload,
    and keeps the decision logic in one tested function.

GRAPH
    START -> bootstrap -> agent
    agent         -> END               the message has no tool calls (the answer)
    agent         -> agent             only malformed tool calls: the model tries again (bounded by max_steps)
    agent         -> read_tools        the message has read calls (a Send carrying just those calls)
    agent         -> approval_gate     the message has only write calls
    read_tools    -> approval_gate     write calls from the same message are still unanswered
    read_tools    -> agent             nothing left to approve
    approval_gate -> agent             (every call now has a ToolMessage, whatever happened to it)

    Two conditional edges do the splitting: ``route_after_agent`` (which may return a ``Send``) and
    ``route_after_reads``. ``bootstrap`` creates ``env`` once per thread, resets the step counter for each user message
    and, when you picked a ready-made case from the ``scenario`` dropdown, turns it into the chat input.

LANGGRAPH FEATURES
    * ``interrupt()`` / ``Command(resume=...)`` with approve / edit / reject payloads and a re-ask loop
    * the interrupt-restarts-the-node rule (ask first, act after) and its static-breakpoint equivalent (see DESIGN)
    * prebuilt ``ToolNode`` fed by a conditional edge returning ``Send`` (it accepts a bare list of tool calls),
      with ``InjectedState`` so tools read graph state
    * splitting one AI message's tool calls into read and write paths
    * state-held simulated environment, an append-only ``audit`` list with a reducer
    * a bounded agent loop (``max_steps`` counter in state, reset on each user message by ``bootstrap``)
    * ``input_schema`` / ``output_schema`` with a ``messages`` key so Studio's chat view works

MODELS AND DATA
    OpenAI chat with tool calling (``bind_tools``). No dataset: the environment is invented and fully contained in
    ``initial_env()``. SIMULATED: all of it (services, metrics, logs, deploys, the effect of each action). Offline
    mode uses the deterministic stand-in model, which calls one tool whose name or description overlaps the message;
    that is enough to exercise every path but is not a model of how a real model would diagnose anything. Operator
    identity is not checked; a real deployment needs authentication on whoever can resume the thread.

TRY IT IN STUDIO
    The graph needs a checkpointer for ``interrupt()``; Studio provides one. Pick a case from the **Ready-made case**
    dropdown and press Submit (a blank form on a new thread runs the first). Each case is one message to the copilot:
      * "api 503s ... scale it up": the model should read status and logs, then propose ``scale_service``; the run
        PAUSES at ``approval_gate`` (what the model proposes is its call, so the labels say "watch").
      * "Is the db healthy?": should be reads only, no pause. "Show the worker logs and restart the cache": the read
        runs, then only the restart waits for you. "Why is the worker failing?": reads the logs, usually proposes a rollback.
      * "50 replicas" and "scale the db": the code limits. The approval card shows ``limit_check_preview`` and even
        Approve ends in "REFUSED by safety limits"; the model may also decline before it gets that far.
      * "max_steps=1": the loop stops at its step cap with a fixed message.
    The services, clock and numbers in the case labels are the ones in ``initial_env()``. ``env``, ``audit`` and the chat
    belong to the thread, so a second case on the same thread sees the changes the first one made: start a New Thread
    for the pristine environment the labels describe. To resume a pause, use the interrupt's resume box with the
    values below. Or type your own message (``messages``), for example:
      1. "The api is slow and throwing 503s. Check it and add capacity if that is the right call." The model reads
         status and logs (no pause), then proposes ``scale_service``. The run stops at ``approval_gate``; open the
         interrupt. Resume with {"action": "edit", "args": {"replicas": 4}} and check the ``env`` in the state panel:
         api has 4 replicas, and the ToolMessage tells the model that you changed its number.
      2. "Why is the worker failing?" Reads only: the logs show a KeyError that starts with the 2026.11.4 deploy, and
         the model will usually propose ``rollback_deploy`` to 2026.11.2. Resume with
         {"action": "reject", "reason": "wait for the author"} and watch the model accept it instead of retrying.
      3. Hard limits. Repeat step 1 but resume with {"action": "edit", "args": {"replicas": 50}} (an operator typo):
         the tool result says REFUSED by safety limits, ``env`` is unchanged, and the preview in the interrupt had
         already warned you. Then ask "restart the database": the model sets ``confirm`` because you asked for it;
         resume with {"action": "edit", "args": {"confirm": false}} and the db rule refuses it.
      4. Ask for two things at once ("show the worker logs and restart the cache") to see the split: the read runs,
         then only the write is gated. Open the ``audit`` list in the output for the full trail.
    Resume values for check runs: ``EXAMPLE_RESUMES`` in this module.

EXTEND IT
    Add a ``risk`` field to each write (``low`` auto-approves, ``high`` needs two approvers). Give ``get_error_logs``
    a ``Command``-returning variant that summarises long logs in a subgraph. Replace the simulated ``env`` with
    real API clients behind the same ``apply_write`` function: the gate and the limits stay unchanged.
"""

from __future__ import annotations

import copy
import json
import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Send, interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="ops_copilot",
    title="On-call ops copilot with approved writes",
    category="Human-in-the-loop & persistence",
    problem="Let an on-call copilot read service health freely but run restarts, scaling and rollbacks only after a "
    "human approves, edits or rejects each one, with hard limits enforced in code even after approval.",
    features=[
        "interrupt with approve/edit/reject",
        "ToolNode + Send",
        "tool-call splitting",
        "InjectedState",
        "defence in depth",
        "state-held environment",
        "bounded agent loop",
    ],
    models=["OpenAI chat (tool calling)"],
    datasets=[],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"messages": [{"role": "user", "content": "The api is throwing 503s. Scale it up, it needs more capacity."}]}
EXAMPLE_RESUMES = [{"action": "edit", "args": {"replicas": 4}}]

# Ready-made cases for Studio's dropdown (see deepdive.common.scenarios). Each is one message to the copilot about the
# simulated environment in ``initial_env()`` (api 3 replicas at 93% cpu; worker on 2026.11.4 since deploy D-7731, whose
# previous version is 2026.11.2; db healthy; cache healthy), plus ``max_steps`` (None = the default of 6). What the MODEL
# does with a message is never promised below ("watch ..."); only what the code guarantees (the limits, the step cap) is.
CASES: dict[str, dict] = {
    "api 503s, 3 replicas at 93% cpu: 'scale it up' -> watch for the approval pause on scale_service (try edit to 4)": {
        "messages": list(EXAMPLE_INPUT["messages"]), "max_steps": None},
    "'Is the database healthy?' (db is healthy) -> read-only: watch for an answer with no approval pause": {
        "messages": [{"role": "user", "content": "Is the database healthy right now?"}], "max_steps": None},
    "worker errors since deploy D-7731 (2026.11.4): 'why is it failing?' -> watch for a rollback proposal to 2026.11.2": {
        "messages": [{"role": "user", "content": "Why is the worker failing?"}], "max_steps": None},
    "'Show the worker logs and restart the cache' -> watch the log read run, then a pause for only the restart": {
        "messages": [{"role": "user", "content": "Show me the worker logs and restart the cache."}], "max_steps": None},
    "'Scale the api to 50 replicas' (limit is 20) -> if proposed, even Approve is REFUSED by the limit check": {
        "messages": [{"role": "user", "content": "Scale the api to 50 replicas right now."}], "max_steps": None},
    "'Scale the db to 3 replicas' (stateful) -> never runs: the model declines, or code REFUSES even if you approve": {
        "messages": [{"role": "user", "content": "Scale the database to 3 replicas."}], "max_steps": None},
    "max_steps=1, 'check the api and the worker, which is worse?' -> watch for the fixed 'used my 1 reasoning steps' stop": {
        "messages": [{"role": "user", "content": "Check the api and the worker, and tell me which one is worse."}], "max_steps": 1},
}
SCENARIO = scenario_field(CASES)

# --------------------------------------------------------------------------- hard limits (code, not prompt)
MIN_REPLICAS, MAX_REPLICAS = 1, 20
NOT_SCALABLE = ("db",)  # stateful: scaled by the DBA runbook, not by a chat
NOT_ROLLBACKABLE = ("db",)  # schema migrations do not roll back by version number
CONFIRM_TO_RESTART = ("db",)
MAX_REASKS = 2  # an invalid operator answer is re-asked at most this many times
DEFAULT_MAX_STEPS = 6  # model calls allowed per user message

ServiceName = Literal["api", "worker", "db", "cache"]
WRITE_TOOLS = ("restart_service", "scale_service", "rollback_deploy")


# --------------------------------------------------------------------------- the simulated environment
def initial_env() -> dict:
    """The invented production environment every new thread starts with (fresh copy each call)."""
    return {
        "clock": "2026-11-08T14:32:00Z",
        "services": {
            # `cause` is simulation-internal (never shown by the read tools): what a fix has to address
            "api": {"status": "degraded", "cpu": 93, "memory": 71, "error_rate": 6.8, "replicas": 3, "version": "2026.11.3", "restarts_24h": 0, "cause": "saturation"},
            "worker": {"status": "degraded", "cpu": 58, "memory": 64, "error_rate": 12.4, "replicas": 2, "version": "2026.11.4", "restarts_24h": 0, "cause": "bad_deploy"},
            "db": {"status": "healthy", "cpu": 41, "memory": 73, "error_rate": 0.1, "replicas": 1, "version": "pg-15.4", "restarts_24h": 0, "cause": None},
            "cache": {"status": "healthy", "cpu": 22, "memory": 48, "error_rate": 0.0, "replicas": 2, "version": "redis-7.2.4", "restarts_24h": 0, "cause": None},
        },
        "deploys": [
            {"id": "D-7731", "service": "worker", "version": "2026.11.4", "previous_version": "2026.11.2", "deployed_at": "2026-11-08T13:50:00Z", "author": "dana", "summary": "Rewrite of the nightly SKU sync batch"},
            {"id": "D-7724", "service": "api", "version": "2026.11.3", "previous_version": "2026.11.1", "deployed_at": "2026-11-05T09:15:00Z", "author": "lee", "summary": "Checkout copy changes"},
            {"id": "D-7702", "service": "worker", "version": "2026.11.2", "previous_version": "2026.10.9", "deployed_at": "2026-11-03T16:40:00Z", "author": "sam", "summary": "Retry backoff tuning"},
            {"id": "D-7690", "service": "api", "version": "2026.11.1", "previous_version": "2026.10.8", "deployed_at": "2026-11-02T11:05:00Z", "author": "lee", "summary": "Rate-limit headers"},
            {"id": "D-7655", "service": "cache", "version": "redis-7.2.4", "previous_version": "redis-7.2.3", "deployed_at": "2026-10-28T08:00:00Z", "author": "ops", "summary": "Patch release"},
        ],
        "logs": {
            "api": [
                "14:31:58 WARN  upstream timeout after 2000ms (pool exhausted, 3/3 replicas at cpu>90%)",
                "14:31:41 ERROR 503 on POST /checkout request queue depth 412",
                "14:31:12 WARN  autoscaler disabled by flag AUTOSCALE_OFF",
                "14:30:47 ERROR 503 on GET /cart request queue depth 398",
                "14:30:02 WARN  upstream timeout after 2000ms (pool exhausted, 3/3 replicas at cpu>90%)",
                "14:25:30 INFO  traffic 2.4x the weekly baseline (campaign email sent 14:20)",
            ],
            "worker": [
                "14:32:03 ERROR KeyError: 'sku' in sync_batch (version 2026.11.4)",
                "14:31:33 ERROR KeyError: 'sku' in sync_batch (version 2026.11.4)",
                "14:30:41 ERROR KeyError: 'sku' in sync_batch (version 2026.11.4)",
                "14:12:09 ERROR KeyError: 'sku' in sync_batch (version 2026.11.4)",
                "13:51:20 INFO  worker started, version 2026.11.4",
                "13:49:58 INFO  worker stopped, version 2026.11.2 (deploy D-7731)",
            ],
            "db": ["14:20:00 INFO  checkpoint complete", "14:05:12 INFO  autovacuum finished", "13:50:00 INFO  connections 85/100"],
            "cache": ["14:30:00 INFO  hit rate 94%", "14:00:00 INFO  hit rate 95%"],
        },
    }


def derive_status(svc: dict) -> str:
    if svc["error_rate"] >= 25:
        return "down"
    if svc["error_rate"] >= 2 or svc["cpu"] >= 85 or svc["memory"] >= 90:
        return "degraded"
    return "healthy"


def known_versions(env: dict, service: str) -> list[str]:
    seen: list[str] = []
    for d in env["deploys"]:
        if d["service"] == service:
            for v in (d["version"], d["previous_version"]):
                if v not in seen:
                    seen.append(v)
    return seen


def apply_write(env: dict, name: str, args: dict) -> tuple[dict, str]:
    """The ONLY function that changes ``env``. Returns ``(new_env, message_for_the_model)``. Limits are checked by the caller."""
    env = copy.deepcopy(env)
    svc = env["services"][args["service"]]
    before = {k: svc[k] for k in ("status", "cpu", "memory", "error_rate", "replicas", "version")}
    if name == "restart_service":
        svc["restarts_24h"] += 1
        svc["memory"] = min(svc["memory"], 40)
        msg = f"Restarted all {svc['replicas']} replica(s) of {args['service']}; memory {before['memory']}% -> {svc['memory']}%."
    elif name == "scale_service":
        old, new = svc["replicas"], args["replicas"]
        if new == old:
            return env, f"{args['service']} already has {old} replica(s); nothing changed."
        svc["replicas"] = new
        svc["cpu"] = max(5, round(svc["cpu"] * old / new))
        if svc["cause"] == "saturation" and svc["cpu"] < 85:
            svc["error_rate"], svc["cause"] = 0.6, None
        msg = f"Scaled {args['service']} from {old} to {new} replica(s); cpu {before['cpu']}% -> {svc['cpu']}%."
    elif name == "rollback_deploy":
        current = svc["version"]
        svc["version"] = args["to_version"]
        env["deploys"].insert(0, {
            "id": f"D-{7732 + len(env['deploys'])}", "service": args["service"], "version": args["to_version"],
            "previous_version": current, "deployed_at": env["clock"], "author": "ops_copilot (approved rollback)",
            "summary": f"Rollback from {current}: {args['reason']}",
        })
        if svc["cause"] == "bad_deploy":
            svc["error_rate"], svc["cause"] = 0.4, None
        msg = f"Rolled {args['service']} back from {current} to {args['to_version']}."
    else:  # pragma: no cover - the gate only passes WRITE_TOOLS
        raise ValueError(f"not a write tool: {name}")
    svc["status"] = derive_status(svc)
    after = {k: svc[k] for k in ("status", "cpu", "memory", "error_rate", "replicas", "version")}
    changes = ", ".join(f"{k} {before[k]} -> {after[k]}" for k in after if before[k] != after[k])
    return env, f"{msg} Now: {changes or 'no metric changed'}."


def check_limits(env: dict, name: str, args: dict) -> list[str]:
    """Hard rules, as code. Returns the violated rules (empty list = allowed). Runs after the human has decided."""
    svc_name = args["service"]
    problems: list[str] = []
    if name == "scale_service":
        if svc_name in NOT_SCALABLE:
            problems.append(f"{svc_name} cannot be scaled from here (stateful service; use the DBA runbook).")
        if not MIN_REPLICAS <= args["replicas"] <= MAX_REPLICAS:
            problems.append(f"replicas must be between {MIN_REPLICAS} and {MAX_REPLICAS}; got {args['replicas']}.")
    elif name == "restart_service":
        if svc_name in CONFIRM_TO_RESTART and not args.get("confirm"):
            problems.append(f"restarting {svc_name} requires confirm=true (explicit confirmation).")
    elif name == "rollback_deploy":
        if svc_name in NOT_ROLLBACKABLE:
            problems.append(f"{svc_name} cannot be rolled back by version number.")
        else:
            versions = known_versions(env, svc_name)
            if args["to_version"] not in versions:
                problems.append(f"{args['to_version']} was never deployed to {svc_name}; known versions: {', '.join(versions)}.")
            elif args["to_version"] == env["services"][svc_name]["version"]:
                problems.append(f"{svc_name} is already running {args['to_version']}.")
    return problems


# --------------------------------------------------------------------------- tools
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in an operator's edit is an error, not silently ignored


class RestartArgs(_Args):
    service: ServiceName
    reason: str = Field(description="Why a restart is the right fix, citing what you observed.")
    confirm: bool = Field(default=False, description="Must be true to restart the database; set it only if the engineer explicitly asked for that.")


class ScaleArgs(_Args):
    service: ServiceName
    replicas: int = Field(description="Target number of replicas (hard limit 1 to 20; the database cannot be scaled).")
    reason: str = Field(description="Why this capacity, citing what you observed.")


class RollbackArgs(_Args):
    service: ServiceName
    to_version: str = Field(description="A version previously deployed to this service, e.g. the 'previous_version' of its latest deploy.")
    reason: str = Field(description="Why a rollback is the right fix, citing what you observed.")


ARGS_SCHEMAS: dict[str, type[_Args]] = {"restart_service": RestartArgs, "scale_service": ScaleArgs, "rollback_deploy": RollbackArgs}


@tool
def get_service_status(service: ServiceName, env: Annotated[dict, InjectedState("env")]) -> str:
    """Read the current status, CPU %, memory %, error rate %, replica count and version of one service (api, worker, db or cache)."""
    s = env["services"][service]
    return json.dumps({"service": service, **{k: s[k] for k in ("status", "cpu", "memory", "error_rate", "replicas", "version", "restarts_24h")}})


@tool
def get_error_logs(service: ServiceName, env: Annotated[dict, InjectedState("env")], limit: int = 5) -> str:
    """Read the most recent log lines (newest first) of one service. `limit` is capped at 20."""
    lines = env["logs"].get(service, [])[: max(1, min(int(limit), 20))]
    return "\n".join(lines) or "(no log lines)"


@tool
def list_recent_deploys(env: Annotated[dict, InjectedState("env")], service: ServiceName | None = None) -> str:
    """List recent deploys, newest first, with version, previous version, time, author and summary. Optionally for one service."""
    rows = [d for d in env["deploys"] if service in (None, d["service"])]
    return json.dumps(rows)


@tool(args_schema=RestartArgs)
def restart_service(service: str, reason: str, confirm: bool = False) -> str:
    """WRITE. Restart every replica of a service (brief downtime). Needs operator approval; the database also needs confirm=true."""
    raise RuntimeError("write tools run only inside approval_gate")


@tool(args_schema=ScaleArgs)
def scale_service(service: str, replicas: int, reason: str) -> str:
    """WRITE. Scale a service to a target number of replicas to add or remove capacity. Needs operator approval; hard limit 1 to 20."""
    raise RuntimeError("write tools run only inside approval_gate")


@tool(args_schema=RollbackArgs)
def rollback_deploy(service: str, to_version: str, reason: str) -> str:
    """WRITE. Roll a service back to a version that was deployed to it before. Needs operator approval."""
    raise RuntimeError("write tools run only inside approval_gate")


READ_TOOLS = [get_service_status, get_error_logs, list_recent_deploys]
WRITE_TOOL_OBJECTS = [restart_service, scale_service, rollback_deploy]


# --------------------------------------------------------------------------- state
class OpsIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made case; adds its message and sets max_steps
    messages: Annotated[list[AnyMessage], add_messages]
    max_steps: int


class OpsOut(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    env: dict
    audit: list[dict]
    agent_steps: int
    mode: str


class OpsState(OpsIn, total=False):
    env: dict
    audit: Annotated[list[dict], operator.add]
    agent_steps: int
    mode: str


# --------------------------------------------------------------------------- helpers over the message history
def last_ai_with_calls(messages: list[AnyMessage]) -> tuple[int, AIMessage | None]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return (i, messages[i]) if messages[i].tool_calls else (i, None)
    return -1, None


def pending_write_calls(messages: list[AnyMessage]) -> list[dict]:
    """Write calls of the latest AI message that no ToolMessage has answered yet."""
    idx, ai = last_ai_with_calls(messages)
    if ai is None:
        return []
    answered = {m.tool_call_id for m in messages[idx + 1 :] if isinstance(m, ToolMessage)}
    return [c for c in ai.tool_calls if c["name"] in WRITE_TOOLS and c["id"] not in answered]


def describe_call(env: dict, name: str, args: dict) -> str:
    svc = env["services"].get(args.get("service"), {})
    if name == "scale_service":
        return f"Scale {args['service']} from {svc.get('replicas')} to {args['replicas']} replica(s)"
    if name == "restart_service":
        return f"Restart {args['service']} ({svc.get('replicas')} replica(s))" + (" with confirm=true" if args.get("confirm") else "")
    return f"Roll back {args['service']} from {svc.get('version')} to {args['to_version']}"


def parse_decision(answer: Any, proposal: dict, tool: str) -> tuple[dict | None, str | None]:
    """Turn an operator's resume value into ``{"action", "args", "reason"}`` or an error message (pure function)."""
    usage = 'Resume with {"action": "approve"} | {"action": "edit", "args": {...}} | {"action": "reject", "reason": "..."}.'
    if not isinstance(answer, dict):
        return None, f"The answer must be a JSON object. {usage}"
    action = answer.get("action")
    if action == "approve":
        return {"action": "approve", "args": proposal, "reason": ""}, None
    if action == "reject":
        return {"action": "reject", "args": proposal, "reason": str(answer.get("reason") or "").strip() or "no reason given"}, None
    if action == "edit":
        edits = answer.get("args")
        if not isinstance(edits, dict) or not edits:
            return None, f"'edit' needs a non-empty 'args' object with the fields to change. {usage}"
        try:
            merged = ARGS_SCHEMAS[tool].model_validate({**proposal, **edits}).model_dump()
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return None, f"The edited arguments are invalid ({problems})."
        return {"action": "edit", "args": merged, "reason": str(answer.get("reason") or "").strip()}, None
    return None, f"Unknown action {action!r}. {usage}"


def _tool_message(call: dict, text: str, ok: bool) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call["id"], name=call["name"], status="success" if ok else "error")


# --------------------------------------------------------------------------- nodes
SYSTEM_PROMPT = (
    "You are Lumen Home's on-call operations copilot. You help an on-call engineer understand the health of four "
    "services (api, worker, db, cache) and, when asked or clearly needed, change them.\n"
    "- Reading is free: get_service_status, get_error_logs, list_recent_deploys. Look before you act and say what you "
    "found in a sentence or two.\n"
    "- Writing changes production: restart_service, scale_service, rollback_deploy. Every write is shown to the human "
    "operator, who may approve it, change its arguments or reject it. Give a precise reason for each, citing what you saw.\n"
    "- After a write, the tool result tells you what REALLY happened: the operator may have edited your arguments or "
    "rejected the call, or a safety limit may have refused it. Report that faithfully. Never claim a change the tool "
    "result does not confirm, never retry a rejected or refused call with the same arguments, and do not argue.\n"
    "- Hard limits (enforced in code whatever the operator approves): replicas 1 to 20; the database cannot be scaled "
    "or rolled back; restarting the database needs confirm=true, which you may set only if the engineer explicitly "
    "asked for a database restart; a rollback target must be a version previously deployed to that service.\n"
    "- Prefer the least invasive fix that matches the evidence: saturated CPU and timeouts -> scale; errors that start "
    "at a deploy -> roll back; restart only for stuck or leaking processes. Propose at most two writes at a time.\n"
    "- Be brief: plain sentences, no tables."
)


def bootstrap(state: OpsState) -> dict:
    """First node of every user message: create the environment once per thread, reset the step counter, and turn a
    picked ready-made case into the chat input.

    ``messages`` has the ``add_messages`` reducer, so a case's message is APPENDED to the chat. "Real input" therefore
    means "the state already has messages": a blank form on a thread with history must not inject a new opening
    message, while a picked case always does. The environment is not reset by a case (it belongs to the thread).
    """
    update: dict[str, Any] = {
        **expand(state, CASES, has_input=lambda s: bool(s.get("messages"))),
        "agent_steps": 0,
        "mode": mode_label(),
    }
    if not state.get("env"):
        update["env"] = initial_env()
    return update


def agent(state: OpsState) -> dict:
    steps, cap = state.get("agent_steps", 0), state.get("max_steps") or DEFAULT_MAX_STEPS
    if steps >= cap:  # bounded loop: stop with a fixed message (every earlier tool call has already been answered)
        return {"messages": [AIMessage(content=f"I have used my {cap} reasoning steps for this request and will stop here. Ask me to continue if you want me to keep going.")]}
    llm = get_llm().bind_tools(READ_TOOLS + WRITE_TOOL_OBJECTS)
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    # A tool call whose arguments were not valid JSON is not in ``reply.tool_calls`` but is still sent back to OpenAI
    # with the history, which then needs an answer for it too. Answer it with an error so the model can try again.
    unparsable = [
        ToolMessage(content="That tool call had invalid JSON arguments and was not run. Call it again with valid arguments.",
                    tool_call_id=c["id"], name=c.get("name") or "unknown", status="error")
        for c in reply.invalid_tool_calls
        if c.get("id")
    ]
    return {"messages": [reply, *unparsable], "agent_steps": steps + 1}


def route_after_agent(state: OpsState) -> list[Send] | str:
    """Split one AI message: read calls to the ToolNode, write calls to the gate, no calls -> done."""
    ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if ai is None:
        return END
    if not ai.tool_calls:
        return "agent" if ai.invalid_tool_calls else END  # only malformed calls: let the model try again (bounded by max_steps)
    # unknown tool names go to the ToolNode too: it answers them with an error ToolMessage
    reads = [{**c, "type": "tool_call"} for c in ai.tool_calls if c["name"] not in WRITE_TOOLS]
    if reads:
        return [Send("read_tools", reads)]
    return "approval_gate"


def route_after_reads(state: OpsState) -> str:
    return "approval_gate" if pending_write_calls(state["messages"]) else "agent"


def ask_operator(call: dict, proposal: dict, env: dict, position: int, total: int) -> dict:
    """Phase 1 of the gate: ask until the answer is valid (at most MAX_REASKS re-asks), else fail closed with a reject."""
    violations = check_limits(env, call["name"], proposal)
    payload: dict[str, Any] = {
        "kind": "approve_write_action",
        "call_id": call["id"],
        "tool": call["name"],
        "args": proposal,
        "summary": describe_call(env, call["name"], proposal),
        "service_now": {k: v for k, v in env["services"][proposal["service"]].items() if k != "cause"},
        "limit_check_preview": {"ok": not violations, "violations": violations, "note": "Advisory. The limits are enforced again after your decision."},
        "position": f"{position} of {total}",
        "allowed_actions": ["approve", "edit", "reject"],
        "resume_with": {
            "approve": {"action": "approve"},
            "edit": {"action": "edit", "args": {"<field>": "<new value>"}},
            "reject": {"action": "reject", "reason": "<why>"},
        },
    }
    for attempt in range(MAX_REASKS + 1):
        decision, error = parse_decision(interrupt(payload), proposal, call["name"])
        if decision:
            return decision
        payload = {**payload, "error": error, "attempt": attempt + 1, "attempts_left": MAX_REASKS - attempt}
    return {"action": "reject", "args": proposal, "reason": f"no valid operator answer after {MAX_REASKS + 1} attempts (fail closed)"}


def approval_gate(state: OpsState) -> dict:
    """Ask the operator about every pending write, THEN (and only then) execute what was allowed."""
    env = state["env"]
    results: dict[str, ToolMessage] = {}
    audit: list[dict] = []
    calls = pending_write_calls(state["messages"])

    # -- arguments the schema rejects never reach the operator
    askable: list[tuple[dict, dict]] = []
    for call in calls:
        try:
            askable.append((call, ARGS_SCHEMAS[call["name"]].model_validate(call["args"]).model_dump()))
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            results[call["id"]] = _tool_message(call, f"INVALID ARGUMENTS, nothing was executed: {problems}", ok=False)
            audit.append({"call_id": call["id"], "tool": call["name"], "proposed_args": call["args"], "executed_args": None, "outcome": "invalid_arguments", "detail": problems})

    # -- phase 1: collect every decision. No side effects here: this node restarts from the top on each resume.
    decisions = [ask_operator(call, proposal, env, i, len(askable)) for i, (call, proposal) in enumerate(askable, 1)]

    # -- phase 2: act. The limits are checked on the arguments that will really be used, whatever was approved.
    for (call, proposal), d in zip(askable, decisions):
        entry = {"call_id": call["id"], "tool": call["name"], "proposed_args": proposal, "operator_reason": d["reason"]}
        if d["action"] == "reject":
            results[call["id"]] = _tool_message(call, f"REJECTED by the operator ({d['reason']}). Nothing was executed. Do not retry the same call.", ok=False)
            audit.append({**entry, "executed_args": None, "outcome": "rejected", "detail": d["reason"]})
            continue
        args = d["args"]
        violations = check_limits(env, call["name"], args)
        if violations:
            detail = " ".join(violations)
            results[call["id"]] = _tool_message(call, f"REFUSED by safety limits even though the operator approved it. Nothing was executed. {detail}", ok=False)
            audit.append({**entry, "executed_args": None, "outcome": "refused_by_limits", "detail": detail})
            continue
        env, message = apply_write(env, call["name"], args)
        edited = args != proposal
        note = f" The operator edited your arguments before approving; what actually ran: {args} (you proposed {proposal})." if edited else ""
        results[call["id"]] = _tool_message(call, f"OK. {message}{note}", ok=True)
        audit.append({**entry, "executed_args": args, "outcome": "executed_edited" if edited else "executed", "detail": message})

    return {"messages": [results[c["id"]] for c in calls], "env": env, "audit": audit}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(OpsState, input_schema=OpsIn, output_schema=OpsOut)
    g.add_node("bootstrap", bootstrap)
    g.add_node("agent", agent)
    g.add_node("read_tools", ToolNode(READ_TOOLS, handle_tool_errors=True))
    g.add_node("approval_gate", approval_gate)
    g.add_edge(START, "bootstrap")
    g.add_edge("bootstrap", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["read_tools", "approval_gate", "agent", END])
    g.add_conditional_edges("read_tools", route_after_reads, ["approval_gate", "agent"])
    g.add_edge("approval_gate", "agent")
    return g.compile(**compile_kwargs)


graph = build_graph()

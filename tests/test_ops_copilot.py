from __future__ import annotations

import copy

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepdive.common.fakes import set_chat_handler
from deepdive.common.testing import check_example, run_graph
from deepdive.examples import ops_copilot as ex


# --------------------------------------------------------------------------- helpers
def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def ai(*calls, text=""):
    return AIMessage(content=text, tool_calls=list(calls))


def script(*replies):
    """Chat handler returning ``replies`` in order (the last repeats); returns the list of prompts the model was shown."""
    prompts: list[list] = []

    def handler(messages, tools):
        prompts.append(list(messages))
        reply = replies[min(len(prompts) - 1, len(replies) - 1)]
        return reply.model_copy(update={"id": None})  # fresh message each time: add_messages merges by id

    set_chat_handler(handler)
    return prompts


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def make():
    return ex.build_graph(checkpointer=InMemorySaver())


def thread(name="t1"):
    return {"configurable": {"thread_id": name}, "recursion_limit": 60}


def tool_messages(messages):
    return {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}


def assert_valid_history(messages):
    """OpenAI's rule: every tool call of an assistant message is answered by exactly one tool message, before the next turn."""
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and (m.tool_calls or m.invalid_tool_calls):
            following = []
            for later in messages[i + 1 :]:
                if not isinstance(later, ToolMessage):
                    break
                following.append(later.tool_call_id)
            expected = [c["id"] for c in m.tool_calls] + [c["id"] for c in m.invalid_tool_calls]
            assert sorted(following) == sorted(expected), (expected, following)


def replicas(res_or_values, service="api"):
    values = res_or_values.output if hasattr(res_or_values, "output") else res_or_values
    return values["env"]["services"][service]["replicas"]


# --------------------------------------------------------------------------- contract and structure
def test_example_contract_and_the_example_input_reaches_the_gate_and_applies_the_edit():
    res = check_example(ex)
    assert res.interrupts and res.interrupts[0]["tool"] == "scale_service"
    assert replicas(res) == 4, "EXAMPLE_RESUMES edits replicas to 4"
    assert_valid_history(res.output["messages"])


def test_the_model_sees_clean_tool_schemas_and_write_tools_cannot_run_outside_the_gate():
    schema = convert_to_openai_tool(ex.get_service_status)["function"]["parameters"]
    assert set(schema["properties"]) == {"service"}, "the injected env must be invisible to the model"
    assert set(convert_to_openai_tool(ex.scale_service)["function"]["parameters"]["properties"]) == {"service", "replicas", "reason"}
    with pytest.raises(RuntimeError, match="approval_gate"):
        ex.scale_service.invoke({"service": "api", "replicas": 2, "reason": "x"})
    assert not set(ex.WRITE_TOOLS) & {t.name for t in ex.READ_TOOLS}


def test_apply_write_is_pure_and_check_limits_is_a_function_of_the_environment():
    env = ex.initial_env()
    frozen = copy.deepcopy(env)
    new_env, message = ex.apply_write(env, "scale_service", {"service": "api", "replicas": 4, "reason": "x"})
    assert env == frozen and new_env["services"]["api"]["replicas"] == 4
    assert new_env["services"]["api"]["cpu"] == 70 and new_env["services"]["api"]["status"] == "healthy"
    assert "replica" in message


# --------------------------------------------------------------------------- reads never interrupt
def test_read_only_conversation_never_interrupts_and_does_not_touch_the_environment():
    prompts = script(
        ai(call("get_service_status", {"service": "api"}, "r1"), call("list_recent_deploys", {"service": "worker"}, "r2")),
        AIMessage("The api is degraded: cpu 93%."),
    )
    res = run_graph(make(), user("how is the api?"))
    assert res.interrupts == []
    msgs = res.output["messages"]
    answers = tool_messages(msgs)
    assert '"cpu": 93' in answers["r1"].content and "D-7731" in answers["r2"].content
    assert res.output["env"] == ex.initial_env()
    assert res.output.get("audit", []) == []
    assert msgs[-1].content.startswith("The api is degraded")
    assert len(prompts) == 2 and isinstance(prompts[0][0], SystemMessage)
    assert_valid_history(msgs)


def test_a_tool_call_with_unparsable_arguments_is_answered_and_the_model_gets_to_retry():
    bad = AIMessage(content="", invalid_tool_calls=[{"name": "scale_service", "args": '{"service": "api", "replicas":', "id": "bad1", "error": "bad json", "type": "invalid_tool_call"}])
    prompts = script(bad, AIMessage("Sorry, I could not form that call."))
    res = run_graph(make(), user("scale the api"))
    answer = tool_messages(res.output["messages"])["bad1"]
    assert answer.status == "error" and "invalid JSON" in answer.content
    assert len(prompts) == 2, "the loop went back to the model instead of ending"
    assert res.output["messages"][-1].content.startswith("Sorry")
    assert res.interrupts == [] and res.output["env"] == ex.initial_env()
    assert_valid_history(res.output["messages"])


def test_unknown_tool_names_get_an_error_tool_message_instead_of_breaking_the_history():
    script(ai(call("format_disk", {}, "x1")), AIMessage("I cannot do that."))
    res = run_graph(make(), user("wipe it"))
    answer = tool_messages(res.output["messages"])["x1"]
    assert answer.status == "error" and "not a valid tool" in answer.content
    assert res.interrupts == []
    assert_valid_history(res.output["messages"])


# --------------------------------------------------------------------------- writes pause for a human
def test_a_write_interrupts_with_the_call_and_nothing_changes_until_the_operator_answers():
    script(ai(call("scale_service", {"service": "api", "replicas": 10, "reason": "cpu 93%"}, "w1")), AIMessage("done"))
    graph, cfg = make(), thread()
    graph.invoke(user("scale the api"), cfg)
    snap = graph.get_state(cfg)
    assert snap.next == ("approval_gate",)
    ask = snap.tasks[0].interrupts[0].value
    assert ask["kind"] == "approve_write_action" and ask["tool"] == "scale_service" and ask["call_id"] == "w1"
    assert ask["args"] == {"service": "api", "replicas": 10, "reason": "cpu 93%"}
    assert ask["summary"] == "Scale api from 3 to 10 replica(s)"
    assert ask["limit_check_preview"]["ok"] is True
    assert "cause" not in ask["service_now"] and ask["service_now"]["replicas"] == 3
    assert ask["allowed_actions"] == ["approve", "edit", "reject"]
    assert snap.values["env"]["services"]["api"]["replicas"] == 3, "no change before approval"
    assert "w1" not in tool_messages(snap.values["messages"])


def test_approve_runs_the_call_exactly_as_proposed_and_tells_the_model():
    prompts = script(ai(call("scale_service", {"service": "api", "replicas": 10, "reason": "cpu 93%"}, "w1")), AIMessage("Scaled."))
    res = run_graph(make(), user("scale the api"), [{"action": "approve"}])
    assert replicas(res) == 10
    answer = tool_messages(res.output["messages"])["w1"]
    assert answer.status == "success" and answer.content.startswith("OK. Scaled api from 3 to 10")
    assert "edited" not in answer.content
    assert [a["outcome"] for a in res.output["audit"]] == ["executed"]
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "w1" for m in prompts[1]), "the model saw the result"
    assert_valid_history(res.output["messages"])


def test_edit_changes_the_arguments_that_actually_run_and_the_model_is_told():
    prompts = script(ai(call("scale_service", {"service": "api", "replicas": 10, "reason": "cpu 93%"}, "w1")), AIMessage("Scaled to 4."))
    res = run_graph(make(), user("scale the api"), [{"action": "edit", "args": {"replicas": 4}}])
    assert replicas(res) == 4
    answer = tool_messages(res.output["messages"])["w1"]
    assert "edited your arguments" in answer.content and "'replicas': 4" in answer.content and "'replicas': 10" in answer.content
    entry = res.output["audit"][0]
    assert entry["outcome"] == "executed_edited"
    assert entry["proposed_args"]["replicas"] == 10 and entry["executed_args"]["replicas"] == 4
    assert entry["executed_args"]["service"] == "api" and entry["executed_args"]["reason"] == "cpu 93%", "an edit is merged over the proposal"
    assert "edited your arguments" in next(m for m in prompts[1] if isinstance(m, ToolMessage)).content


def test_reject_executes_nothing_and_yields_a_rejection_tool_message_the_model_can_read():
    prompts = script(ai(call("rollback_deploy", {"service": "worker", "to_version": "2026.11.2", "reason": "KeyError since 11.4"}, "w1")), AIMessage("Understood, waiting."))
    res = run_graph(make(), user("fix the worker"), [{"action": "reject", "reason": "wait for dana"}])
    assert res.output["env"] == ex.initial_env()
    answer = tool_messages(res.output["messages"])["w1"]
    assert answer.status == "error" and "REJECTED" in answer.content and "wait for dana" in answer.content
    assert "wait for dana" in next(m for m in prompts[1] if isinstance(m, ToolMessage)).content
    assert res.output["audit"][0]["outcome"] == "rejected"
    assert_valid_history(res.output["messages"])


# --------------------------------------------------------------------------- hard limits hold after approval
@pytest.mark.parametrize(
    ("tool", "args", "fragment"),
    [
        ("scale_service", {"service": "api", "replicas": 50, "reason": "x"}, "between 1 and 20"),
        ("scale_service", {"service": "api", "replicas": 21, "reason": "x"}, "between 1 and 20"),
        ("scale_service", {"service": "api", "replicas": 0, "reason": "x"}, "between 1 and 20"),
        ("scale_service", {"service": "db", "replicas": 2, "reason": "x"}, "cannot be scaled"),
        ("restart_service", {"service": "db", "reason": "x"}, "requires confirm=true"),
        ("rollback_deploy", {"service": "worker", "to_version": "9.9.9", "reason": "x"}, "never deployed"),
        ("rollback_deploy", {"service": "worker", "to_version": "2026.11.4", "reason": "x"}, "already running"),
        ("rollback_deploy", {"service": "db", "to_version": "pg-15.3", "reason": "x"}, "cannot be rolled back"),
    ],
)
def test_limits_are_enforced_after_approval_and_the_environment_is_untouched(tool, args, fragment):
    script(ai(call(tool, args, "w1")), AIMessage("ok"))
    res = run_graph(make(), user("do it"), [{"action": "approve"}])
    assert res.interrupts[0]["limit_check_preview"]["ok"] is False, "the operator is warned in advance"
    answer = tool_messages(res.output["messages"])["w1"]
    assert answer.status == "error" and "REFUSED by safety limits" in answer.content and fragment in answer.content
    assert res.output["env"] == ex.initial_env()
    assert res.output["audit"][0]["outcome"] == "refused_by_limits"
    assert_valid_history(res.output["messages"])


@pytest.mark.parametrize(
    ("tool", "args", "service", "key", "expected"),
    [
        ("scale_service", {"service": "api", "replicas": 20, "reason": "x"}, "api", "replicas", 20),
        ("scale_service", {"service": "api", "replicas": 1, "reason": "x"}, "api", "replicas", 1),
        ("restart_service", {"service": "db", "reason": "x", "confirm": True}, "db", "restarts_24h", 1),
        ("restart_service", {"service": "cache", "reason": "x"}, "cache", "restarts_24h", 1),
        ("rollback_deploy", {"service": "worker", "to_version": "2026.11.2", "reason": "x"}, "worker", "version", "2026.11.2"),
    ],
)
def test_calls_at_the_edge_of_the_limits_are_allowed(tool, args, service, key, expected):
    script(ai(call(tool, args, "w1")), AIMessage("ok"))
    res = run_graph(make(), user("do it"), [{"action": "approve"}])
    assert res.output["env"]["services"][service][key] == expected
    assert res.interrupts[0]["limit_check_preview"]["ok"] is True


def test_an_edit_cannot_smuggle_a_call_past_the_limits_and_an_edit_can_supply_the_db_confirmation():
    script(ai(call("scale_service", {"service": "api", "replicas": 5, "reason": "x"}, "w1")), AIMessage("ok"))
    res = run_graph(make(), user("scale"), [{"action": "edit", "args": {"replicas": 99}}])
    assert replicas(res) == 3 and res.output["audit"][0]["outcome"] == "refused_by_limits"

    script(ai(call("restart_service", {"service": "db", "reason": "stuck connections"}, "w2")), AIMessage("ok"))
    res = run_graph(make(), user("restart the database"), [{"action": "edit", "args": {"confirm": True}}])
    assert res.output["audit"][0]["outcome"] == "executed_edited"
    assert res.output["env"]["services"]["db"]["restarts_24h"] == 1


def test_a_rollback_really_fixes_a_bad_deploy_and_a_scale_really_relieves_saturation():
    script(
        ai(call("rollback_deploy", {"service": "worker", "to_version": "2026.11.2", "reason": "KeyError"}, "w1"), call("scale_service", {"service": "api", "replicas": 6, "reason": "cpu"}, "w2")),
        AIMessage("both done"),
    )
    res = run_graph(make(), user("fix both"), [{"action": "approve"}, {"action": "approve"}])
    services = res.output["env"]["services"]
    assert services["worker"]["error_rate"] == 0.4 and services["worker"]["status"] == "healthy"
    assert services["api"]["error_rate"] == 0.6 and services["api"]["status"] == "healthy"
    assert res.output["env"]["deploys"][0]["service"] == "worker" and res.output["env"]["deploys"][0]["version"] == "2026.11.2"


# --------------------------------------------------------------------------- mixed and multiple calls
def test_mixed_message_runs_the_reads_first_and_gates_only_the_write():
    script(
        ai(call("get_error_logs", {"service": "worker", "limit": 2}, "r1"), call("restart_service", {"service": "cache", "reason": "stale"}, "w1"), call("get_service_status", {"service": "cache"}, "r2")),
        AIMessage("Logs shown; cache restarted."),
    )
    graph, cfg = make(), thread()
    graph.invoke(user("show worker logs and restart the cache"), cfg)
    paused = graph.get_state(cfg)
    answered = tool_messages(paused.values["messages"])
    assert set(answered) == {"r1", "r2"}, "both reads ran before the engineer was asked about the write"
    assert "KeyError" in answered["r1"].content and answered["r1"].content.count("\n") == 1
    assert paused.tasks[0].interrupts[0].value["tool"] == "restart_service"
    assert paused.values["env"]["services"]["cache"]["restarts_24h"] == 0

    final = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert set(tool_messages(final["messages"])) == {"r1", "r2", "w1"}
    assert final["env"]["services"]["cache"]["restarts_24h"] == 1
    assert_valid_history(final["messages"])


def test_two_writes_in_one_message_are_asked_one_at_a_time_and_nothing_runs_until_both_are_answered(monkeypatch):
    applied: list[str] = []
    real = ex.apply_write

    def counting(env, name, args):
        applied.append(f"{name}:{args['service']}")
        return real(env, name, args)

    monkeypatch.setattr(ex, "apply_write", counting)
    script(
        ai(call("scale_service", {"service": "api", "replicas": 5, "reason": "cpu"}, "w1"), call("restart_service", {"service": "worker", "reason": "stuck"}, "w2")),
        AIMessage("ok"),
    )
    graph, cfg = make(), thread()
    first = graph.invoke(user("do both"), cfg)
    assert first["__interrupt__"][0].value["position"] == "1 of 2"
    second = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert second["__interrupt__"][0].value["position"] == "2 of 2" and second["__interrupt__"][0].value["tool"] == "restart_service"
    assert applied == [], "approving call 1 must not execute it while call 2 is still being asked (the node restarts on resume)"
    final = graph.invoke(Command(resume={"action": "reject", "reason": "not the worker"}), cfg)
    assert applied == ["scale_service:api"], "executed exactly once, after every answer was in"
    assert replicas(final) == 5 and final["env"]["services"]["worker"]["restarts_24h"] == 0
    answers = tool_messages(final["messages"])
    assert answers["w1"].status == "success" and answers["w2"].status == "error"
    assert [m.tool_call_id for m in final["messages"] if isinstance(m, ToolMessage)] == ["w1", "w2"]


# --------------------------------------------------------------------------- invalid input, loops
@pytest.mark.parametrize(
    ("answer", "needle"),
    [
        ("approve", "JSON object"),
        ({"action": "yes"}, "Unknown action"),
        ({"action": "edit"}, "non-empty 'args'"),
        ({"action": "edit", "args": {}}, "non-empty 'args'"),
        ({"action": "edit", "args": {"replicas": "many"}}, "invalid"),
        ({"action": "edit", "args": {"replica": 4}}, "invalid"),  # typo: extra fields are forbidden, not ignored
    ],
)
def test_parse_decision_rejects_malformed_answers(answer, needle):
    decision, error = ex.parse_decision(answer, {"service": "api", "replicas": 10, "reason": "x"}, "scale_service")
    assert decision is None and needle in error


def test_invalid_operator_answers_are_re_asked_twice_then_the_call_is_rejected_fail_closed():
    script(ai(call("scale_service", {"service": "api", "replicas": 10, "reason": "cpu"}, "w1")), AIMessage("ok"))
    res = run_graph(make(), user("scale"), ["yes", {"action": "edit"}, {"action": "edit", "args": {"replicas": "many"}}])
    assert len(res.interrupts) == ex.MAX_REASKS + 1 == 3
    assert "error" not in res.interrupts[0] and "JSON object" in res.interrupts[1]["error"] and res.interrupts[2]["attempts_left"] == 1
    assert replicas(res) == 3
    answer = tool_messages(res.output["messages"])["w1"]
    assert "REJECTED" in answer.content and "fail closed" in answer.content


def test_arguments_the_schema_rejects_never_reach_the_operator():
    script(
        ai(call("scale_service", {"service": "api", "replicas": "lots", "reason": "x"}, "w1"), call("restart_service", {"service": "cache", "reason": "x", "force": True}, "w2")),
        AIMessage("sorry"),
    )
    res = run_graph(make(), user("scale"))
    assert res.interrupts == []
    answers = tool_messages(res.output["messages"])
    assert answers["w1"].status == "error" and answers["w1"].content.startswith("INVALID ARGUMENTS")
    assert "force" in answers["w2"].content
    assert [e["outcome"] for e in res.output["audit"]] == ["invalid_arguments"] * 2
    assert res.output["env"] == ex.initial_env()


def test_the_agent_loop_is_bounded_by_max_steps_and_every_tool_call_is_still_answered():
    prompts = script(ai(call("get_service_status", {"service": "api"}, "r")))  # a model that never stops calling tools
    res = run_graph(make(), {**user("loop forever"), "max_steps": 3})
    assert len(prompts) == 3, "exactly max_steps model calls, then the loop stops"
    last = res.output["messages"][-1]
    assert isinstance(last, AIMessage) and not last.tool_calls and "3 reasoning steps" in last.content
    assert res.output["agent_steps"] == 3
    # ids repeat in this scripted model, so check the shape instead: every AI tool call is followed by its answer
    kinds = [type(m).__name__ for m in res.output["messages"]]
    assert kinds.count("AIMessage") == 4 and kinds.count("ToolMessage") == 3


def test_default_step_cap_applies_when_none_is_given():
    prompts = script(ai(call("get_service_status", {"service": "db"}, "r")))
    run_graph(make(), user("loop"))
    assert len(prompts) == ex.DEFAULT_MAX_STEPS


# --------------------------------------------------------------------------- state: persistence and isolation
def test_the_environment_persists_across_turns_and_is_isolated_between_threads():
    graph = make()
    script(ai(call("scale_service", {"service": "api", "replicas": 4, "reason": "cpu"}, "w1")), AIMessage("scaled"))
    a = thread("alice")
    graph.invoke(user("scale api to 4"), a)
    after = graph.invoke(Command(resume={"action": "approve"}), a)
    assert replicas(after) == 4

    script(ai(call("get_service_status", {"service": "api"}, "r1")), AIMessage("status"))
    second_turn = graph.invoke(user("and now?"), a)
    assert '"replicas": 4' in tool_messages(second_turn["messages"])["r1"].content, "turn 2 sees the change made in turn 1"
    assert second_turn["agent_steps"] == 2, "the step counter restarts for each user message"

    script(ai(call("get_service_status", {"service": "api"}, "r1")), AIMessage("status"))
    other = graph.invoke(user("status?"), thread("bob"))
    assert '"replicas": 3' in tool_messages(other["messages"])["r1"].content, "another thread has its own environment"
    assert replicas(other) == 3

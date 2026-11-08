from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import plan_execute_analyst as ex

THERM = "Lumen Therm"


def step(tool, why="because", **kw):
    return ex.PlanStep(tool=tool, why=why, **kw)


def plan(*steps):
    return ex.Plan(steps=list(steps))


def replan(decision, *new_steps, reasoning="r"):
    return ex.Replan(decision=decision, reasoning=reasoning, new_steps=list(new_steps))


CONTINUE, FINISH = replan("continue"), replan("finish")


class Spy:
    """Scripted structured handler that remembers every prompt it received."""

    def __init__(self, *values):
        self.next = scripted(values)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.next(prompt)


def conclusion_model(text="The answer is yes [1]."):
    prompts = []

    def chat(messages, tools):
        prompts.append(messages[-1].content)
        return AIMessage(text)

    chat.prompts = prompts  # type: ignore[attr-defined]
    return chat


def completed():
    df = read_csv("sales.csv")
    df["d"] = pd.to_datetime(df["order_date"])
    return df[df["status"] != "cancelled"]


def run(plan_value, replans, question="Did the Therm price cut work?", conclusion=None, **extra):
    planner = Spy(plan_value)
    replanner = Spy(*replans)
    chat = conclusion or conclusion_model()
    set_structured_handler(ex.Plan, planner)
    set_structured_handler(ex.Replan, replanner)
    set_chat_handler(chat)
    out = ex.graph.invoke({"question": question, **extra})
    return out, planner, replanner, chat


def test_contract_and_offline_run():
    out = check_example(ex).output
    assert out["stop_reason"] in ("finished", "plan_complete", "max_steps") and len(out["steps"]) <= ex.DEFAULT_MAX_STEPS


# --------------------------------------------------------------------------- happy path
def test_a_three_step_plan_executes_in_order_with_exact_numbers():
    d = completed()
    t = d[d["product"] == THERM]
    before = t[(t["d"] >= "2026-02-01") & (t["d"] <= "2026-04-30")]
    after = t[(t["d"] >= "2026-05-01") & (t["d"] <= "2026-07-31")]
    p = plan(
        step("revenue_summary", product="therm", period="2026-02..2026-04"),
        step("revenue_summary", product="therm", period="2026-05..2026-07"),
        step("calculate", expression="(s2.units - s1.units) / s1.units * 100"),
    )
    conclusion = conclusion_model("Units rose about 47% [3], revenue changed [2].")
    out, planner, replanner, chat = run(p, [CONTINUE, CONTINUE, FINISH], conclusion=conclusion)

    assert [s["tool"] for s in out["steps"]] == ["revenue_summary", "revenue_summary", "calculate"]
    assert [s["n"] for s in out["steps"]] == [1, 2, 3] and all(s["ok"] for s in out["steps"])
    s1, s2, s3 = out["steps"]
    assert s1["numbers"]["units"] == before["quantity"].sum() and s2["numbers"]["units"] == after["quantity"].sum()
    assert s1["numbers"]["revenue"] == pytest.approx(before["revenue"].sum(), abs=0.01)
    assert s1["args"]["product"] == "therm", "arguments are stored exactly as planned; the tool canonicalises them"
    assert s3["numbers"]["value"] == pytest.approx((after["quantity"].sum() - before["quantity"].sum()) / before["quantity"].sum() * 100, abs=1e-4)
    assert out["stop_reason"] == "finished" and out["revisions"] == 0
    assert len(replanner.prompts) == 3, "the replanner is consulted after every step"
    assert out["conclusion"] == "Units rose about 47% [3], revenue changed [2]." and out["invalid_citations"] == []
    assert "[3] calculate(" in conclusion.prompts[0] and "[1] revenue_summary(product=therm" in conclusion.prompts[0]
    assert out["mode"] == "offline"


def test_each_replan_prompt_shows_results_so_far_and_the_remaining_plan():
    p = plan(step("revenue_summary", product=THERM, period="2026Q1"), step("revenue_summary", product=THERM, period="2026Q2"))
    _, planner, replanner, _ = run(p, [CONTINUE, FINISH])
    first, second = replanner.prompts
    assert "[1] revenue_summary(product=Lumen Therm, period=2026Q1) -> Lumen Therm" in first
    assert "STEPS STILL PLANNED\n[2] revenue_summary(product=Lumen Therm, period=2026Q2)" in first
    assert "5 tool calls left" in first and "Revisions left: 2" in first
    assert "STEPS STILL PLANNED\n(none)" in second and "4 tool calls left" in second
    assert "revenue_summary" in planner.prompts[0]
    assert "Today is 2026-10-31" in planner.prompts[0] and "'Last quarter' = 2026Q3" in planner.prompts[0]


# --------------------------------------------------------------------------- replanning
def test_revise_remaining_replaces_the_rest_of_the_plan_once():
    p = plan(
        step("revenue_summary", product=THERM, period="2026Q1"),
        step("top_n", group_by="region", metric="units"),  # the original step 2: must never run
    )
    revised = replan("revise_remaining", step("compare_periods", metric="units", period="2026Q1", period_b="2026Q2", product=THERM),
                     step("calculate", expression="s2.pct_change * 2"))
    out, _, replanner, _ = run(p, [revised, CONTINUE, FINISH])
    assert [s["tool"] for s in out["steps"]] == ["revenue_summary", "compare_periods", "calculate"], "the original step 2 was replaced"
    assert out["revisions"] == 1 and out["stop_reason"] == "finished"
    assert out["steps"][2]["ok"] and out["steps"][2]["numbers"]["value"] == pytest.approx(out["steps"][1]["numbers"]["pct_change"] * 2)
    assert "The first new step will be step 2" in replanner.prompts[0]


def test_revisions_are_capped_and_the_step_budget_always_wins():
    p = plan(step("revenue_summary", period="2026Q1"), step("revenue_summary", period="2026Q2"), step("revenue_summary", period="2026Q3"))
    again = replan("revise_remaining", *[step("revenue_summary", period=f"2026-0{m}") for m in (1, 2, 3)])
    out, _, replanner, _ = run(p, [again], max_steps=5)
    assert out["revisions"] == ex.MAX_REVISIONS == 2, "a third rewrite request is ignored"
    assert len(out["steps"]) == 5 and out["stop_reason"] == "max_steps"
    assert len(replanner.prompts) == 4, "no replanner call once the budget is spent"


def test_max_steps_truncates_the_plan_and_stops_the_run():
    p = plan(*[step("revenue_summary", period=f"2026-0{m}") for m in range(1, 7)])
    out, _, replanner, _ = run(p, [CONTINUE], max_steps=2)
    assert [s["args"]["period"] for s in out["steps"]] == ["2026-01", "2026-02"]
    assert out["stop_reason"] == "max_steps" and len(replanner.prompts) == 1
    assert out["conclusion"], "the run still ends with a conclusion"


def test_finish_early_leaves_planned_steps_unrun():
    p = plan(step("revenue_summary", period="2026Q1"), step("revenue_summary", period="2026Q2"), step("revenue_summary", period="2026Q3"))
    out, *_ = run(p, [FINISH])
    assert len(out["steps"]) == 1 and out["stop_reason"] == "finished"


def test_continue_with_an_empty_plan_ends_as_plan_complete():
    out, *_ = run(plan(step("revenue_summary", period="2026Q1")), [CONTINUE])
    assert len(out["steps"]) == 1 and out["stop_reason"] == "plan_complete"


def test_empty_plan_goes_straight_to_the_conclusion():
    out, _, replanner, chat = run(plan(), [CONTINUE])
    assert out["steps"] == [] and out["stop_reason"] == "no_plan" and replanner.prompts == []
    assert len(chat.prompts) == 1


# --------------------------------------------------------------------------- errors become observations
def test_bad_product_becomes_an_observation_the_replanner_can_fix():
    p = plan(step("revenue_summary", product="Lumen Thermm", period="2026Q2"))
    fix = replan("revise_remaining", step("revenue_summary", product=THERM, period="2026Q2"))
    out, _, replanner, chat = run(p, [fix, FINISH])
    bad, good = out["steps"]
    assert bad["ok"] is False and "Did you mean 'Lumen Therm'" in bad["error"] and bad["numbers"] == {}
    assert good["ok"] is True and good["numbers"]["units"] > 0
    assert "ERROR: Unknown product 'Lumen Thermm'" in replanner.prompts[0], "the error text reaches the model"
    assert "Valid product values" in replanner.prompts[0]
    assert out["stop_reason"] == "finished" and out["conclusion"]
    assert "ERROR: Unknown product" in chat.prompts[0]


@pytest.mark.parametrize(
    "bad_step,fragment",
    [
        (step("revenue_summary", period="2026-13-45"), "not a real calendar date"),
        (step("revenue_summary", period="next tuesday"), "Could not read period"),
        (step("revenue_summary", period="2024Q1"), "outside the data"),
        (step("revenue_summary", region="Mars"), "Valid region values"),
        (step("compare_periods", metric="units", period="2026Q1"), "period_b is required"),
        (step("compare_periods", metric="units", period="2026Q2", period_b="2026Q1", group_by="product", region="Nowhere"), "Unknown region"),
        (step("top_n", metric="units"), "needs group_by"),
        (step("top_n", group_by="product", top_n=0), "top_n must be"),
        (step("metric_by_period", metric="units", interval="quarter", product="Lumen Lock", region="APAC", period="2026-05-02..2026-05-03"), "No data"),
        (step("calculate"), "needs an expression"),
        (step("calculate", expression="s9.value + 1"), "step 9, which has not run"),
        (step("calculate", expression="1 / 0"), "Division by zero"),
        (step("calculate", expression="2 +"), "not a valid arithmetic"),
        (step("calculate", expression="__import__('os').system('echo hi')"), "only arithmetic"),
        (step("calculate", expression="open('/etc/passwd').read()"), "only arithmetic"),
        (step("calculate", expression="9 ** 999"), "exponent too large"),
    ],
)
def test_every_kind_of_bad_argument_is_an_error_observation_not_a_crash(bad_step, fragment):
    out, *_ = run(plan(bad_step), [FINISH])
    record = out["steps"][0]
    assert record["ok"] is False and fragment in record["error"], record["error"]
    assert out["conclusion"] and out["stop_reason"] == "finished"


def test_calculate_refuses_to_use_a_failed_step_and_names_the_available_numbers():
    p = plan(step("revenue_summary", product="Nope"), step("calculate", expression="s1.revenue * 2"),
             step("revenue_summary", period="2026Q2"), step("calculate", expression="s3.profit"))
    out, *_ = run(p, [CONTINUE] * 4)
    assert "Step 1 failed" in out["steps"][1]["error"]
    assert "no number called 'profit'" in out["steps"][3]["error"] and "revenue" in out["steps"][3]["error"]


def test_run_step_is_a_closed_registry_and_never_raises():
    assert set(ex.TOOLS) == set(ex.TOOL_NAMES)
    unknown = ex.run_step({"tool": "os.system", "args": {"expression": "ls"}}, [])
    assert unknown["ok"] is False and "revenue_summary" in unknown["error"] and "os.system" in unknown["error"]
    assert ex.run_step({"tool": "__import__", "args": {}}, [])["ok"] is False
    original = ex.TOOLS["top_n"]
    ex.TOOLS["top_n"] = lambda a, d: 1 / 0  # simulate a bug inside a tool
    try:
        crash = ex.run_step({"tool": "top_n", "args": {}}, [])
    finally:
        ex.TOOLS["top_n"] = original
    assert crash["ok"] is False and "ZeroDivisionError" in crash["error"]


# --------------------------------------------------------------------------- the tools themselves match pandas
def test_tools_return_pandas_exact_numbers():
    d = completed()
    q2 = d[(d["d"] >= "2026-04-01") & (d["d"] <= "2026-06-30")]
    r = ex.run_step({"tool": "revenue_summary", "args": {"period": "2026Q2"}}, [])
    assert r["ok"] and r["numbers"]["revenue"] == pytest.approx(q2["revenue"].sum(), abs=0.01) and r["numbers"]["orders"] == q2["order_id"].nunique()
    top = ex.run_step({"tool": "top_n", "args": {"group_by": "region", "metric": "revenue", "period": "2026Q2", "top_n": 2}}, [])
    by_region = q2.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert top["numbers"]["rank1"] == pytest.approx(by_region.iloc[0], abs=0.01) and top["numbers"]["rank2"] == pytest.approx(by_region.iloc[1], abs=0.01)
    assert "rank3" not in top["numbers"] and f"1. {by_region.index[0]}" in top["text"]
    series = ex.run_step({"tool": "metric_by_period", "args": {"metric": "units", "product": "Therm", "interval": "month"}}, [])
    monthly = d[d["product"] == THERM].groupby(d["d"].dt.strftime("%Y-%m"))["quantity"].sum()
    assert series["numbers"]["count"] == len(monthly) and series["numbers"]["total"] == monthly.sum() and series["numbers"]["max"] == monthly.max()
    assert "2026-05: 14" in series["text"]
    cmp_ = ex.run_step({"tool": "compare_periods", "args": {"metric": "revenue", "period": "2026Q1", "period_b": "2026Q2", "group_by": "product"}}, [])
    assert cmp_["numbers"]["change"] == pytest.approx(cmp_["numbers"]["b"] - cmp_["numbers"]["a"], abs=0.01) and "different lengths" in cmp_["text"]


def test_calculate_substitutes_negative_numbers_safely():
    done = [{"n": 1, "tool": "compare_periods", "ok": True, "numbers": {"change": -12.5, "a": 50.0}}]
    r = ex.run_step({"tool": "calculate", "args": {"expression": "s1.change / s1.a * 100 - -s1.change"}}, done)
    assert r["ok"] and r["numbers"]["value"] == pytest.approx(-37.5), "(-12.5 / 50 * 100) - (-(-12.5))"


# --------------------------------------------------------------------------- citations, inputs, reruns
def test_citations_must_point_at_steps_that_exist_and_succeeded():
    p = plan(step("revenue_summary", product="Nope"), step("revenue_summary", period="2026Q2"))
    out, *_ = run(p, [CONTINUE, FINISH], conclusion=conclusion_model("Revenue was X [2] but see also [1] and [7]."))
    assert out["invalid_citations"] == [1, 7]
    assert ex.check_citations("fine [1][2]", [{"n": 1, "ok": True}, {"n": 2, "ok": True}]) == []


def test_inputs_are_validated():
    for bad, msg in (({}, "Provide a question"), ({"question": "x" * 700}, "too long"), ({"question": "q", "max_steps": 0}, "max_steps must be"), ({"question": "q", "max_steps": 11}, "max_steps must be")):
        with pytest.raises(ValueError, match=msg):
            ex.graph.invoke(bad)


def test_state_is_reset_when_a_thread_is_reused():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    set_structured_handler(ex.Plan, scripted([plan(step("revenue_summary", period="2026Q2"))]))
    set_structured_handler(ex.Replan, scripted([FINISH]))
    set_chat_handler(conclusion_model())
    first = g.invoke({"question": "q1"}, cfg)
    second = g.invoke({"question": "q2"}, cfg)
    assert len(first["steps"]) == len(second["steps"]) == 1 and [s["n"] for s in second["steps"]] == [1]


def test_null_like_text_in_unused_fields_means_no_filter():
    queue = ex.to_queue([step("revenue_summary", product="N/A", region="all", period="2026Q2", expression="", metric=None)])
    assert queue == [{"tool": "revenue_summary", "why": "because", "args": {"period": "2026Q2"}}]
    out, *_ = run(plan(step("revenue_summary", product="none", region=" ", period="2026Q2")), [FINISH])
    assert out["steps"][0]["ok"] and out["steps"][0]["args"] == {"period": "2026Q2"}

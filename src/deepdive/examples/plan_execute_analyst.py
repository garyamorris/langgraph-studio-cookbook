"""Plan-and-execute analyst for multi-step business questions.

PROBLEM
    "Did the Therm price cut on 2026-05-01 increase units enough to offset the lower price?" cannot be answered with
    one query. It needs dependent steps (units before, units after, revenue before and after, then arithmetic) and a
    comparison that is fair about period length. A free-roaming ReAct agent tends to wander: it repeats queries,
    compares unequal periods, and does the final arithmetic in its head. Analysts need the opposite: a short plan they
    can read, exact numbers from a fixed set of tools, and a clear stop.

DESIGN
    The model writes the plan and the conclusion; code does everything in between.

      * make_plan      the LLM writes an ordered plan of at most ``max_steps`` steps. Every step names ONE tool from a
                       closed registry (revenue_summary, metric_by_period, compare_periods, top_n, calculate) and gives
                       its arguments as flat typed fields (a Literal tool name, a few strings, one int). There is no
                       free-form dict and no code: the model chooses arguments, never writes logic.
      * execute_step   code runs the next step through ``_sales_tools`` (pandas) and stores its text and numbers. A
                       step that raises (unknown product, impossible date, a reference to a step that failed) does NOT
                       crash the run: the error becomes the step's result, written for the model to read.
      * replan         after every step the LLM sees everything so far and chooses ``continue`` (keep the plan),
                       ``revise_remaining`` (replace all remaining steps, for example to correct a bad argument) or
                       ``finish``. Code enforces the limits: a hard ``max_steps`` budget and at most two revisions.
      * conclude       the LLM writes the answer, citing step results by number: "units rose 46.9% [2]". Code checks
                       that every citation points at a step that exists and succeeded.

    Arithmetic is a tool (``calculate``) that reads named numbers from earlier steps (``s2.pct_change``) and evaluates
    only + - * / ** % // with the same safe evaluator as the shared calculator: no ``eval``.

    Why plan-then-execute and not ReAct? The plan is visible before any query runs, each step is independently
    inspectable in Studio, the tool set is closed, and the budget is a number. The replanner keeps it from being rigid:
    it is the only place the plan can change, and it can change only the remaining steps.

GRAPH
    START -> make_plan --(plan has steps)--> execute_step -> replan --(continue / revise_remaining)--> execute_step
                 |                                             |
                 |  (empty plan)                               |  (finish, or step budget spent)
                 +-----------------------> conclude <---------+
                                              |
                                             END

LANGGRAPH FEATURES
    * plan-as-state: ``pending`` (steps still to run) and ``steps`` (steps run, with results) are ordinary state keys,
      so Studio shows the plan shrinking and the evidence growing, and you can edit them with update_state
    * a loop with a conditional edge and explicit counters (``max_steps``, ``revisions``) instead of relying on the
      recursion limit
    * a closed tool registry executed by code, with tool errors turned into observations
    * structured output with ``Literal`` fields (strict-mode friendly) for plan and decisions; input/output schemas

MODELS AND DATA
    OpenAI chat model: structured output for the plan and the replan decisions, plain text for the conclusion. Data:
    ``sales.csv`` (see ``_sales_tools`` for the exact metric definitions). Offline, the stand-in planner fills
    arguments with placeholder text, so you will see failed steps being fed back as observations and the run still
    finishing within its budget; run it live for a real analysis. Not covered: the model can still choose a poor
    comparison window (the notes returned by ``compare_periods`` warn about unequal lengths); a human reviewer should
    read the plan, which is why it is a first-class state key.

TRY IT IN STUDIO
    Open the ``Ready-made case`` dropdown, pick one and press Submit (or leave the form blank on a new thread: that runs
    the first one). Open ``make_plan`` to read the plan, then step through ``execute_step`` / ``replan`` and watch
    ``steps`` fill. Each case sets the question and the step budget:
      * ``Did the Therm price cut on 2026-05-01 offset the lower price?`` - a sound run compares equal windows either
        side of the cut (for example Feb-Apr with May-Jul) and finds units up about 47% (32 -> 47) and revenue up about
        44%, with a caveat that summer demand may share the credit.
      * ``Which region had the highest return rate for Cam Pro last quarter?`` (3 steps) - APAC leads at 20%, but on
        only 10 order lines; a good conclusion says the sample is thin.
      * ``Top 2 products by margin last quarter, and their share of total margin?`` - a ``top_n`` step feeds a
        ``calculate`` step through named numbers (``s1.rank1``).
      * ``What was customer satisfaction in Q2?`` - the sales data cannot answer it; check that the conclusion says so.
      * ``How did the Lumen Doorbell sell last quarter?`` (2 steps) - there is no such product (it is ``Lumen Bell``):
        the tool error becomes an observation, and the replanner can fix the step within the budget.
      * ``Break down 2026 revenue by region, channel and product`` (2 steps) - needs more steps than the budget; watch
        the plan being cut and ``stop_reason`` become ``max_steps``.
    To ask your own, paste ``{"question": "Did the Therm price cut on 2026-05-01 increase units enough to offset the
    lower price?", "max_steps": 6}``.

EXTEND IT
    Add a tool that reads ``product_reviews.csv`` so the analyst can look at sentiment around the price cut. Add an
    ``interrupt()`` after ``make_plan`` so a person approves the plan before anything runs. Let ``replan`` use a
    Decisions API ``predicate`` ("do the results already answer the question?") instead of a free-text decision.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.common.scenarios import expand, scenario_field
from deepdive.common.tools import safe_eval
from deepdive.examples import _sales_tools as st

META = ExampleMeta(
    id="plan_execute_analyst",
    title="Plan, execute, replan: analyst for multi-step questions",
    category="Agents & tools",
    problem="Answer dependent, multi-step sales questions with exact numbers by planning against a closed tool "
    "registry, executing deterministically, replanning on observations, and stopping at a hard step budget.",
    features=["plan-as-state", "loop with replanning and hard caps", "closed tool registry", "tool errors as observations", "structured output"],
    models=["OpenAI chat"],
    datasets=["sales.csv"],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"question": "Did the Therm price cut on 2026-05-01 increase units enough to offset the lower price?"}

# Ready-made questions for Studio's dropdown (see deepdive.common.scenarios). The first one runs on a blank form. Every
# case sets the same two keys (None = the default budget). The numbers in the labels come from data/sales.csv and are
# pinned by tests/test_plan_execute_analyst.py; what the planner does with each question is only ever "watch for".
CASES: dict[str, dict] = {
    "Did the Therm price cut on 2026-05-01 offset the lower price? -> Feb-Apr vs May-Jul: units 32 -> 47, revenue +44%": {
        "question": "Did the Therm price cut on 2026-05-01 increase units enough to offset the lower price?",
        "max_steps": None},
    "Which region had the highest return rate for Cam Pro last quarter? (3 steps) -> APAC 20% on just 10 lines: thin sample": {
        "question": "Which region had the highest return rate for Cam Pro last quarter?",
        "max_steps": 3},
    "Top 2 products by margin last quarter, and their share of total margin? -> Therm and Hub: watch top_n feed calculate": {
        "question": "Which two products had the highest margin last quarter, and what share of total margin do they make up?",
        "max_steps": None},
    "What was customer satisfaction in Q2? -> the sales data has no such field: does the conclusion say so?": {
        "question": "What was customer satisfaction in Q2?",
        "max_steps": None},
    "How did the Lumen Doorbell sell last quarter? (2 steps) -> no such product ('Lumen Bell'): watch the error and replan": {
        "question": "How did the Lumen Doorbell sell last quarter?",
        "max_steps": 2},
    "Break down 2026 revenue by region, channel and product (2 steps) -> needs 3 steps: watch the budget cut it short": {
        "question": "Break down 2026 revenue by region, by channel and by product, and tell me which of the three is the weakest.",
        "max_steps": 2},
}
SCENARIO = scenario_field(CASES)

DEFAULT_MAX_STEPS = 6
MAX_STEPS_LIMIT = 10
MAX_REVISIONS = 2  # how many times the replanner may rewrite the remaining plan
MAX_QUESTION_CHARS = 600
TOOL_NAMES = ("revenue_summary", "metric_by_period", "compare_periods", "top_n", "calculate")
ALL_METRICS = list(st.METRICS)


# --------------------------------------------------------------------------- schemas
class PlanStep(BaseModel):
    """One tool call. Only fill the fields the chosen tool uses; leave the rest null."""

    tool: Literal["revenue_summary", "metric_by_period", "compare_periods", "top_n", "calculate"]
    why: str = Field(description="One sentence: what this step contributes to answering the question")
    metric: Literal["revenue", "units", "orders", "margin", "return_rate", "avg_discount"] | None = Field(
        default=None, description="metric_by_period, compare_periods, top_n (default revenue)"
    )
    group_by: Literal["product", "category", "region", "segment", "channel", "month", "quarter", "status"] | None = Field(
        default=None, description="top_n: what to rank (required). compare_periods: optional breakdown"
    )
    interval: Literal["month", "quarter"] | None = Field(default=None, description="metric_by_period: bucket size (default month)")
    product: str | None = Field(default=None, description="Filter, e.g. 'Lumen Therm'")
    region: str | None = Field(default=None, description="Filter, e.g. 'Europe'")
    period: str | None = Field(
        default=None,
        description="Period A / the period to analyse: 2026Q2, 2026-05, 2026, 2026-05-01, or a range 2026-02..2026-04 or 2026-04-01..2026-06-30. Null = all data",
    )
    period_b: str | None = Field(default=None, description="compare_periods only: period B, same formats; the change is B vs A")
    top_n: int | None = Field(default=None, description="top_n only: how many rows to keep (default 3)")
    expression: str | None = Field(
        default=None, description="calculate only: arithmetic over earlier results, e.g. '(s2.b - s1.b) / s1.b * 100'. Refer to a step number n and a number name as s<n>.<name>"
    )


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="The ordered steps, fewest that can answer the question")


class Replan(BaseModel):
    decision: Literal["continue", "revise_remaining", "finish"]
    reasoning: str = Field(description="One or two sentences")
    new_steps: list[PlanStep] = Field(
        description="Only for revise_remaining: the COMPLETE list of steps to run from now on (they replace every remaining step). Otherwise empty"
    )


class AnalystIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made question (and budget); replaces the two fields below
    question: str
    max_steps: int  # tool-call budget, default 6, at most 10


class AnalystOut(TypedDict):
    question: str
    conclusion: str
    steps: list[dict]
    stop_reason: str
    revisions: int
    invalid_citations: list[int]
    mode: str


class AnalystState(AnalystIn, total=False):
    pending: list[dict]  # steps still to run: {"tool", "why", "args"}
    steps: list[dict]  # steps already run, with their result or error
    revisions: int
    decision: str
    stop_reason: str
    conclusion: str
    invalid_citations: list[int]
    mode: str


# --------------------------------------------------------------------------- the closed tool registry
Tool = Callable[[dict[str, Any], list[dict]], dict[str, Any]]


def _clean(numbers: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in numbers.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _scope(a: dict) -> str:
    return ", ".join(x for x in (a.get("product"), a.get("region")) if x) or "all products and regions"


def _metric(a: dict) -> str:
    m = a.get("metric") or "revenue"
    if m not in st.METRICS:
        raise ValueError(f"Unsupported metric {m!r}; choose from {', '.join(st.METRICS)}.")
    return m


def tool_revenue_summary(a: dict, done: list[dict]) -> dict:
    """All six metrics for one scope and period."""
    r = st.query_sales(metrics=ALL_METRICS, product=a.get("product"), region=a.get("region"), period=a.get("period"))
    t, w = r["total"], r["window"]
    text = (
        f"{_scope(a)}, {w['start']} to {w['end']} ({r['lines']} order lines): "
        + ", ".join(f"{m} {st.format_value(m, t[m])}" for m in ALL_METRICS)
        + (". Notes: " + " ".join(r["notes"]) if r["notes"] else ".")
    )
    return {"text": text, "numbers": _clean({**t, "lines": r["lines"]})}


def tool_metric_by_period(a: dict, done: list[dict]) -> dict:
    """One metric per month or quarter, plus its summary statistics."""
    metric, interval = _metric(a), a.get("interval") or "month"
    if interval not in st.TIME_GROUPS:
        raise ValueError("interval must be 'month' or 'quarter'.")
    r = st.query_sales(metrics=[metric], group_by=interval, product=a.get("product"), region=a.get("region"), period=a.get("period"))
    values = [row[metric] for row in r["rows"] if row[metric] is not None]
    if not values:
        raise ValueError("No data for that product, region and period.")
    series = "; ".join(f"{row['group']}: {st.format_value(metric, row[metric])}" for row in r["rows"])
    text = f"{metric} per {interval}, {_scope(a)}, {r['window']['start']} to {r['window']['end']}: {series}. Overall {st.format_value(metric, r['total'][metric])}."
    nums = {"total": r["total"][metric], "mean": sum(values) / len(values), "min": min(values), "max": max(values), "first": values[0], "last": values[-1], "count": len(values)}
    return {"text": text, "numbers": _clean(nums)}


def tool_compare_periods(a: dict, done: list[dict]) -> dict:
    """One metric in period A and period B; B vs A."""
    r = st.compare_periods(
        metric=_metric(a), period_a=a.get("period") or "", period_b=a.get("period_b") or "",
        group_by=a.get("group_by"), product=a.get("product"), region=a.get("region"),
    )
    m = r["metric"]
    text = (
        f"{m} for {_scope(a)}: period A {r['period_a']['start']}..{r['period_a']['end']} ({r['period_a']['days']} days) = {st.format_value(m, r['a'])}; "
        f"period B {r['period_b']['start']}..{r['period_b']['end']} ({r['period_b']['days']} days) = {st.format_value(m, r['b'])}; "
        f"change B vs A = {r['change']} {r['unit']} ({'n/a' if r['pct_change'] is None else str(r['pct_change']) + '%'} relative)."
    )
    if r.get("groups"):
        text += " By " + a["group_by"] + ": " + "; ".join(f"{g['group']} {g['a']} -> {g['b']}" for g in r["groups"][:8]) + "."
    if r["notes"]:
        text += " Notes: " + " ".join(r["notes"])
    return {"text": text, "numbers": _clean({k: r[k] for k in ("a", "b", "change", "pct_change")})}


def tool_top_n(a: dict, done: list[dict]) -> dict:
    """Rank groups by one metric."""
    if not a.get("group_by"):
        raise ValueError("top_n needs group_by (what to rank), e.g. 'product' or 'region'.")
    metric, n = _metric(a), 3 if a.get("top_n") is None else a["top_n"]
    r = st.query_sales(
        metrics=[metric], group_by=a["group_by"], product=a.get("product"), region=a.get("region"),
        period=a.get("period"), top_n=int(n), sort_by=metric,
    )
    rows = r["rows"]
    ranking = "; ".join(f"{i}. {row['group']} {st.format_value(metric, row[metric])}" for i, row in enumerate(rows, 1))
    text = f"Top {len(rows)} by {metric} per {a['group_by']}, {_scope(a)}, {r['window']['start']} to {r['window']['end']}: {ranking}. All groups together: {st.format_value(metric, r['total'][metric])}."
    nums: dict[str, Any] = {f"rank{i}": row[metric] for i, row in enumerate(rows[:5], 1)}
    nums["total"] = r["total"][metric]
    return {"text": text, "numbers": _clean(nums)}


_REF = re.compile(r"\bs(\d+)\.([a-z_][a-z0-9_]*)\b")


def tool_calculate(a: dict, done: list[dict]) -> dict:
    """Arithmetic over named numbers from earlier steps."""
    expression = (a.get("expression") or "").strip()
    if not expression:
        raise ValueError("calculate needs an expression, for example '(s2.b - s1.b) / s1.b * 100'.")

    def resolve(m: re.Match[str]) -> str:
        n, key = int(m.group(1)), m.group(2)
        record = next((d for d in done if d["n"] == n), None)
        if record is None:
            raise ValueError(f"The expression uses step {n}, which has not run. Steps run so far: {[d['n'] for d in done] or 'none'}.")
        if not record["ok"]:
            raise ValueError(f"Step {n} failed, so s{n}.{key} has no value. Repeat that step with corrected arguments first.")
        if key not in record["numbers"]:
            raise ValueError(f"Step {n} ({record['tool']}) has no number called '{key}'. It has: {', '.join(record['numbers'])}.")
        return f"({record['numbers'][key]!r})"

    resolved = _REF.sub(resolve, expression)
    try:
        value = safe_eval(resolved)
    except ZeroDivisionError:
        raise ValueError(f"Division by zero in '{expression}'.") from None
    except SyntaxError:
        raise ValueError(f"'{expression}' is not a valid arithmetic expression.") from None
    return {"text": f"{expression} = {round(value, 4)}", "numbers": {"value": float(round(value, 6))}}


TOOLS: dict[str, Tool] = {
    "revenue_summary": tool_revenue_summary,
    "metric_by_period": tool_metric_by_period,
    "compare_periods": tool_compare_periods,
    "top_n": tool_top_n,
    "calculate": tool_calculate,
}
assert set(TOOLS) == set(TOOL_NAMES)

TOOL_GUIDE = """TOOLS (use only these; leave unused fields null)
1. revenue_summary(product?, region?, period?) -> all six metrics for the scope. Numbers: revenue, units, orders, margin, return_rate, avg_discount, lines.
2. metric_by_period(metric, interval=month|quarter, product?, region?, period?) -> one value per month/quarter. Numbers: total, mean, min, max, first, last, count.
3. compare_periods(metric, period, period_b, group_by?, product?, region?) -> period A vs period B. Numbers: a, b, change, pct_change. Prefer periods of equal length.
4. top_n(group_by, metric, top_n=3, product?, region?, period?) -> ranking. Numbers: rank1, rank2, ..., total.
5. calculate(expression) -> arithmetic over earlier results. Write s<step>.<number>, e.g. (s2.b - s1.b) / s1.b * 100. Never do arithmetic yourself."""


def run_step(step: dict, done: list[dict]) -> dict:
    """Run one plan step. Never raises: failures come back as ``{"ok": False, "error": ...}`` for the replanner."""
    name = step.get("tool")
    fn = TOOLS.get(name)  # a closed dict lookup: nothing outside TOOLS can be reached
    if fn is None:
        return {"ok": False, "error": f"Unknown tool {name!r}. The only tools are: {', '.join(TOOL_NAMES)}."}
    try:
        out = fn(step.get("args", {}), done)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - a bug in one step must not kill the run; surface it as an observation
        return {"ok": False, "error": f"Unexpected {type(e).__name__}: {e}"}
    return {"ok": True, "text": out["text"], "numbers": out["numbers"]}


# --------------------------------------------------------------------------- helpers (pure, unit-tested)
NULLISH = {"", "null", "none", "n/a", "na", "all", "any"}  # what a model writes in a text field when it means "no filter"


def to_queue(steps: list[PlanStep] | list[dict]) -> list[dict]:
    """PlanStep models -> state dicts ``{"tool", "why", "args": {fields that carry a value}}``."""
    queue = []
    for s in steps:
        d = s.model_dump() if isinstance(s, PlanStep) else dict(s)
        args = {
            k: v
            for k, v in d.items()
            if k not in ("tool", "why") and v is not None and not (isinstance(v, str) and v.strip().lower() in NULLISH)
        }
        queue.append({"tool": d["tool"], "why": d.get("why", ""), "args": args})
    return queue


def show_call(step: dict) -> str:
    return f"{step['tool']}({', '.join(f'{k}={v}' for k, v in step['args'].items())})"


def show_result(record: dict) -> str:
    return record["text"] if record["ok"] else f"ERROR: {record['error']}"


def show_steps(done: list[dict]) -> str:
    return "\n".join(f"[{d['n']}] {show_call(d)} -> {show_result(d)}" for d in done) or "(no steps run yet)"


def check_citations(text: str, steps: list[dict]) -> list[int]:
    """Step numbers cited as [n] that do not exist or did not succeed."""
    ok = {d["n"] for d in steps if d["ok"]}
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)} - ok)


# --------------------------------------------------------------------------- nodes
PLANNER_PROMPT = (
    "You are the planning step of a sales-data analyst for Lumen Home. Break the user's question into the FEWEST "
    "steps that can answer it, at most {max_steps}, each calling exactly one tool.\n\n{brief}\n\n{tools}\n\n"
    "Steps are numbered 1, 2, 3 ... in the order they run. Rules: resolve relative dates yourself ('last quarter', 'the price cut') into explicit periods. For before/after "
    "questions compare periods of EQUAL length on either side of the event and avoid November-December, which is "
    "seasonally high. Judge a price change by revenue and margin as well as units. Put arithmetic in a calculate "
    "step, never in your head. If the data cannot answer the question, plan the one step that shows what the data "
    "does contain. Do not ask questions."
)


def make_plan(state: AnalystState) -> dict:
    # This is the entry node, so it first resolves a picked ready-made question (or runs the first one on a blank form).
    # A typed ``max_steps`` with no question is real input too: it gets the "Provide a question" error, not a case.
    picked = expand(
        state, CASES, has_input=lambda s: bool((s.get("question") or "").strip()) or s.get("max_steps") is not None
    )
    state = {**state, **picked}
    question = (state.get("question") or "").strip()
    if not question:
        raise ValueError("Provide a question, e.g. 'Which region had the highest return rate for Cam Pro last quarter?'")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"The question is too long ({len(question)} characters; the limit is {MAX_QUESTION_CHARS}).")
    cap = DEFAULT_MAX_STEPS if state.get("max_steps") is None else state["max_steps"]
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= MAX_STEPS_LIMIT:
        raise ValueError(f"max_steps must be a whole number between 1 and {MAX_STEPS_LIMIT}; got {cap!r}.")
    plan = structured_llm(Plan).invoke(
        [SystemMessage(PLANNER_PROMPT.format(max_steps=cap, brief=st.dataset_brief(), tools=TOOL_GUIDE)), HumanMessage(question)]
    )
    pending = to_queue(plan.steps)[:cap]
    return {
        **picked,
        "question": question,
        "max_steps": cap,
        "pending": pending,
        "steps": [],
        "revisions": 0,
        "decision": "continue" if pending else "finish",
        "stop_reason": "" if pending else "no_plan",
        "conclusion": "",
        "invalid_citations": [],
    }


def execute_step(state: AnalystState) -> dict:
    """Run the next pending step through the registry and record the outcome."""
    step, rest = state["pending"][0], state["pending"][1:]
    done = state["steps"]
    record = {"n": len(done) + 1, "tool": step["tool"], "why": step["why"], "args": step["args"], "numbers": {}, **run_step(step, done)}
    return {"pending": rest, "steps": done + [record]}


REPLAN_PROMPT = (
    "You supervise a sales-data analyst that works through a plan one step at a time. After each step decide:\n"
    "- continue: the next planned step is still the right thing to do;\n"
    "- revise_remaining: a step failed (fix its arguments using the error text, which lists valid values), the "
    "results show a different analysis is needed, or something essential is missing. Return the COMPLETE list of "
    "steps to run from now on in new_steps; they replace every remaining step. The first new step will be step {next_n}.\n"
    "- finish: the results already answer the question, or more steps cannot help. Finishing with a partial answer "
    "is fine; the conclusion will say what is unknown.\n"
    "Budget: {left} tool calls left (a failed call still counts). Revisions left: {revisions_left}. Never plan more "
    "steps than the budget allows.\n\n{brief}\n\n{tools}"
)


def replan(state: AnalystState) -> dict:
    """After each step: continue, revise the remaining steps, or finish. Hard limits are enforced here in code."""
    done, pending = state["steps"], state["pending"]
    left = state["max_steps"] - len(done)
    if left <= 0:
        return {"decision": "finish", "stop_reason": "max_steps"}  # budget spent: no model call needed
    revisions_left = MAX_REVISIONS - state["revisions"]
    remaining = "\n".join(f"[{len(done) + i}] {show_call(p)} ({p['why']})" for i, p in enumerate(pending, 1)) or "(none)"
    r = structured_llm(Replan, "fast").invoke(
        [
            SystemMessage(REPLAN_PROMPT.format(next_n=len(done) + 1, left=left, revisions_left=max(0, revisions_left), brief=st.dataset_brief(), tools=TOOL_GUIDE)),
            HumanMessage(f"Question: {state['question']}\n\nSTEPS RUN\n{show_steps(done)}\n\nSTEPS STILL PLANNED\n{remaining}"),
        ]
    )
    if r.decision == "finish":
        return {"decision": "finish", "stop_reason": "finished"}
    if r.decision == "revise_remaining" and revisions_left > 0:
        revised = to_queue(r.new_steps)[:left]
        if not revised:
            return {"decision": "finish", "stop_reason": "finished"}
        return {"decision": "revised", "pending": revised, "revisions": state["revisions"] + 1}
    if not pending:  # asked to continue (or to revise past the limit) with nothing left to run
        return {"decision": "finish", "stop_reason": "plan_complete"}
    return {"decision": "continue"}


def after_plan(state: AnalystState) -> Literal["execute_step", "conclude"]:
    return "execute_step" if state["pending"] else "conclude"


def after_replan(state: AnalystState) -> Literal["execute_step", "conclude"]:
    return "conclude" if state["decision"] == "finish" or not state["pending"] else "execute_step"


CONCLUDE_PROMPT = (
    "You write the final answer of a sales-data analysis for a business reader, in under 150 words. Start with a direct "
    "answer (yes / no / partly / cannot tell). Support it with figures taken ONLY from the numbered step results, and "
    "cite the step in square brackets right after each figure, like 'units rose 46.9% [2]'. Never compute or estimate "
    "a number yourself. Mention caveats the steps reveal (unequal period lengths, seasonality, small samples). If a step "
    "failed or the run stopped early ({stop}), say precisely what could not be established. Do not cite failed steps."
)


def conclude(state: AnalystState) -> dict:
    stop = state.get("stop_reason") or "finished"
    msg = get_llm().invoke(
        [
            SystemMessage(CONCLUDE_PROMPT.format(stop=stop)),
            HumanMessage(f"Question: {state['question']}\n\nSTEP RESULTS\n{show_steps(state['steps'])}"),
        ]
    )
    text = str(msg.text).strip()
    return {"conclusion": text, "invalid_citations": check_citations(text, state["steps"]), "stop_reason": stop, "mode": mode_label()}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(AnalystState, input_schema=AnalystIn, output_schema=AnalystOut)
    g.add_node("make_plan", make_plan)
    g.add_node("execute_step", execute_step)
    g.add_node("replan", replan)
    g.add_node("conclude", conclude)
    g.add_edge(START, "make_plan")
    g.add_conditional_edges("make_plan", after_plan, ["execute_step", "conclude"])
    g.add_edge("execute_step", "replan")
    g.add_conditional_edges("replan", after_replan, ["execute_step", "conclude"])
    g.add_edge("conclude", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

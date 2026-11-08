"""Quarterly sales report whose sections are planned from what the data shows.

PROBLEM
    A finance lead wants a quarterly sales report. What deserves a section depends on the quarter: 2026Q2 has the
    Lumen Cam Pro recall (returns need their own section) and the Lumen Therm price cut (pricing does), while an
    uneventful quarter needs neither. A fixed template either misses the story or pads quiet quarters. And a report
    that a language model has written from memory is worthless to finance: every figure has to come from the data.

DESIGN
    Facts first, prose last, and a check in between that does not trust the prose.

      1. gather_facts      deterministic pandas (via ``_sales_tools``): KPIs, revenue / units / return rate by product,
                           region and channel, change against the like-for-like previous quarter, price changes, and
                           z-score anomaly flags (a product's average month this quarter, for revenue or return
                           rate, against the same product's other months; |z| >= 2, at most 5 flags). Every fact has
                           a stable key and a one-line text with the numbers already computed, so no model does
                           arithmetic.
      2. plan_sections     an LLM planner reads the fact catalog and picks 3 to 6 sections; each section names the
                           fact keys it needs. Unknown keys are discarded in code. Two code guards: if the planner
                           returns fewer than 3 usable sections, default ones fill in; and any flagged anomaly the
                           planner did not cover gets an "Other flagged changes" section, so the planner cannot
                           silently drop the recall.
      3. write_section     one worker per section, started with ``Send``. Each worker receives ONLY the facts its
                           section asked for, which keeps prompts small and makes "this number came from these facts"
                           checkable.
      4. synthesize        puts the sections back in planned order and writes the executive summary.
      5. verify_numbers    deterministic: every number in each narrative must appear in the facts that were supplied
                           (within the precision it was written with). Anything else is reported in the output and at
                           the end of the report. The run does not fail: the finance lead sees the report AND the list
                           of figures to distrust.
      6. render_report     assembles the markdown.

    Why the previous quarter is "like-for-like": the data begins on 2025-11-01 and ends on 2026-10-31, so 2025Q4 and
    2026Q4 are partial. Comparing October with the whole of Q3 would show a fake collapse of roughly 70%, so a
    partial quarter is compared with the same days of the prior quarter, and a quarter whose predecessor is not fully
    covered (2025Q4, 2026Q1) says so instead of inventing a change.

GRAPH
    START -> gather_facts -> plan_sections --(one Send per section)--> write_section x N (parallel)
                                                                              |
                  END <- render_report <- verify_numbers <- synthesize <------+

LANGGRAPH FEATURES
    * orchestrator-workers: a conditional edge returns one ``Send("write_section", payload)`` per planned section,
      so the number of parallel workers is decided at run time by the model's plan
    * a worker payload that is NOT the graph state (each worker sees only its own facts)
    * an ``operator.add`` reducer collects the workers' drafts; ``synthesize`` sorts them by planned index because
      the order in which parallel results are merged is an implementation detail you should not rely on
    * ``Overwrite([])`` resets the reducer list at the start of each run, so a second run on the same Studio thread
      does not append to the first
    * input/output schemas; deterministic nodes around the LLM nodes

MODELS AND DATA
    OpenAI chat model: structured output for the plan, plain text for sections and the summary. Data: ``sales.csv``.
    Offline, the stand-in planner returns keys that do not exist, so the default three-section plan is used and the
    "narratives" are placeholder text; the facts, the section fan-out, the ordering and the verification are real.
    Limits: the number check catches figures that are not in the facts, not wrong words ("rose" for "fell"), and it
    ignores years and whole numbers up to 10. Spelled-out numbers are not checked.

TRY IT IN STUDIO
    Input ``{"quarter": "2026Q2"}``. Open ``gather_facts`` to see the fact catalog: ``anomaly.lumen_cam_pro.return_rate``
    and ``pricing.lumen_therm`` are the two stories. Open ``plan_sections`` to see which sections the planner chose,
    then watch the graph fan out to one ``write_section`` run per section. Compare with ``{"quarter": "2026Q3"}``
    (no anomaly, a quieter plan) or ``{"quarter": "2026Q4"}`` (October only: compared with July, labelled partial).
    Check ``verification`` in the output for numbers that did not come from the facts.

EXTEND IT
    Let a worker call ``_sales_tools.query_sales`` for a drill-down the facts do not contain (and add those results to
    the verification set). Add a human approval step before the report is published. Run the planner twice and keep
    the sections both runs agree on.
"""

from __future__ import annotations

import math
import operator
import re
from datetime import timedelta
from typing import Annotated

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.examples import _sales_tools as st
from deepdive.examples._sales_tools import money, pct

META = ExampleMeta(
    id="report_orchestrator",
    title="Quarterly report with planned sections",
    category="Core workflows",
    problem="Write a quarterly sales report whose sections are chosen from what the data shows, with every number "
    "traced back to computed facts and unsupported figures flagged.",
    features=["Send (orchestrator-workers)", "operator.add reducer + order restoration", "Overwrite reset", "post-hoc verification node", "input/output schemas"],
    models=["OpenAI chat"],
    datasets=["sales.csv"],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"quarter": "2026Q2"}

MIN_SECTIONS, MAX_SECTIONS = 3, 6
Z_THRESHOLD = 2.0  # |z| at or above this flags a product-month pattern as unusual
MAX_ANOMALIES = 5
MIN_LINES_FOR_RATE_ANOMALY = 8  # a return rate over fewer order lines is noise
ALL_METRICS = ["revenue", "units", "orders", "margin", "return_rate", "avg_discount"]


# --------------------------------------------------------------------------- schemas
class SectionPlan(BaseModel):
    """One section the planner wants in the report."""

    title: str = Field(description="Short section title, e.g. 'Cam Pro returns'")
    purpose: str = Field(description="One sentence: the question this section answers for the finance lead")
    fact_keys: list[str] = Field(description="Keys from the fact catalog that this section needs, copied exactly")


class ReportPlan(BaseModel):
    sections: list[SectionPlan] = Field(description="3 to 6 sections, in reading order")


class ReportIn(TypedDict, total=False):
    quarter: str  # e.g. "2026Q2"


class ReportOut(TypedDict):
    quarter: str
    plan: list[dict]
    sections: list[dict]
    executive_summary: str
    verification: dict
    flags: list[str]
    report: str
    mode: str


class ReportState(ReportIn, total=False):
    facts: dict[str, dict]
    plan: list[dict]
    section_drafts: Annotated[list[dict], operator.add]  # one entry appended per worker
    sections: list[dict]
    executive_summary: str
    verification: dict
    flags: list[str]
    report: str
    mode: str


# --------------------------------------------------------------------------- formatting
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def change_text(cur: float | None, prev: float | None, *, rate: bool = False) -> str:
    """'+9.8%' for amounts, '+4.2 points' for percentage metrics, 'n/a' when there is nothing to compare."""
    if cur is None or prev is None:
        return "n/a"
    if rate:
        return f"{cur - prev:+.1f} points"
    if prev == 0:
        return "n/a"
    return f"{(cur - prev) / prev * 100:+.1f}%"


# --------------------------------------------------------------------------- fact gathering (deterministic)
def _agg(group_by: str | None, start: str, end: str) -> dict:
    return st.query_sales(metrics=ALL_METRICS, group_by=group_by, start_date=start, end_date=end)


def _fact(key: str, kind: str, title: str, text: str, data: dict | None = None) -> dict:
    return {"key": key, "kind": kind, "title": title, "text": text, "data": data or {}}


def _price_facts(q: st.Quarter, w: dict) -> list[dict]:
    """A fact for every product whose list price changed inside the previous quarter or this one."""
    df = st.load_sales()
    df = df[df["status"] != "cancelled"]
    lo, hi = w["prior_start"], w["end"]
    out = []
    for product, g in df.groupby("product"):
        daily = g.groupby("date")["unit_price"].first().sort_index()
        changed = daily[daily.diff().fillna(0) != 0]
        for ts, new_price in changed.items():
            day = ts.date()
            if not lo <= day <= hi:
                continue
            old_price = float(daily[daily.index < ts].iloc[-1])
            if w["start"] <= day <= w["end"]:
                inside = g[(g["date"] >= pd.Timestamp(w["start"])) & (g["date"] <= pd.Timestamp(w["end"]))]
                before, after = inside[inside["date"] < ts], inside[inside["date"] >= ts]
                d_before, d_after = (day - w["start"]).days, (w["end"] - day).days + 1
                parts = [f"{product} list price changed from {money(old_price)} to {money(float(new_price))}; the first order at the new price was on {day} (inside {q.label})."]
                if d_before > 0 and d_after > 0:
                    parts.append(
                        f"Before the change ({w['start']} to {day - timedelta(days=1)}, {d_before} days): "
                        f"{int(before['quantity'].sum())} units, {before['quantity'].sum() / d_before:.2f} units per day, "
                        f"{money(float(before['revenue'].sum()) / d_before)} revenue per day. "
                        f"After ({day} to {w['end']}, {d_after} days): {int(after['quantity'].sum())} units, "
                        f"{after['quantity'].sum() / d_after:.2f} units per day, {money(float(after['revenue'].sum()) / d_after)} revenue per day."
                    )
                where = "inside"
            else:
                parts = [
                    f"{product} list price changed from {money(old_price)} to {money(float(new_price))}; the first order at the new price was "
                    f"on {day}, in the previous quarter, so {q.label} is sold entirely at {money(float(new_price))}."
                ]
                where = "previous"
            out.append(_fact(f"pricing.{slug(product)}", "pricing", f"{product} price change", " ".join(parts),
                             {"product": product, "date": day.isoformat(), "old": old_price, "new": float(new_price), "when": where}))
    return out


def _anomaly_facts(q: st.Quarter, w: dict) -> list[dict]:
    """z-score flags: this quarter's average month vs the same product's other months (revenue and return rate)."""
    df = st.load_sales()
    df = df[df["status"] != "cancelled"]
    in_q = df[(df["date"] >= pd.Timestamp(w["start"])) & (df["date"] <= pd.Timestamp(w["end"]))]
    q_months = set(in_q["month"])
    found = []
    for product, g in df.groupby("product"):
        monthly = g.groupby("month").agg(
            revenue=("revenue", "sum"),
            return_rate=("status", lambda s: float((s == "returned").mean() * 100)),
            lines=("order_id", "size"),
        )
        inside, base = monthly[monthly.index.isin(q_months)], monthly[~monthly.index.isin(q_months)]
        if inside.empty or len(base) < 3:
            continue
        for metric, label, fmt in (("revenue", "monthly revenue", money), ("return_rate", "monthly return rate", pct)):
            sd = float(base[metric].std(ddof=1))
            if not sd or math.isnan(sd):
                continue
            if metric == "return_rate" and int(inside["lines"].sum()) < MIN_LINES_FOR_RATE_ANOMALY:
                continue
            z = (float(inside[metric].mean()) - float(base[metric].mean())) / sd
            if abs(z) >= Z_THRESHOLD:
                text = (
                    f"UNUSUAL: {product} {label} averaged {fmt(float(inside[metric].mean()))} in {q.label}, against "
                    f"{fmt(float(base[metric].mean()))} across its other months (z-score {z:+.1f}, "
                    f"{'above' if z > 0 else 'below'} normal)."
                )
                found.append((abs(z), _fact(f"anomaly.{slug(product)}.{metric}", "anomaly", f"{product} {label} anomaly", text,
                                            {"product": product, "metric": metric, "z": round(z, 2)})))
    found.sort(key=lambda t: -t[0])
    return [f for _, f in found[:MAX_ANOMALIES]]


def build_facts(quarter: str) -> dict[str, dict]:
    """Everything the report may say, as keyed one-line facts with the numbers already computed."""
    q = st.parse_quarter(quarter)
    w = st.quarter_windows(q)  # raises ValueError when the data has nothing for this quarter
    start, end = w["start"].isoformat(), w["end"].isoformat()
    cur = _agg(None, start, end)
    prior_ok = w["prior_comparable"]
    p_start, p_end = w["prior_start"].isoformat(), w["prior_end"].isoformat()
    prior = _agg(None, p_start, p_end) if prior_ok else None
    cur_by = {g: _agg(g, start, end) for g in ("product", "region", "channel")}
    prior_by = {g: _agg(g, p_start, p_end) for g in ("product", "region")} if prior_ok else {}

    facts: dict[str, dict] = {}
    t = cur["total"]
    coverage = f"full quarter, {w['days']} days" if not w["partial"] else f"partial quarter: the data covers {w['days']} of {w['quarter_days']} days"
    margin_pct = t["margin"] / t["revenue"] * 100 if t["revenue"] else None
    text = (
        f"{q.label} ({start} to {end}; {coverage}): revenue {money(t['revenue'])}, units {t['units']}, orders {t['orders']}, "
        f"margin {money(t['margin'])} ({pct(margin_pct)} of revenue), return rate {pct(t['return_rate'])}, average discount {pct(t['avg_discount'])}."
    )
    if prior:
        pt = prior["total"]
        text += (
            f" Compared with {w['prior']} ({p_start} to {p_end}, same days): revenue {change_text(t['revenue'], pt['revenue'])} "
            f"(from {money(pt['revenue'])}), units {change_text(t['units'], pt['units'])}, margin {change_text(t['margin'], pt['margin'])}, "
            f"return rate {change_text(t['return_rate'], pt['return_rate'], rate=True)} (from {pct(pt['return_rate'])})."
        )
    else:
        text += f" There is no like-for-like previous quarter: the dataset does not fully cover {w['prior']} ({p_start} to {p_end})."
    facts["kpi.overview"] = _fact("kpi.overview", "kpi", "Quarter at a glance", text, {"window": [start, end], "prior_comparable": prior_ok, **t})

    rows = sorted(cur_by["channel"]["rows"], key=lambda r: -r["revenue"])
    total_rev = t["revenue"] or 1
    facts["mix.channel"] = _fact(
        "mix.channel", "mix", "Revenue by channel",
        "Revenue by channel: " + "; ".join(f"{r['group']} {money(r['revenue'])} ({pct(r['revenue'] / total_rev * 100)})" for r in rows) + ".",
        {r["group"]: r["revenue"] for r in rows},
    )

    for dim, label, noun in (("product", "product", "product"), ("region", "region", "region")):
        prev_rows = {r["group"]: r for r in prior_by[dim]["rows"]} if prior_ok else {}
        for r in sorted(cur_by[dim]["rows"], key=lambda r: r["group"]):
            p = prev_rows.get(r["group"])
            line = f"{r['group']}: revenue {money(r['revenue'])}, units {r['units']}, return rate {pct(r['return_rate'])}, average discount {pct(r['avg_discount'])}"
            if prior_ok and p:
                line += (
                    f"; vs {w['prior']}: revenue {change_text(r['revenue'], p['revenue'])}, units {change_text(r['units'], p['units'])}, "
                    f"return rate {change_text(r['return_rate'], p['return_rate'], rate=True)} (from {pct(p['return_rate'])})"
                )
            elif prior_ok:
                line += f"; no sales in {w['prior']}"
            key = f"{noun}.{slug(r['group'])}"
            facts[key] = _fact(key, label, f"{r['group']} ({noun})", line + ".", {k: r[k] for k in ALL_METRICS if k in r})

    for f in _price_facts(q, w) + _anomaly_facts(q, w):
        facts[f["key"]] = f
    return facts


# --------------------------------------------------------------------------- plan hygiene (pure, unit-tested)
def default_sections(facts: dict[str, dict]) -> list[dict]:
    keys = list(facts)
    return [
        {"title": "Headline results", "purpose": "How did the quarter go overall?", "fact_keys": [k for k in ("kpi.overview", "mix.channel") if k in facts]},
        {"title": "Product performance", "purpose": "Which products drove the result?", "fact_keys": [k for k in keys if k.startswith("product.")]},
        {"title": "Regional performance", "purpose": "How did each region do?", "fact_keys": [k for k in keys if k.startswith("region.")]},
    ]


def finalize_plan(raw: list[dict], facts: dict[str, dict]) -> list[dict]:
    """Turn the planner's output into a trustworthy plan.

    * unknown fact keys are dropped (the model cannot name facts that do not exist); sections left empty are dropped
    * at most MAX_SECTIONS planner sections are kept
    * fewer than MIN_SECTIONS usable sections: default sections fill the gap
    * every flagged anomaly must appear in some section; leftovers go in an appended "Other flagged changes" section
    """
    plan: list[dict] = []
    for s in raw:
        keys = [k for k in dict.fromkeys(s["fact_keys"]) if k in facts]
        title = (s.get("title") or "").strip()
        if keys and title:
            plan.append({"title": title, "purpose": (s.get("purpose") or "").strip(), "fact_keys": keys, "added_by": "planner"})
    plan = plan[:MAX_SECTIONS]
    for d in default_sections(facts):
        if len(plan) >= MIN_SECTIONS:
            break
        if d["fact_keys"] and d["title"].lower() not in {p["title"].lower() for p in plan}:
            plan.append({**d, "added_by": "default"})
    covered = {k for p in plan for k in p["fact_keys"]}
    missed = [k for k in facts if k.startswith("anomaly.") and k not in covered]
    if missed:
        plan.append(
            {"title": "Other flagged changes", "purpose": "What else looked unusual this quarter?", "fact_keys": missed, "added_by": "coverage_check"}
        )
    return [{"index": i, **p} for i, p in enumerate(plan)]


def order_sections(drafts: list[dict]) -> list[dict]:
    """Planned order. Parallel results can be merged in any order, so never trust list position."""
    return sorted(drafts, key=lambda d: d["index"])


# --------------------------------------------------------------------------- number verification (pure, unit-tested)
_STRIP = [
    re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b"),  # dates and months
    re.compile(r"\b\d{4}\s?-?Q[1-4]\b|\bQ[1-4]\s?\d{4}\b", re.IGNORECASE),  # quarter labels
    re.compile(r"\b\d+-pack\b", re.IGNORECASE),  # product names
    re.compile(r"\b20\d{2}\b"),  # bare years
]
_NUMBER = re.compile(r"(?<![\w.])(?P<cur>\$)?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<pct>\s?%)?")


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """Checkable numbers in ``text`` as (as written, absolute value, decimals).

    Years, dates, quarter labels and "4-pack" are removed first; so are bare whole numbers up to 10 (counts such as
    "two regions" are too common to police). Signs are ignored: the check is about the figure, not its direction.
    """
    for pattern in _STRIP:
        text = pattern.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group("num").rstrip(",")
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not m.group("cur") and not m.group("pct") and decimals == 0 and value <= 10:
            continue
        out.append(((m.group("cur") or "") + raw + (m.group("pct") or "").strip(), value, decimals))
    return out


def unsupported_numbers(narrative: str, supporting_texts: list[str]) -> list[str]:
    """Numbers in ``narrative`` that match nothing in ``supporting_texts`` (to the precision they were written with)."""
    allowed = [v for t in supporting_texts for _, v, _ in extract_numbers(t)]
    bad = []
    for token, value, decimals in extract_numbers(narrative):
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - a) <= tolerance for a in allowed):
            bad.append(token)
    return list(dict.fromkeys(bad))


# --------------------------------------------------------------------------- nodes
def gather_facts(state: ReportState) -> dict:
    quarter = (state.get("quarter") or "").strip()
    q = st.parse_quarter(quarter)  # ValueError for "2026-X", "Q5 2026" ...
    facts = build_facts(q.label)
    return {
        "quarter": q.label,
        "facts": facts,
        "flags": [f["text"] for f in facts.values() if f["kind"] == "anomaly"],
        "section_drafts": Overwrite([]),  # fresh run: forget drafts left by a previous run on this thread
    }


PLANNER_PROMPT = (
    "You plan a quarterly sales report for the finance lead of Lumen Home, a smart-home hardware company. Below is a "
    "catalog of computed facts, one per line as `key: text`. Choose 3 to 6 sections that tell the story of THIS quarter "
    "and in reading order. Each section lists the fact keys it needs, copied exactly from the catalog. Rules: "
    "start with the overall result; give any fact marked UNUSUAL a place (a returns or quality section if returns "
    "spiked, a pricing section if a price changed in or just before the quarter); merge related products or regions "
    "into one section instead of one section per row; do not invent keys; do not plan a section the facts cannot support."
)


def plan_sections(state: ReportState) -> dict:
    catalog = "\n".join(f"{k}: {f['text']}" for k, f in state["facts"].items())
    plan = structured_llm(ReportPlan).invoke(
        [SystemMessage(PLANNER_PROMPT), HumanMessage(f"Quarter: {state['quarter']}\n\nFACT CATALOG\n{catalog}")]
    )
    return {"plan": finalize_plan([s.model_dump() for s in plan.sections], state["facts"])}


def dispatch(state: ReportState) -> list[Send]:
    """One worker per planned section; each payload carries only that section's facts."""
    n = len(state["plan"])
    return [
        Send(
            "write_section",
            {
                "quarter": state["quarter"],
                "index": p["index"],
                "of": n,
                "title": p["title"],
                "purpose": p["purpose"],
                "fact_keys": p["fact_keys"],
                "facts": [{"key": k, "text": state["facts"][k]["text"]} for k in p["fact_keys"]],
            },
        )
        for p in state["plan"]
    ]


WRITER_PROMPT = (
    "You write ONE section of a quarterly sales report for a finance lead. Use only the FACTS provided. Copy every "
    "number exactly as written in the facts (same digits and rounding); never round differently, convert units, "
    "estimate, or compute a new figure such as a difference or a total. Say what happened and, where the facts show it, "
    "the size of the change. 70 to 130 words, plain prose, no heading, no bullet list. If the facts do not support a "
    "claim, leave it out."
)


def write_section(payload: dict) -> dict:
    """Worker: narrate one section from its own facts (runs in parallel with its siblings)."""
    facts = "\n".join(f"[{f['key']}] {f['text']}" for f in payload["facts"])
    msg = get_llm("fast").invoke(
        [
            SystemMessage(WRITER_PROMPT),
            HumanMessage(
                f"Report quarter: {payload['quarter']}\nSection {payload['index'] + 1} of {payload['of']}: {payload['title']}\n"
                f"This section must answer: {payload['purpose']}\n\nFACTS\n{facts}"
            ),
        ]
    )
    draft = {"index": payload["index"], "title": payload["title"], "fact_keys": payload["fact_keys"], "body": str(msg.text).strip()}
    return {"section_drafts": [draft]}


SUMMARY_PROMPT = (
    "You write the executive summary of a quarterly sales report for a finance lead: 3 to 4 sentences covering the "
    "overall result and the one or two things that most need attention. Use only numbers that appear in the quarter "
    "facts or the section texts below, copied exactly; do not compute new figures."
)


def synthesize(state: ReportState) -> dict:
    sections = order_sections(state["section_drafts"])
    body = "\n\n".join(f"{s['title']}: {s['body']}" for s in sections)
    summary = get_llm().invoke(
        [
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage(f"QUARTER FACTS\n{state['facts']['kpi.overview']['text']}\n\nSECTIONS\n{body}"),
        ]
    )
    return {"sections": sections, "executive_summary": str(summary.text).strip()}


def verify_numbers(state: ReportState) -> dict:
    """Deterministic audit: report any figure in the prose that is not in the facts the writer was given."""
    facts = state["facts"]
    issues = []
    all_supplied: set[str] = {"kpi.overview"}
    for s in state["sections"]:
        all_supplied.update(s["fact_keys"])
        for token in unsupported_numbers(s["body"], [facts[k]["text"] for k in s["fact_keys"]]):
            issues.append({"where": f"section {s['index'] + 1}: {s['title']}", "number": token})
    for token in unsupported_numbers(state["executive_summary"], [facts[k]["text"] for k in sorted(all_supplied)]):
        issues.append({"where": "executive summary", "number": token})
    return {"verification": {"ok": not issues, "sections_checked": len(state["sections"]), "unsupported_numbers": issues}}


def render_report(state: ReportState) -> dict:
    v = state["verification"]
    lines = [
        f"# Quarterly sales report: {state['quarter']}",
        "",
        "## Executive summary",
        state["executive_summary"],
    ]
    for s in state["sections"]:
        lines += ["", f"## {s['index'] + 1}. {s['title']}", s["body"]]
    lines += ["", "## Number check"]
    if v["ok"]:
        lines.append(f"All figures in the narrative match the computed facts ({v['sections_checked']} sections and the summary checked).")
    else:
        lines.append("These figures in the narrative do not appear in the computed facts for their section. Treat them as unverified:")
        lines += [f"- {i['number']} in {i['where']}" for i in v["unsupported_numbers"]]
    return {"report": "\n".join(lines), "mode": mode_label()}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(ReportState, input_schema=ReportIn, output_schema=ReportOut)
    g.add_node("gather_facts", gather_facts)
    g.add_node("plan_sections", plan_sections)
    g.add_node("write_section", write_section)
    g.add_node("synthesize", synthesize)
    g.add_node("verify_numbers", verify_numbers)
    g.add_node("render_report", render_report)
    g.add_edge(START, "gather_facts")
    g.add_edge("gather_facts", "plan_sections")
    g.add_conditional_edges("plan_sections", dispatch, ["write_section"])  # returns [Send, ...]: N workers
    g.add_edge("write_section", "synthesize")
    g.add_edge("synthesize", "verify_numbers")
    g.add_edge("verify_numbers", "render_report")
    g.add_edge("render_report", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

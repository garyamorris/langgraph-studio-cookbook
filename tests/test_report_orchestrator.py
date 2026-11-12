from __future__ import annotations

import re
import threading
import time

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import report_orchestrator as ex

Q2 = "2026Q2"
CAM_ANOMALY = "anomaly.lumen_cam_pro.return_rate"


def section(title, *keys):
    return ex.SectionPlan(title=title, purpose=f"{title}?", fact_keys=list(keys))


def three_sections():
    return ex.ReportPlan(
        sections=[
            section("Headline", "kpi.overview"),
            section("Price cut", "pricing.lumen_therm", "product.lumen_therm"),
            section("Cam Pro returns", CAM_ANOMALY, "product.lumen_cam_pro"),
        ]
    )


class Chat:
    """Scripted chat model: section writers echo their facts (so every number is supported) unless told otherwise."""

    def __init__(self, section_override=None, summary=None, delay=False):
        self.section_override = section_override or {}
        self.summary = summary
        self.delay = delay
        self.worker_prompts: list[str] = []
        self.summary_prompts: list[str] = []
        self._lock = threading.Lock()

    @property
    def workers(self) -> int:
        return len(self.worker_prompts)

    def __call__(self, messages, tools):
        system, human = messages[0].content, messages[-1].content
        if "write ONE section" in system:
            m = re.search(r"Section (\d+) of (\d+)", human)
            idx, n = int(m.group(1)) - 1, int(m.group(2))
            if self.delay:
                time.sleep(0.04 * (n - idx))  # section 1 finishes LAST
            with self._lock:
                self.worker_prompts.append(human)
            if idx in self.section_override:
                return AIMessage(self.section_override[idx])
            return AIMessage("Facts say: " + human.split("FACTS\n", 1)[1])
        with self._lock:
            self.summary_prompts.append(human)
        return AIMessage(self.summary or "Summary built from the quarter facts above.")


@pytest.fixture
def facts():
    return ex.build_facts(Q2)


def run(plan=None, chat=None, quarter=Q2):
    set_structured_handler(ex.ReportPlan, scripted([plan or three_sections()]))
    chat = chat or Chat()
    set_chat_handler(chat)
    return ex.graph.invoke({"quarter": quarter}), chat


def test_contract_and_offline_run():
    out = check_example(ex).output
    assert out["mode"] == "offline" and len(out["sections"]) >= 3


# --------------------------------------------------------------------------- orchestrator-workers
def test_a_three_section_plan_runs_exactly_three_workers_and_restores_planned_order():
    out, chat = run(chat=Chat(delay=True))
    assert chat.workers == 3, "one worker per planned section"
    assert len(chat.summary_prompts) == 1
    assert [s["title"] for s in out["sections"]] == ["Headline", "Price cut", "Cam Pro returns"]
    assert [s["index"] for s in out["sections"]] == [0, 1, 2]
    assert [p["added_by"] for p in out["plan"]] == ["planner"] * 3
    headings = re.findall(r"^## (\d)\. (.+)$", out["report"], flags=re.M)
    assert headings == [("1", "Headline"), ("2", "Price cut"), ("3", "Cam Pro returns")]
    assert out["report"].index("## Executive summary") < out["report"].index("## 1. Headline")


def test_each_worker_sees_only_the_facts_its_section_asked_for(facts):
    _, chat = run()
    by_title = {re.search(r"Section \d of \d: (.+)", p).group(1): p for p in chat.worker_prompts}
    assert facts["pricing.lumen_therm"]["text"] in by_title["Price cut"]
    assert facts["product.lumen_therm"]["text"] in by_title["Price cut"]
    assert facts[CAM_ANOMALY]["text"] not in by_title["Price cut"]
    assert facts["product.lumen_hub"]["text"] not in "".join(by_title.values()), "unrequested facts are never sent to any worker"
    assert facts["kpi.overview"]["text"] in by_title["Headline"] and "[pricing.lumen_therm]" not in by_title["Headline"]


def test_planner_that_ignores_the_anomaly_cannot_hide_it():
    plan = ex.ReportPlan(sections=[section("Headline", "kpi.overview"), section("Products", "product.lumen_hub"), section("Regions", "region.apac")])
    out, chat = run(plan)
    assert chat.workers == 4
    last = out["plan"][-1]
    assert (last["title"], last["added_by"], last["fact_keys"]) == ("Other flagged changes", "coverage_check", [CAM_ANOMALY])
    assert out["sections"][-1]["title"] == "Other flagged changes"


def test_unusable_planner_output_falls_back_to_a_default_plan():
    bad = ex.ReportPlan(sections=[section("Made up", "kpi.nonexistent", "revenue.everything")])
    out, chat = run(bad)
    assert [p["added_by"] for p in out["plan"][:3]] == ["default"] * 3
    assert [p["title"] for p in out["plan"][:3]] == ["Headline results", "Product performance", "Regional performance"]
    assert out["plan"][-1]["added_by"] == "coverage_check", "and the recall anomaly is still covered"
    assert chat.workers == len(out["plan"])


def test_the_planner_is_shown_the_story_facts():
    seen = {}

    def spy(prompt: str):
        seen["prompt"] = prompt
        return three_sections()

    set_structured_handler(ex.ReportPlan, spy)
    set_chat_handler(Chat())
    ex.graph.invoke({"quarter": Q2})
    assert CAM_ANOMALY in seen["prompt"] and "pricing.lumen_therm" in seen["prompt"] and "UNUSUAL" in seen["prompt"]


# --------------------------------------------------------------------------- verification
def test_fabricated_numbers_are_reported_but_do_not_fail_the_run(facts):
    hub_revenue = re.search(r"revenue (\$[\d,]+\.\d\d)", facts["product.lumen_hub"]["text"]).group(1)
    chat = Chat(section_override={1: "Therm revenue reached $99,999.99 after the cut.", 0: f"Hub made {hub_revenue}."})
    out, _ = run(chat=chat)
    v = out["verification"]
    assert v["ok"] is False
    found = {(i["where"], i["number"]) for i in v["unsupported_numbers"]}
    assert ("section 2: Price cut", "$99,999.99") in found
    assert ("section 1: Headline", hub_revenue) in found, "a real figure that was NOT in this section's facts is still unsupported"
    assert "Number check" in out["report"] and "$99,999.99" in out["report"] and "Treat them as unverified" in out["report"]


def test_clean_run_reports_ok_and_numbers_in_the_summary_are_checked_too():
    out, _ = run()
    assert out["verification"] == {"ok": True, "sections_checked": 3, "unsupported_numbers": []}
    assert "All figures in the narrative match" in out["report"]
    out, _ = run(chat=Chat(summary="Revenue was $1,234,567.89 this quarter."))
    assert out["verification"]["unsupported_numbers"] == [{"where": "executive summary", "number": "$1,234,567.89"}]


def test_summary_may_quote_numbers_from_any_supplied_fact(facts):
    price = re.search(r"to (\$[\d,]+\.\d\d);", facts["pricing.lumen_therm"]["text"]).group(1)
    out, _ = run(chat=Chat(summary=f"The Therm now sells at {price}."))
    assert out["verification"]["ok"]


def test_extract_and_check_numbers():
    text = "Revenue was $33,848.40 (up 12.4%) in 2026Q2 across 2 regions; the 4-pack sold 429 units on 2026-05-01, plus 3 more."
    assert [t for t, _, _ in ex.extract_numbers(text)] == ["$33,848.40", "12.4%", "429"], "years, quarters, dates, 4-pack and small counts are ignored"
    facts = ["revenue $33,848.40, units 429, change -12.4%, 25.7% return rate"]
    assert ex.unsupported_numbers("Revenue of $33,848.40, units 429.", facts) == []
    assert ex.unsupported_numbers("Revenue of about $33,848, down 12.4%.", facts) == [], "correct rounding and sign changes are fine"
    assert ex.unsupported_numbers("Revenue of $33,849 and 25.8% returns.", facts) == ["$33,849", "25.8%"]
    assert ex.unsupported_numbers("Revenue was $34 million.", facts) == ["$34"]
    assert ex.unsupported_numbers("Sold 430 units.", facts) == ["430"]


# --------------------------------------------------------------------------- facts and the quarter filter
def test_facts_cover_only_the_requested_quarter(facts):
    raw = read_csv("sales.csv")
    raw["d"] = pd.to_datetime(raw["order_date"])
    q2 = raw[(raw["d"] >= "2026-04-01") & (raw["d"] <= "2026-06-30") & (raw["status"] != "cancelled")]
    kpi = facts["kpi.overview"]
    assert kpi["data"]["window"] == ["2026-04-01", "2026-06-30"]
    assert kpi["data"]["revenue"] == pytest.approx(q2["revenue"].sum(), abs=0.01)
    assert kpi["data"]["units"] == q2["quantity"].sum()
    products = [f for k, f in facts.items() if k.startswith("product.")]
    regions = [f for k, f in facts.items() if k.startswith("region.")]
    assert len(products) == 8 and len(regions) == 4
    assert sum(f["data"]["revenue"] for f in products) == pytest.approx(kpi["data"]["revenue"], abs=0.05)
    assert sum(f["data"]["revenue"] for f in regions) == pytest.approx(kpi["data"]["revenue"], abs=0.05)
    assert "same days" in kpi["text"] and "2026-01-01 to 2026-03-31" in kpi["text"]
    other = ex.build_facts("2026Q3")["kpi.overview"]["data"]
    assert other["window"] == ["2026-07-01", "2026-09-30"] and other["revenue"] != kpi["data"]["revenue"]


def test_the_built_in_stories_surface_as_facts(facts):
    assert facts[CAM_ANOMALY]["data"]["z"] >= ex.Z_THRESHOLD and "UNUSUAL" in facts[CAM_ANOMALY]["text"]
    assert "$159.00 to $149.00" in facts["pricing.lumen_therm"]["text"] and "2026-05-01" in facts["pricing.lumen_therm"]["text"]
    assert facts["pricing.lumen_therm"]["data"]["when"] == "inside"
    q3 = ex.build_facts("2026Q3")
    assert not [k for k in q3 if k.startswith("anomaly.")], "a quiet quarter has no anomaly facts"
    assert q3["pricing.lumen_therm"]["data"]["when"] == "previous" and "entirely at $149.00" in q3["pricing.lumen_therm"]["text"]
    assert ex.build_facts("2025Q4")["anomaly.lumen_lock.revenue"], "the Nov/Dec seasonality shows up as a flag in the first quarter"


def test_partial_and_uncovered_quarters_are_labelled_instead_of_compared_unfairly():
    q4 = ex.build_facts("2026Q4")["kpi.overview"]
    assert "partial quarter" in q4["text"] and "31 of 92 days" in q4["text"]
    assert "2026-07-01 to 2026-07-31" in q4["text"], "October is compared with July, not with the whole of Q3"
    first = ex.build_facts("2025Q4")["kpi.overview"]
    assert "no like-for-like previous quarter" in first["text"] and first["data"]["prior_comparable"] is False
    assert "no like-for-like previous quarter" in ex.build_facts("2026Q1")["kpi.overview"]["text"]


def test_bad_quarters_are_rejected_with_a_clear_message():
    for bad, msg in (("2026-X", "Could not read quarter"), ("2026Q5", "Could not read quarter"), ("2024Q1", "No data for 2024Q1")):
        with pytest.raises(ValueError, match=msg):
            ex.graph.invoke({"quarter": bad})
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        ex.graph.invoke({"scenario": "not a case"})


# --------------------------------------------------------------------------- ready-made cases (the Studio dropdown)
def case(prefix: str) -> str:
    return next(label for label in ex.CASES if label.startswith(prefix))


def test_a_blank_form_runs_the_first_ready_made_quarter_instead_of_failing():
    for blank in ({}, {"quarter": ""}, {"quarter": "   "}, {"quarter": None}):
        out, _ = run_input(blank)
        assert out["quarter"] == Q2 == ex.EXAMPLE_INPUT["quarter"], blank
        assert [s["title"] for s in out["sections"]] == ["Headline", "Price cut", "Cam Pro returns"]


def run_input(payload):
    set_structured_handler(ex.ReportPlan, scripted([three_sections()]))
    chat = Chat()
    set_chat_handler(chat)
    return ex.graph.invoke(payload), chat


def test_the_first_case_is_the_example_input_and_every_case_names_a_quarter_the_data_covers():
    assert next(iter(ex.CASES.values())) == ex.EXAMPLE_INPUT
    for label, fields in ex.CASES.items():
        assert set(fields) == {"quarter"} and label.startswith(fields["quarter"]), label
        ex.build_facts(fields["quarter"])  # raises if the dataset has nothing for it


def test_case_labels_quote_what_the_facts_say_about_their_quarter():
    for label, fields in ex.CASES.items():
        facts = ex.build_facts(fields["quarter"])
        kpi = facts["kpi.overview"]
        if found := re.search(r"revenue (\$[\d,]+\.\d\d)", label):
            assert found.group(1) == ex.money(kpi["data"]["revenue"]), label
        if found := re.search(r"return rate ([\d.]+%)", label):
            assert found.group(1) == ex.pct(kpi["data"]["return_rate"]), label
        if found := re.search(r"(\d+) of (\d+) days", label):
            assert f"the data covers {found.group(1)} of {found.group(2)} days" in kpi["text"], label
        anomalies = [k for k in facts if k.startswith("anomaly.")]
        if "no anomaly flags" in label:
            assert not anomalies, f"{label}: anomaly facts are {anomalies}"
        if "flagged" in label:
            assert anomalies, label
        if "no like-for-like previous quarter" in label or "no previous quarter" in label:
            assert "no like-for-like previous quarter" in kpi["text"] and kpi["data"]["prior_comparable"] is False, label
        if "partial" in label or "Nov-Dec only" in label or "October only" in label:
            assert "partial quarter" in kpi["text"], label
        if "full quarter" in label:
            assert "full quarter" in kpi["text"], label
        if "compared with July" in label:
            assert "2026-07-01 to 2026-07-31" in kpi["text"], label
        if "$149" in label:
            assert "entirely at $149.00" in facts["pricing.lumen_therm"]["text"], label


def test_the_flagged_anomalies_named_in_the_labels_are_the_ones_the_facts_hold():
    assert "anomaly.lumen_cam_pro.return_rate" in ex.build_facts("2026Q2") and "Cam Pro" in case("2026Q2")
    assert "anomaly.lumen_lock.revenue" in ex.build_facts("2025Q4") and "Lock" in case("2025Q4")


def test_a_flagged_case_always_gets_a_section_whatever_the_planner_chose():
    for label in (case("2026Q2"), case("2025Q4")):
        facts = ex.build_facts(ex.CASES[label]["quarter"])
        anomalies = {k for k in facts if k.startswith("anomaly.")}
        out, _ = run(plan=ex.ReportPlan(sections=[section("Headline", "kpi.overview")]), quarter=ex.CASES[label]["quarter"])
        covered = {k for p in out["plan"] for k in p["fact_keys"]}
        assert anomalies <= covered, f"{label}: the planner's plan hid {anomalies - covered}"


def test_picking_a_quarter_after_another_on_one_thread_reports_the_picked_quarter():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "picks"}}
    set_structured_handler(ex.ReportPlan, scripted([three_sections()]))
    set_chat_handler(Chat())
    for label in (case("2026Q4"), case("2026Q2"), case("2025Q4")):
        out = g.invoke({"scenario": label}, cfg)
        assert out["quarter"] == ex.CASES[label]["quarter"] and out["report"].startswith(f"# Quarterly sales report: {out['quarter']}")
        state = g.get_state(cfg).values
        assert state["scenario"] is None and len(state["section_drafts"]) == len(out["sections"])


# --------------------------------------------------------------------------- plan hygiene and ordering
def test_finalize_plan_rules(facts):
    raw = [{"title": f"S{i}", "purpose": "p", "fact_keys": ["kpi.overview", "kpi.overview", "nope"]} for i in range(9)]
    plan = ex.finalize_plan(raw, facts)
    non_catch = [p for p in plan if p["added_by"] != "coverage_check"]
    assert len(non_catch) == ex.MAX_SECTIONS, "at most 6 planner sections"
    assert all(p["fact_keys"] == ["kpi.overview"] for p in non_catch), "duplicates and unknown keys removed"
    assert [p["index"] for p in plan] == list(range(len(plan)))
    one = ex.finalize_plan([{"title": "Only", "purpose": "p", "fact_keys": ["mix.channel"]}], facts)
    assert [p["added_by"] for p in one[:3]] == ["planner", "default", "default"]
    covered = ex.finalize_plan([{"title": "Recall", "purpose": "p", "fact_keys": [CAM_ANOMALY]}], facts)
    assert not any(p["added_by"] == "coverage_check" for p in covered)


def test_order_sections_sorts_by_planned_index():
    shuffled = [{"index": 2, "title": "c"}, {"index": 0, "title": "a"}, {"index": 1, "title": "b"}]
    assert [d["title"] for d in ex.order_sections(shuffled)] == ["a", "b", "c"]


def test_a_second_run_on_the_same_thread_does_not_append_to_the_first_runs_drafts():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    set_structured_handler(ex.ReportPlan, scripted([three_sections()]))
    set_chat_handler(Chat())
    first = g.invoke({"quarter": Q2}, cfg)
    second = g.invoke({"quarter": Q2}, cfg)
    assert len(first["sections"]) == len(second["sections"]) == 3
    assert len(g.get_state(cfg).values["section_drafts"]) == 3

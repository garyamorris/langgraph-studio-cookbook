from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.fakes import set_chat_handler
from deepdive.common.testing import check_example
from deepdive.examples import pricing_whatif as ex

NODES = ("load_history", "estimate_elasticity", "review_estimate", "assumptions", "simulate", "recommend")


def instrument(monkeypatch) -> Counter:
    """Wrap every node function so tests can count how many times each one really executed."""
    counts: Counter = Counter()

    def make(original, key):
        def wrapper(state):
            counts[key] += 1
            return original(state)

        return wrapper

    for name in NODES:
        monkeypatch.setattr(ex, name, make(getattr(ex, name), name))
    return counts


def count_llm_calls() -> list[str]:
    calls: list[str] = []

    def handler(messages, tools):
        calls.append(str(messages[0].content)[:30])
        return AIMessage("not a faithful paragraph")

    set_chat_handler(handler)
    return calls


def fork_point(graph, config):
    """The checkpoint right after set_assumptions: its ``next`` is ``simulate``."""
    return next(s for s in graph.get_state_history(config) if s.next == ("simulate",))


def fork(graph, config, **changes):
    point = fork_point(graph, config)
    edited = {**copy.deepcopy(point.values["assumptions"]), **changes}
    return point, graph.update_state(point.config, {"assumptions": edited})


def cfg(name="t"):
    return {"configurable": {"thread_id": name}}


# --------------------------------------------------------------------------- contract
def test_example_contract():
    res = check_example(ex)
    assert res.output["product"] == "Lumen Therm"
    assert res.output["recommendation"]["status"] == "recommended"
    assert res.output["mode"] == "offline"


# --------------------------------------------------------------------------- the demand model on the real data
def test_discounts_lift_units_in_the_data_and_the_elasticity_has_the_right_sign_and_size():
    out = ex.graph.invoke({"product": "Lumen Therm"})
    e = out["elasticity"]
    assert -8 < e["point"] < -2, e
    assert e["lo"] < e["point"] < e["hi"] < 0, "the whole 90% interval is on the negative side"
    assert e["n_weeks"] >= 40 and 120 < e["price_min"] < e["price_max"] <= 160
    # direct check of the premise on the raw weekly rows: cheaper weeks sell more
    rows = ex.load_history({"product": "Lumen Therm"})["history"]
    cutoff = np.median([r["avg_price"] for r in rows])
    cheap = np.mean([r["units"] for r in rows if r["avg_price"] <= cutoff])
    dear = np.mean([r["units"] for r in rows if r["avg_price"] > cutoff])
    assert cheap > dear


def test_the_estimate_is_reproducible_and_the_seed_moves_only_the_interval():
    rows = ex.load_history({"product": "Lumen Therm"})["history"]
    prices, units = [r["avg_price"] for r in rows], [r["units"] for r in rows]
    assert ex.fit_elasticity(prices, units) == ex.fit_elasticity(prices, units)
    a, b = ex.fit_elasticity(prices, units, seed=1), ex.fit_elasticity(prices, units, seed=2)
    assert a["point"] == b["point"] and (a["lo"], a["hi"]) != (b["lo"], b["hi"])


def test_fit_recovers_a_known_elasticity_and_refuses_to_fit_nothing():
    prices = list(np.linspace(80, 120, 30))
    units = [1000 * p ** -2.0 for p in prices]
    fit = ex.fit_elasticity(prices, units)
    assert fit["point"] == pytest.approx(-2.0, abs=0.01) and fit["r2"] > 0.999
    assert fit["lo"] <= -2.0 + 0.01 and fit["hi"] >= -2.0 - 0.01
    with pytest.raises(ValueError, match="never changed"):
        ex.fit_elasticity([100.0] * 20, list(range(1, 21)))
    with pytest.raises(ValueError, match="at least"):
        ex.fit_elasticity([100.0, 110.0, 120.0], [5, 4, 3])


def test_other_products_and_unknown_products():
    out = ex.graph.invoke({"product": "Lumen Cam Pro"})
    assert out["elasticity"]["point"] < -1 and out["assumptions"]["unit_cost"] == 38.0
    with pytest.raises(ValueError, match="Unknown product"):
        ex.graph.invoke({"product": "Lumen Toaster"})


# --------------------------------------------------------------------------- assumptions, simulation, recommendation
def test_assumptions_hold_everything_downstream_needs():
    a = ex.graph.invoke({"product": "Lumen Therm", "margin_target": 0.33})["assumptions"]
    assert a["margin_target"] == 0.33 and a["unit_cost"] == 71.0 and a["variable_cost_rate"] == 0.22
    assert a["elasticity_low"] <= a["elasticity"] <= a["elasticity_high"] < 0
    assert a["candidate_prices"] == sorted(a["candidate_prices"]) and a["candidate_prices"][0] >= 0.85 * a["baseline_price"]
    assert 135 < a["baseline_price"] < 150 and a["baseline_units_per_week"] == pytest.approx(4.0, abs=0.5)
    assert a["support_min"] < a["baseline_price"] < a["support_max"]


def test_a_positive_fitted_elasticity_is_clamped_with_a_note():
    state = {
        "product": "X", "unit_cost": 50.0,
        "history": [{"week": f"2026-0{m}-01", "units": 5, "revenue": 500.0, "avg_price": 100.0} for m in range(1, 10)],
        "elasticity": {"point": 0.7, "lo": -0.5, "hi": 1.9, "price_min": 90.0, "price_max": 110.0},
    }
    a = ex.assumptions(state)["assumptions"]
    assert a["elasticity"] == ex.ELASTICITY_CEILING and any("clamped" in n for n in a["notes"])
    assert a["elasticity_high"] < 0 and a["elasticity_low"] <= a["elasticity"]


def test_simulation_behaves_like_a_demand_curve():
    out = ex.graph.invoke({"product": "Lumen Therm"})
    a, sim = out["assumptions"], out["simulation"]
    assert sim["baseline"]["units_per_week"] == pytest.approx(a["baseline_units_per_week"], abs=0.01)
    units = [r["units_per_week"] for r in sim["rows"]]
    assert units == sorted(units, reverse=True), "higher price -> fewer units"
    margins = [r["margin_pct"] for r in sim["rows"]]
    assert margins == sorted(margins), "higher price -> higher margin %"
    for r in sim["rows"]:
        assert r["units_per_week_low"] <= r["units_per_week"] <= r["units_per_week_high"]
        assert r["revenue"] == pytest.approx(r["units_per_week"] * a["horizon_weeks"] * r["price"], rel=5e-3)
        assert r["meets_margin_target"] == (r["margin_pct"] >= a["margin_target"])
    assert sim["rows"][-1]["within_support"] is False and sim["rows"][0]["within_support"] is True


def test_recommendation_is_the_best_feasible_candidate_chosen_in_code():
    rows = [
        {"price": 100, "contribution": 900.0, "meets_margin_target": False, "within_support": True},
        {"price": 110, "contribution": 800.0, "meets_margin_target": True, "within_support": True},
        {"price": 120, "contribution": 800.0, "meets_margin_target": True, "within_support": True},
        {"price": 130, "contribution": 950.0, "meets_margin_target": True, "within_support": False},
    ]
    assert ex.choose_price(rows)["price"] == 110, "infeasible and unsupported rows are ignored; ties go to the lower price"
    assert ex.choose_price([{**r, "meets_margin_target": False} for r in rows]) is None


def test_margin_target_drives_the_recommendation_and_warns_when_extrapolating():
    low = ex.graph.invoke({"product": "Lumen Therm", "margin_target": 0.28})["recommendation"]
    high = ex.graph.invoke({"product": "Lumen Therm", "margin_target": 0.35})["recommendation"]
    assert (low["price"], high["price"]) == (145.0, 170.0)
    assert low["margin_pct"] >= 0.28 and high["margin_pct"] >= 0.35
    assert high["units_per_week"] < low["units_per_week"] and high["contribution"] < low["contribution"]
    assert any("extrapolation" in w for w in high["warnings"]) and not any("extrapolation" in w for w in low["warnings"])
    none = ex.graph.invoke({"product": "Lumen Therm", "margin_target": 0.45})
    assert none["recommendation"]["status"] == "no_feasible_price" and none["recommendation"]["price"] is None
    assert "45%" in none["narrative"]


def test_the_narrative_must_quote_the_simulated_numbers_or_a_template_is_used():
    def faithful(messages, tools):
        found = re.search(r"exactly as written: (.*?)\. Use no other", str(messages[0].content), re.S)
        return AIMessage(f"Go with these figures: {found.group(1)}." if found else "The fit is noisy.")  # only the paragraph call has quotes

    set_chat_handler(faithful)
    ok = ex.graph.invoke({"product": "Lumen Therm"})
    assert ok["narrative_source"].startswith("model") and "$145" in ok["narrative"] and "29.0%" in ok["narrative"]

    set_chat_handler(lambda messages, tools: AIMessage("Price it at $999 for a 99% margin."))
    bad = ex.graph.invoke({"product": "Lumen Therm"})
    assert bad["narrative_source"].startswith("template") and "$999" not in bad["narrative"]
    assert "$145" in bad["narrative"] and "29.0%" in bad["narrative"]


# --------------------------------------------------------------------------- time travel
def test_forking_at_the_assumptions_checkpoint_reruns_only_simulate_and_recommend(monkeypatch):
    counts = instrument(monkeypatch)
    llm = count_llm_calls()
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()

    graph.invoke({"product": "Lumen Therm"}, c)
    assert counts == Counter({n: 1 for n in NODES}) and len(llm) == 2

    _, fork_config = fork(graph, c, margin_target=0.35)
    assert counts == Counter({n: 1 for n in NODES}), "update_state runs no node"
    graph.invoke(None, fork_config)
    assert counts["load_history"] == counts["estimate_elasticity"] == counts["review_estimate"] == counts["assumptions"] == 1
    assert counts["simulate"] == counts["recommend"] == 2
    assert len(llm) == 3, "the fork paid for one model call (the paragraph), not for the upstream review again"


def test_the_original_run_is_untouched_by_the_fork_and_the_branches_are_visible():
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    graph.invoke({"product": "Lumen Therm"}, c)
    original = graph.get_state(c)
    before = copy.deepcopy(original.values)
    assert original.values["recommendation"]["price"] == 145.0

    point, fork_config = fork(graph, c, margin_target=0.35)
    graph.invoke(None, fork_config)

    again = graph.get_state(original.config)  # addressed by checkpoint_id, not "latest"
    assert again.values == before, "checkpoints are immutable"
    assert again.values["assumptions"]["margin_target"] == 0.28 and again.values["recommendation"]["price"] == 145.0
    latest = graph.get_state(c)
    assert latest.values["recommendation"]["price"] == 170.0 and latest.values["assumptions"]["margin_target"] == 0.35

    history = list(graph.get_state_history(c))
    at_simulate = [s for s in history if s.next == ("simulate",)]
    assert len(at_simulate) == 2, "the fork point and its child: two checkpoints waiting to run simulate"
    child = next(s for s in at_simulate if s.config != point.config)
    assert child.parent_config["configurable"]["checkpoint_id"] == point.config["configurable"]["checkpoint_id"]
    assert child.metadata["source"] == "update"
    finals = [s for s in history if s.next == ()]
    assert {s.values["recommendation"]["price"] for s in finals} == {145.0, 170.0}, "both branches end in their own final checkpoint"


def test_a_fork_gives_the_same_answer_as_a_fresh_full_run_with_that_assumption():
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    graph.invoke({"product": "Lumen Therm"}, c)
    _, fork_config = fork(graph, c, margin_target=0.35)
    graph.invoke(None, fork_config)
    forked = graph.get_state(c).values

    fresh = ex.graph.invoke({"product": "Lumen Therm", "margin_target": 0.35})
    assert forked["recommendation"] == fresh["recommendation"]
    assert forked["simulation"] == fresh["simulation"]
    assert forked["assumptions"] == fresh["assumptions"]


def test_downstream_nodes_read_only_the_assumptions_dict_not_the_original_input():
    """The input still says 0.28; the edited assumptions say 0.35; the answer follows the assumptions."""
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    graph.invoke({"product": "Lumen Therm", "margin_target": 0.28}, c)
    _, fork_config = fork(graph, c, margin_target=0.35)
    graph.invoke(None, fork_config)
    values = graph.get_state(c).values
    assert values["margin_target"] == 0.28 and values["recommendation"]["margin_target"] == 0.35 and values["recommendation"]["price"] == 170.0


def test_other_assumptions_can_be_forked_too_and_two_forks_share_one_parent():
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    graph.invoke({"product": "Lumen Therm"}, c)
    base_units = graph.get_state(c).values["simulation"]["rows"][5]["units"]

    # fork_point() looks at the thread's whole history, so pin the ORIGINAL fork point for both forks
    original_point = fork_point(graph, c)

    def fork_original(**changes):
        edited = {**copy.deepcopy(original_point.values["assumptions"]), **changes}
        return graph.update_state(original_point.config, {"assumptions": edited})

    graph.invoke(None, fork_original(elasticity=-3.0, elasticity_low=-3.5, elasticity_high=-2.5))
    scenario_a = graph.get_state(c).values
    assert scenario_a["simulation"]["rows"][5]["units"] != base_units
    graph.invoke(None, fork_original(margin_target=0.45))
    scenario_b = graph.get_state(c).values
    assert scenario_a["recommendation"]["status"] == "recommended" and scenario_b["recommendation"]["status"] == "no_feasible_price"
    parents = {s.parent_config["configurable"]["checkpoint_id"] for s in graph.get_state_history(c) if s.metadata.get("source") == "update"}
    assert parents == {original_point.config["configurable"]["checkpoint_id"]}, "both what-ifs were forked from the same original checkpoint"


def test_replaying_a_checkpoint_without_editing_it_reruns_only_what_comes_after(monkeypatch):
    counts = instrument(monkeypatch)
    graph, c = ex.build_graph(checkpointer=InMemorySaver()), cfg()
    first = graph.invoke({"product": "Lumen Therm"}, c)
    point = fork_point(graph, c)
    replayed = graph.invoke(None, point.config)
    assert counts["simulate"] == counts["recommend"] == 2 and counts["load_history"] == counts["estimate_elasticity"] == 1
    assert replayed["recommendation"] == first["recommendation"] and replayed["simulation"] == first["simulation"]


# --------------------------------------------------------------------------- the demo script
def test_time_travel_demo_script_runs_and_shows_the_saving():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "time_travel_demo.py")],
        capture_output=True, text=True, cwd=root, env={**os.environ, "DEEPDIVE_OFFLINE": "1"}, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "nodes executed: simulate -> recommend" in out.stdout
    assert "original run" in out.stdout and "forked what-if" in out.stdout
    assert "$145" in out.stdout and "$170" in out.stdout

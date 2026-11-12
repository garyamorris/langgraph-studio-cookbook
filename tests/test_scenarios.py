"""Contract for ready-made Studio cases (``CASES``, see ``deepdive.common.scenarios``).

Studio shows each graph's input schema as a form; examples keyed by dataset ids would be a dead end for anyone who has
not read the data. An example that declares ``CASES`` must therefore (1) offer them as a ``scenario`` dropdown in
its input schema, (2) behave identically whether the case is picked or its fields are typed by hand, (3) run the first
case on a blank form, and (4) let a later pick fully replace the earlier run's input on a reused thread.
"""

from __future__ import annotations

import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common import scenarios as sc
from deepdive.common.embeddings import make_store
from deepdive.common.testing import normalise, run_graph
from test_all_examples import EXEMPT_FROM_REUSE, IDS, module_of

@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Cases such as live_digest_stream's pace_s demo knob really sleep; these tests run every case, so skip the waits."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)


WITH_SCENARIOS = [gid for gid in IDS if hasattr(module_of(gid), "CASES")]


def fresh(mod):
    return mod.build_graph(checkpointer=InMemorySaver(), store=make_store())


def run(mod, payload, graph=None, thread=None):
    """One run, summarised for comparison: paused at an interrupt or finished, plus the normalised output."""
    graph = graph or fresh(mod)
    cfg = {"configurable": {"thread_id": thread or f"sc-{uuid.uuid4().hex[:8]}"}}
    res = run_graph(graph, payload, (), config=cfg)
    return {"output": normalise(res.output), "interrupts": normalise(res.interrupts)}, graph, res.config


def test_the_helper_rejects_an_empty_set_of_cases():
    with pytest.raises(ValueError):
        sc.scenario_field({})


def test_expand_rules():
    cases = {"first": {"k": 1}, "second": {"k": 2}}
    never = lambda s: False  # noqa: E731
    always = lambda s: True  # noqa: E731
    assert sc.expand({"scenario": "second"}, cases, has_input=always) == {"k": 2, "scenario": None}
    assert sc.expand({}, cases, has_input=never) == {"k": 1, "scenario": None}, "a blank form runs the first case"
    assert sc.expand({}, cases, has_input=always) == {}, "real input typed by hand is left alone"
    assert sc.expand({"scenario": None}, cases, has_input=always) == {"scenario": None}
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        sc.expand({"scenario": "nope"}, cases, has_input=always)


def test_scenarios_are_declared_where_the_form_needs_data():
    # Not every example needs them (the ones keyed by nothing in a dataset do not), but most do: guard against the
    # feature quietly disappearing from the examples that motivated it.
    assert len(WITH_SCENARIOS) >= 20, f"only {len(WITH_SCENARIOS)} examples declare CASES"


@pytest.mark.parametrize("gid", WITH_SCENARIOS)
def test_scenario_declarations_are_well_formed(gid):
    mod = module_of(gid)
    scenarios = mod.CASES
    assert len(scenarios) >= 3, f"{gid}: offer at least 3 cases"
    assert all(isinstance(k, str) and k.strip() and len(k) <= 120 for k in scenarios), f"{gid}: label 1-120 chars"
    assert all(isinstance(v, dict) and v for v in scenarios.values()), f"{gid}: every case is a non-empty input dict"
    keysets = {frozenset(v) for v in scenarios.values()}
    assert len(keysets) == 1, (
        f"{gid}: every case must set the same keys (a key one case omits would leak from the previous run); "
        f"got {sorted(map(sorted, keysets))}"
    )
    assert "scenario" not in next(iter(keysets)), f"{gid}: a case must not set 'scenario' itself"


@pytest.mark.parametrize("gid", WITH_SCENARIOS)
def test_the_input_schema_offers_the_cases_as_an_optional_dropdown(gid):
    mod = module_of(gid)
    schema = fresh(mod).get_input_jsonschema()
    prop = schema["properties"].get("scenario")
    assert prop is not None, f"{gid}: the input schema has no `scenario` field"
    assert prop.get("enum") == list(mod.CASES), f"{gid}: the dropdown must list exactly the CASES labels"
    assert "scenario" not in schema.get("required", []), f"{gid}: `scenario` must be optional"
    assert not schema.get("required"), f"{gid}: nothing may be required, or Submit on a blank form is refused"


@pytest.mark.parametrize("gid", WITH_SCENARIOS)
def test_a_blank_form_runs_the_first_case(gid):
    mod = module_of(gid)
    first = next(iter(mod.CASES.values()))
    blank, _, _ = run(mod, {})
    typed, _, _ = run(mod, dict(first))
    assert blank == typed


@pytest.mark.parametrize("gid", WITH_SCENARIOS)
def test_picking_a_case_equals_typing_its_fields(gid):
    mod = module_of(gid)
    for label, fields in mod.CASES.items():
        picked, graph, cfg = run(mod, {"scenario": label})
        typed, _, _ = run(mod, dict(fields))
        assert picked == typed, f"{gid}: picking {label!r} behaves differently from typing its fields"
        assert graph.get_state(cfg).values.get("scenario") is None, f"{gid}: `scenario` must be cleared once used"


@pytest.mark.parametrize("gid", [g for g in WITH_SCENARIOS if g not in EXEMPT_FROM_REUSE])
def test_a_later_pick_replaces_the_earlier_runs_input_on_a_reused_thread(gid):
    mod = module_of(gid)
    labels = list(mod.CASES)
    first, second = labels[-1], labels[0]  # the furthest-apart pair
    graph, thread = fresh(mod), f"sc-{uuid.uuid4().hex[:8]}"
    run(mod, {"scenario": first}, graph, thread)
    switched, _, _ = run(mod, {"scenario": second}, graph, thread)
    alone, _, _ = run(mod, {"scenario": second})
    assert switched == alone, f"{gid}: picking a case after another on one thread leaked the earlier input"

"""Generic checks that run against EVERY graph registered in langgraph.json.

Per-example behaviour is tested in tests/test_<example>.py; this file guarantees the contract that Studio and the
docs generator rely on (see deepdive.common.testing.check_example), so an example cannot be added half-finished.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from deepdive.common.testing import check_example, run_example_twice

ROOT = Path(__file__).resolve().parents[1]
REGISTRY: dict[str, str] = json.loads((ROOT / "langgraph.json").read_text())["graphs"]
IDS = sorted(REGISTRY)


def module_of(graph_id: str):
    path = REGISTRY[graph_id].split(":")[0]  # ./src/deepdive/examples/ticket_router.py
    return importlib.import_module(path.removeprefix("./src/").removesuffix(".py").replace("/", "."))


def test_registry_points_at_graph_objects_and_matches_files():
    assert IDS
    for gid, target in REGISTRY.items():
        assert target == f"./src/deepdive/examples/{gid}.py:graph", f"{gid}: use ./src/deepdive/examples/<id>.py:graph"
    # Files starting with "_" (and __init__) are private helper modules shared between examples, not examples.
    on_disk = {p.stem for p in (ROOT / "src/deepdive/examples").glob("*.py") if not p.stem.startswith("_")}
    assert on_disk == set(IDS), f"examples on disk but not in langgraph.json (or vice versa): {on_disk ^ set(IDS)}"


@pytest.mark.parametrize("gid", IDS)
def test_example_meets_the_contract_and_runs_offline(gid):
    check_example(module_of(gid))


# Examples whose state is meant to outlive a run on the same thread. Everything else must start every run clean.
EXEMPT_FROM_REUSE = {
    "conversation_memory": "a chat: the messages and the running summary are the thread's memory",
    "functional_moderation": "`previous` carries the processed review ids to the next run on purpose",
    "memory_concierge": "a chat: messages persist, and so do the Store-backed memories",
    "ops_copilot": "a chat: messages, the audit log and the simulated environment belong to the thread",
    "refund_approval": "the payout ledger in the Store makes a repeated request_id 'already_paid' on purpose",
    "resilient_enrichment": "its vendor-lookup cache is shared across runs on purpose (cache_hits, from_cache)",
    "sales_react_agent": "a chat: the messages are the thread's memory",
    "support_swarm": "a chat: messages, active_agent and handoffs carry on from the last turn on purpose",
}


def test_every_exemption_names_a_registered_example():
    assert set(EXEMPT_FROM_REUSE) <= set(IDS), set(EXEMPT_FROM_REUSE) - set(IDS)


@pytest.mark.parametrize("gid", IDS)
def test_a_second_run_on_the_same_thread_behaves_like_a_fresh_one(gid):
    """Studio keeps submitting runs on the same thread: leftover accumulating state must not change the next run."""
    if gid in EXEMPT_FROM_REUSE:
        pytest.skip(EXEMPT_FROM_REUSE[gid])
    reused, fresh = run_example_twice(module_of(gid))
    assert reused["interrupts"] == fresh["interrupts"], f"{gid}: the second run paused differently from a fresh one"
    assert reused["output"] == fresh["output"], f"{gid}: the second run on one thread differs from a fresh thread's"

"""Time-travel demo for the ``pricing_whatif`` example: fork a finished run and recompute only what depends on the edit.

    .venv/bin/python scripts/time_travel_demo.py                      # 28% target vs a 35% what-if
    .venv/bin/python scripts/time_travel_demo.py --margin 0.30        # a different what-if
    DEEPDIVE_OFFLINE=1 .venv/bin/python scripts/time_travel_demo.py   # force the offline stand-in model

What it does, in order (each step is printed):
  1. run the graph once on an in-memory checkpointer and show which nodes ran;
  2. list the thread's checkpoints with ``graph.get_state_history`` and pick the one whose ``next`` is ``simulate``
     (the state right after ``set_assumptions``);
  3. fork it with ``graph.update_state`` (a new margin target inside the ``assumptions`` dict);
  4. continue from the fork with ``graph.stream(None, fork_config)`` and show that only ``simulate`` and ``recommend`` ran;
  5. print both outcomes side by side, reading the ORIGINAL run back by its checkpoint id (it was not overwritten).

Checkpoint ids are shortened to their last 6 characters (the leading characters are a shared timestamp).

Studio does the same thing from its UI: pick the step, edit the state, fork.
"""

from __future__ import annotations

import argparse
import copy

from langgraph.checkpoint.memory import InMemorySaver

from deepdive.examples import pricing_whatif


def ran(graph, payload, config) -> list[str]:
    """Run/continue and return the names of the nodes that executed (``stream_mode="updates"`` has one event per node)."""
    return [node for event in graph.stream(payload, config, stream_mode="updates") for node in event]


def row(label: str, values: dict) -> str:
    rec, a = values["recommendation"], values["assumptions"]
    price = f"${rec['price']:.0f}" if rec.get("price") else "none feasible"
    detail = (
        f"{rec['units_per_week']:.1f} units/wk, margin {rec['margin_pct']:.1%}, contribution ${rec['contribution']:,.0f}"
        if rec["status"] == "recommended"
        else rec["reason"]
    )
    return f"{label:<22} target {a['margin_target']:.0%}  ->  price {price:<14} {detail}"


def main(product: str, margin: float) -> None:
    graph = pricing_whatif.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "pricing-demo"}}

    print(f"1) Full run for {product} at the default margin target")
    print("   nodes executed:", " -> ".join(ran(graph, {"product": product}, config)))
    original = graph.get_state(config)  # its config carries the checkpoint_id of the final checkpoint
    original_config = original.config

    print("\n2) Checkpoint history (newest first):")
    history = list(graph.get_state_history(config))
    for snap in history:
        print(f"   step {snap.metadata['step']:>2}  checkpoint {snap.config['configurable']['checkpoint_id'][-6:]}  next={snap.next}")
    fork_point = next(s for s in history if s.next == ("simulate",))
    print(f"   -> fork point: checkpoint {fork_point.config['configurable']['checkpoint_id'][-6:]} (right after set_assumptions)")

    print(f"\n3) Fork: set margin_target to {margin:.0%} inside the assumptions dict")
    edited = {**copy.deepcopy(fork_point.values["assumptions"]), "margin_target": margin}
    fork_config = graph.update_state(fork_point.config, {"assumptions": edited})
    print(f"   new checkpoint {fork_config['configurable']['checkpoint_id'][-6:]} (child of the fork point); next={graph.get_state(fork_config).next}")

    print("\n4) Continue from the fork")
    print("   nodes executed:", " -> ".join(ran(graph, None, fork_config)), "  (load_history, estimate_elasticity, review_estimate, set_assumptions NOT re-run)")
    forked = graph.get_state(config)  # the thread's latest checkpoint now follows the fork

    print("\n5) Side by side (the original is read back by its checkpoint id, not overwritten)")
    print("  ", row("original run", graph.get_state(original_config).values))
    print("  ", row("forked what-if", forked.values))
    print("\n   forked narrative:", forked.values["narrative"])
    print("   checkpoints in the thread now:", len(list(graph.get_state_history(config))), "(both branches share every checkpoint up to the fork point)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="Lumen Therm")
    ap.add_argument("--margin", type=float, default=0.35, help="contribution-margin target for the what-if (default 0.35)")
    args = ap.parse_args()
    main(args.product, args.margin)

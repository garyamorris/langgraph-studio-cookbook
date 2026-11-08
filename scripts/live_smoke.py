"""Opt-in LIVE smoke test: run examples against the real OpenAI API. THIS COSTS MONEY.

Offline tests cannot tell you whether prompts, structured-output schemas, tool schemas and Decisions API question shapes
work with the real service. This runs each chosen example once with its EXAMPLE_INPUT (answering interrupts from
EXAMPLE_RESUMES) and reports pass/fail, wall time and how many model calls it made.

Safety rails: every example runs in its own subprocess with a wall-clock timeout (a hung call cannot stall the rest),
and has a hard budget of model calls (the run aborts if it is exceeded).

    .venv/bin/python scripts/live_smoke.py --go                      # the risky subset (see RISKY)
    .venv/bin/python scripts/live_smoke.py --go corrective_rag       # specific examples
    .venv/bin/python scripts/live_smoke.py --go --all                # everything (most expensive)
    .venv/bin/python scripts/live_smoke.py --go --budget 25 --timeout 120
    .venv/bin/python scripts/live_smoke.py --go --all --save out/    # also write each example's full output to out/<id>.json

PASS means the example ran to completion in live mode; it does not mean the answer was good. Read the saved outputs.
This runs graphs synchronously, so resilient_enrichment (whose TimeoutPolicy needs ainvoke/astream) reports PASS while
every record is "enriched_unnormalized"; that is expected here and correct under Studio.

Without --go it only prints what it would run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Examples whose live behaviour offline tests could not verify (Decisions API shapes, nullable strict schemas, tool
# schemas, streaming leaks, prompts that write SQL / memory operations).
RISKY = [
    "corrective_rag", "subject_line_tournament", "credit_decision_gate", "classifier_bakeoff", "portfolio_desk_lite",
    "plan_execute_analyst", "sales_react_agent", "support_swarm", "ops_copilot", "live_digest_stream",
    "memory_concierge", "sql_analyst", "adaptive_qa_router",
]
# Keep paid runs small where an example scales with an input.
INPUT_OVERRIDES = {"classifier_bakeoff": {"sample_size": 12}}


def run_one(gid: str, budget: int, save_dir: str | None = None) -> dict:
    """Run one example live in THIS process and return a result dict (called in the child process)."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langgraph.cache.memory import InMemoryCache
    from langgraph.checkpoint.memory import InMemorySaver

    from deepdive.common import decisions as D
    from deepdive.common.config import is_offline
    from deepdive.common.embeddings import make_store
    from deepdive.common.testing import run_graph

    class BudgetExceeded(RuntimeError):
        pass

    class CallCounter(BaseCallbackHandler):
        raise_error = True  # let the abort propagate out of the run

        def __init__(self) -> None:
            self.chat = 0

        def _tick(self) -> None:
            self.chat += 1
            if self.chat > budget:
                raise BudgetExceeded(f"more than {budget} model calls")

        def on_chat_model_start(self, *a, **k) -> None:
            self._tick()

        def on_llm_start(self, *a, **k) -> None:
            self._tick()

    if is_offline():
        return {"ok": False, "error": "offline mode is active (DEEPDIVE_OFFLINE set or no OPENAI_API_KEY)"}

    stats = {"decisions": 0}
    original = D._live_call

    def counting(*a, **k):
        stats["decisions"] += 1
        if stats["decisions"] > budget:
            raise BudgetExceeded(f"more than {budget} Decisions API calls")
        return original(*a, **k)

    D._live_call = counting  # type: ignore[assignment]

    mod = importlib.import_module(f"deepdive.examples.{gid}")
    payload = mod.EXAMPLE_INPUT
    if gid in INPUT_OVERRIDES and isinstance(payload, dict):
        payload = {**payload, **INPUT_OVERRIDES[gid]}
    counter, t0 = CallCounter(), time.time()
    try:
        graph = mod.build_graph(checkpointer=InMemorySaver(), store=make_store(), cache=InMemoryCache())
        res = run_graph(graph, payload, list(getattr(mod, "EXAMPLE_RESUMES", [])), config={"callbacks": [counter]})
        out = res.output if isinstance(res.output, dict) else {}
        if save_dir:  # keep the full output so a human can judge quality, not just "did not crash"
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{gid}.json"), "w") as fh:
                json.dump({"input": payload, "interrupts": [str(i) for i in res.interrupts], "output": res.output},
                          fh, indent=2, default=str)
        status = next((out[k] for k in ("status", "route", "outcome", "decision", "stop_reason") if k in out), None)
        return {"ok": True, "secs": time.time() - t0, "chat": counter.chat, "decisions": stats["decisions"],
                "interrupts": len(res.interrupts), "mode": out.get("mode"), "status": str(status)[:70]}
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return {"ok": False, "secs": time.time() - t0, "chat": counter.chat, "decisions": stats["decisions"],
                "error": f"{type(e).__name__}: {str(e)[:280]} [{os.path.basename(tb.filename)}:{tb.lineno}]"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--go", action="store_true", help="actually call OpenAI (costs money)")
    ap.add_argument("--all", action="store_true", help="run every registered example")
    ap.add_argument("--budget", type=int, default=40, help="max chat-model calls per example (default 40)")
    ap.add_argument("--timeout", type=int, default=240, help="wall-clock seconds per example (default 240)")
    ap.add_argument("--save", metavar="DIR", help="write each example's input and full output to DIR/<id>.json")
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: child-process mode
    args = ap.parse_args()

    if args.one:  # child: run a single example and print one machine-readable line
        print("RESULT " + json.dumps(run_one(args.one, args.budget, args.save)), flush=True)
        return 0

    registry = json.load(open(os.path.join(ROOT, "langgraph.json")))["graphs"]
    ids = args.ids or (sorted(registry) if args.all else RISKY)
    unknown = [i for i in ids if i not in registry]
    if unknown:
        print("unknown example ids:", unknown)
        return 2
    if not args.go:
        print(f"dry run - would call OpenAI for {len(ids)} examples (budget {args.budget} model calls, "
              f"{args.timeout}s each):\n  " + ", ".join(ids) + "\nre-run with --go to proceed.")
        return 0

    print(f"LIVE run: {len(ids)} examples, budget {args.budget} model calls and {args.timeout}s each\n", flush=True)
    failures = 0
    for gid in ids:
        t0 = time.time()
        try:
            child = [sys.executable, "-u", __file__, "--one", gid, "--budget", str(args.budget)]
            if args.save:
                child += ["--save", os.path.abspath(args.save)]
            p = subprocess.run(child, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"TIMEOUT {gid:26s} no result after {args.timeout}s (killed)", flush=True)
            continue
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            failures += 1
            print(f"CRASH {gid:26s} {time.time() - t0:5.1f}s  no result; stderr tail: {p.stderr.strip()[-300:]!r}", flush=True)
            continue
        r = json.loads(line[7:])
        counts = f"chat_calls={r.get('chat', 0):<3d} decisions_calls={r.get('decisions', 0):<3d}"
        if r["ok"] and r.get("mode") in (None, "live"):
            print(f"PASS {gid:26s} {r['secs']:5.1f}s  {counts} interrupts={r['interrupts']} mode={r['mode']} status={r['status']!r}", flush=True)
        else:
            failures += 1
            why = r.get("error") or f"mode={r.get('mode')} (expected live)"
            print(f"FAIL {gid:26s} {r.get('secs', 0):5.1f}s  {counts} {why}", flush=True)
    print(f"\n{len(ids) - failures}/{len(ids)} passed live", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

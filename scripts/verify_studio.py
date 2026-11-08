"""Boot a real `langgraph dev` server (what LangGraph Studio talks to) and run every example through its HTTP API.

This checks what unit tests cannot: that Studio can LOAD every graph in langgraph.json (imports, schemas, no blocking
I/O on the server's event loop) and that runs - including interrupt/resume - work through the actual API.

    uv run python scripts/verify_studio.py                  # all examples, offline (no OpenAI calls)
    uv run python scripts/verify_studio.py ticket_router    # just some
    uv run python scripts/verify_studio.py --live           # use your real OPENAI_API_KEY (costs money)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_module(target: str):
    path = target.split(":")[0].removeprefix("./src/").removesuffix(".py").replace("/", ".")
    return importlib.import_module(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="graph ids (default: all in langgraph.json)")
    ap.add_argument("--port", type=int, default=2035)
    ap.add_argument("--live", action="store_true", help="use real OpenAI calls instead of offline stand-ins")
    args = ap.parse_args()

    registry: dict[str, str] = json.loads((ROOT / "langgraph.json").read_text())["graphs"]
    ids = args.ids or sorted(registry)
    env = {**os.environ}
    if not args.live:
        env["DEEPDIVE_OFFLINE"] = "1"

    log = ROOT / ".langgraph_api" / "verify_studio.log"
    log.parent.mkdir(exist_ok=True)
    # Use the CLI that lives next to this interpreter (the project venv). A globally installed `langgraph` would run
    # in another Python without the `deepdive` package and fail to import the examples.
    sibling = Path(sys.executable).parent / "langgraph"
    exe = str(sibling) if sibling.exists() else (shutil.which("langgraph") or "langgraph")
    print(f"starting: {exe} dev --port {args.port}  ({'LIVE' if args.live else 'offline'})")
    with log.open("w") as lf:
        proc = subprocess.Popen(
            [exe, "dev", "--no-browser", "--no-reload", "--port", str(args.port)],
            cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    base = f"http://127.0.0.1:{args.port}"
    failures: list[str] = []
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{base}/ok", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                print("server exited early; log tail:\n" + "\n".join(log.read_text().splitlines()[-30:]))
                return 1
            time.sleep(1)
        else:
            print("server did not become ready; log tail:\n" + "\n".join(log.read_text().splitlines()[-30:]))
            return 1

        with httpx.Client(base_url=base, timeout=300) as c:
            found = {a["graph_id"]: a["assistant_id"] for a in c.post("/assistants/search", json={"limit": 200}).json()}
            assistants = set(found)
            missing = sorted(set(registry) - assistants)
            print(f"registered assistants: {len(assistants)}/{len(registry)}", f"MISSING: {missing}" if missing else "")
            failures += [f"{m}: not registered (failed to load)" for m in missing]

            # What Studio does the moment you open a graph, before any run: draw the canvas and build the input form.
            for gid in ids:
                if gid in missing:
                    continue
                for label, path in (("graph", f"/assistants/{found[gid]}/graph?xray=true"), ("schemas", f"/assistants/{found[gid]}/schemas")):
                    try:
                        r = c.get(path)
                        problem = None if r.status_code == 200 else f"HTTP {r.status_code} {r.text[:200]}"
                    except httpx.HTTPError as e:  # a server error can reset the connection; record it, keep going
                        problem = f"{type(e).__name__}: {e}"
                    if problem:
                        print(f"  FAIL {gid:28s} open/{label}: {problem}")
                        failures.append(f"{gid}: Studio cannot load {label} ({problem})")
            print(f"opened graph + schemas for {len([g for g in ids if g not in missing])} graphs (as Studio does on load)")

            for gid in ids:
                if gid in missing:
                    continue
                mod = load_module(registry[gid])
                resumes = list(getattr(mod, "EXAMPLE_RESUMES", []))
                thread = c.post("/threads", json={}).json()["thread_id"]
                t0 = time.time()
                try:
                    body = {"assistant_id": gid, "input": mod.EXAMPLE_INPUT}
                    n_interrupts = 0
                    while True:
                        r = c.post(f"/threads/{thread}/runs/wait", json=body)
                        if r.status_code >= 400:
                            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                        out = r.json()
                        if isinstance(out, dict) and out.get("__interrupt__"):
                            n_interrupts += 1
                            if not resumes:
                                raise RuntimeError("interrupted but EXAMPLE_RESUMES is exhausted")
                            body = {"assistant_id": gid, "command": {"resume": resumes.pop(0)}}
                            continue
                        break
                    keys = list(out) if isinstance(out, dict) else type(out).__name__
                    print(f"  ok   {gid:28s} {time.time() - t0:5.1f}s  interrupts={n_interrupts}  keys={keys}")
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL {gid:28s} {e}")
                    failures.append(f"{gid}: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Retried exceptions are logged with a traceback at INFO level (resilient_enrichment does this on purpose), so only
    # error-level records and blocking-I/O complaints count as problems.
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", ln) for ln in log.read_text().splitlines()]
    errs = [ln for ln in lines if "Blocking call" in ln or re.search(r"\[(error|critical)\s*\]", ln) or "GraphLoadError" in ln]
    retries = sum(1 for ln in lines if "Retrying task" in ln)
    if retries:
        print(f"\nserver log: {retries} intentional retry records (expected from resilient_enrichment)")
    if errs:
        print(f"\nserver log has {len(errs)} error-level lines (see {log.relative_to(ROOT)}), first few:")
        print("\n".join(e[:240] for e in errs[:5]))
        failures.append("server logged errors")
    print("\nRESULT:", "FAILED" if failures else "all examples load and run through the Studio API")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

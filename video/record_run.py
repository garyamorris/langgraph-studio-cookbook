#!/usr/bin/env python3
"""Record one REAL run of an example graph against a running `langgraph dev` server, as JSON.

WHY
    The video animates real runs. This script is the plain-Python counterpart of viz/live.js: it creates a thread, streams
    `POST /threads/{id}/runs/stream` with the same stream modes the viz uses (values, updates, tasks, checkpoints, custom,
    messages), answers any interrupt with a Command(resume=...) you supply, and writes everything that happened to
    runs/<name>.json: the input, the thread id, every event with its wall-clock time, each node's start/end, the
    interrupts and what resumed them, the checkpoints, and the final state. Use it to (1) probe an example before scripting
    captions for it, and (2) keep a durable record of the runs the video shows.

USAGE
    python3 record_run.py ticket_router                         # blank form = the first ready-made case
    python3 record_run.py refund_approval --scenario "RF-501 ..."  # pick a ready-made case by (prefix of) its label
    python3 record_run.py refund_approval --scenario "RF-9" --resume '{"approve": true}'
    python3 record_run.py report_orchestrator --input '{"quarter": "2026Q3"}' --out runs/report.json
    options: --api http://127.0.0.1:2125   --name NAME   --quiet   --max-rounds 6

The resume value is sent for EVERY interrupt of the run (use --resume-file for a JSON list, one per interrupt in order).
Nothing here touches secrets: the server holds the model key; this client only talks to the local API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["values", "updates", "tasks", "checkpoints", "custom", "messages-tuple"]


def call(api: str, method: str, path: str, body=None, timeout: float = 60):
    req = urllib.request.Request(api + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def sse(api: str, path: str, body: dict):
    """Yield (event, data) pairs from a text/event-stream POST."""
    req = urllib.request.Request(api + path, method="POST", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=900) as r:
        ev, data = None, []
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line == "":
                if ev is not None or data:
                    yield (ev or "message", "\n".join(data))
                ev, data = None, []
            elif line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" ") if line[5:6] == " " else line[5:])


def split(name: str):
    mode, _, ns = name.partition("|")
    return mode, [p.split(":")[0] for p in ns.split("|") if p]


def short(v, n=160):
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def find_case(api: str, graph_id: str, assistant_id: str, wanted: str) -> str:
    sch = call(api, "GET", f"/assistants/{assistant_id}/schemas")
    enum = (((sch.get("input_schema") or {}).get("properties") or {}).get("scenario") or {}).get("enum") or []
    hits = [e for e in enum if e == wanted] or [e for e in enum if e.lower().startswith(wanted.lower())] or [e for e in enum if wanted.lower() in e.lower()]
    if not hits:
        raise SystemExit(f"no ready-made case matching {wanted!r}; the dropdown has:\n  " + "\n  ".join(enum))
    return hits[0]


def record(api: str, graph_id: str, payload: dict, resumes: list, max_rounds: int, quiet: bool) -> dict:
    assistants = call(api, "POST", "/assistants/search", {"limit": 100})
    a = next((x for x in assistants if x["graph_id"] == graph_id), None)
    if not a:
        raise SystemExit(f"no graph {graph_id!r} on {api}; have: " + ", ".join(sorted(x["graph_id"] for x in assistants)))
    if payload.get("scenario"):
        payload = {**payload, "scenario": find_case(api, graph_id, a["assistant_id"], payload["scenario"])}
    thread = call(api, "POST", "/threads", {"metadata": {"graph": graph_id, "source": "record_run"}})
    tid = thread["thread_id"]
    T0 = time.time()
    rec = {"graph": graph_id, "assistant_id": a["assistant_id"], "thread_id": tid, "input": payload, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "events": [], "nodes": [], "checkpoints": [], "interrupts": [], "resumes": [], "errors": []}
    body = {"input": payload}
    open_tasks: dict = {}
    for rnd in range(max_rounds):
        pending = {}
        for event, data in sse(api, f"/threads/{tid}/runs/stream",
                               {"assistant_id": a["assistant_id"], "stream_mode": MODES, "stream_subgraphs": True,
                                "on_disconnect": "continue", "multitask_strategy": "enqueue", **body}):
            t = round(time.time() - T0, 3)
            mode, ns = split(event)
            try:
                d = json.loads(data) if data else None
            except json.JSONDecodeError:
                d = data
            rec["events"].append({"t": t, "round": rnd, "mode": mode, "ns": ns, "data": d if mode in ("tasks", "custom", "checkpoints", "updates", "error") else None})
            if mode == "tasks" and isinstance(d, dict) and d.get("name") and not d["name"].startswith("__"):
                key = d.get("id")
                if "result" in d or ("error" in d and "input" not in d):
                    st = open_tasks.pop(key, None)
                    n = {"name": d["name"], "ns": ns, "task": key, "end": t, "start": st["t"] if st else None, "round": rnd,
                         "error": d.get("error"), "interrupted": bool(d.get("interrupts"))}
                    rec["nodes"].append(n)
                    for it in d.get("interrupts") or []:
                        pending[it["id"]] = it["value"]
                        if not any(x["id"] == it["id"] for x in rec["interrupts"]):
                            rec["interrupts"].append({"id": it["id"], "node": d["name"], "t": t, "value": it["value"]})
                    if not quiet:
                        print(f"  {t:7.2f}s  done   {'/'.join(ns + [d['name']]):<38} {'INTERRUPT' if d.get('interrupts') else ''}{'ERROR ' + short(d['error']) if d.get('error') else ''}")
                else:
                    open_tasks[key] = {"t": t, "name": d["name"]}
                    if not quiet:
                        print(f"  {t:7.2f}s  start  {'/'.join(ns + [d['name']]):<38} input keys: {','.join((d.get('input') or {}).keys()) if isinstance(d.get('input'), dict) else type(d.get('input')).__name__}")
            elif mode == "checkpoints" and isinstance(d, dict) and not ns:
                rec["checkpoints"].append({"t": t, "step": (d.get("metadata") or {}).get("step"), "source": (d.get("metadata") or {}).get("source"), "next": d.get("next")})
            elif mode == "values" and not ns:
                rec["final_values"] = d
            elif mode == "custom" and not quiet:
                print(f"  {t:7.2f}s  custom {short(d, 120)}")
            elif mode == "error":
                rec["errors"].append(d)
                print("  ERROR:", short(d, 300))
        if not pending:
            break
        if not resumes:
            print(f"  paused at an interrupt (no --resume given):\n    {short(list(pending.values())[0], 600)}")
            break
        val = resumes[min(len(rec["resumes"]), len(resumes) - 1)]
        rec["resumes"].append({"t": round(time.time() - T0, 3), "value": val, "for": list(pending)})
        if not quiet:
            print(f"  --- interrupt: {short(list(pending.values())[0], 300)}\n  --- Command(resume={short(val, 200)})")
        body = {"command": {"resume": {k: val for k in pending}}}
    rec["seconds"] = round(time.time() - T0, 2)
    try:
        st = call(api, "GET", f"/threads/{tid}/state")
        rec["final_state"] = st.get("values")
        rec["next"] = st.get("next")
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        rec["errors"].append({"state": str(exc)})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--api", default=os.environ.get("VIDEO_API", "http://127.0.0.1:2125"))
    ap.add_argument("--input", default="{}", help="JSON input (default: blank form = the first ready-made case)")
    ap.add_argument("--scenario", help="pick a ready-made case by label (exact, prefix or substring)")
    ap.add_argument("--resume", action="append", help="JSON resume value for interrupts (repeat for several, in order)")
    ap.add_argument("--out", help="output json (default runs/<name>.json)")
    ap.add_argument("--name", help="file stem (default: the graph id)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=6)
    a = ap.parse_args()
    payload = json.loads(a.input)
    if a.scenario:
        payload["scenario"] = a.scenario
    resumes = [json.loads(r) for r in (a.resume or [])]
    print(f"recording {a.graph} on {a.api} ...")
    rec = record(a.api, a.graph, payload, resumes, a.max_rounds, a.quiet)
    out = a.out or os.path.join(HERE, "runs", f"{a.name or a.graph}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    fv = rec.get("final_state") or {}
    print(f"\n{a.graph}: {rec['seconds']}s, {len(rec['nodes'])} node runs, {len(rec['checkpoints'])} checkpoints, "
          f"{len(rec['interrupts'])} interrupt(s), mode={fv.get('mode')}, errors={len(rec['errors'])}\ninput: {short(rec['input'], 300)}\nwrote {os.path.relpath(out)}")
    print("final state keys:", ", ".join(fv))
    return 1 if rec["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

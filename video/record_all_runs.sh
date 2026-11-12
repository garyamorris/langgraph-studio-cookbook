#!/bin/bash
# Record one REAL run per featured example as JSON in runs/ (the same ready-made cases the video shows), then index them.
# Needs the dev server on :2125 (preview config `langgraph-dev-record`). Costs a few cents of OpenAI calls.
cd "$(dirname "$0")"
python3 record_run.py ticket_router        --scenario "T-2301"                         --name 01_ticket_router        --quiet
python3 record_run.py report_orchestrator  --scenario "2026Q2"                         --name 02_report_orchestrator  --quiet
python3 record_run.py draft_critic_loop    --scenario "Lumen Therm: 16 words"          --name 03_draft_critic_loop    --quiet
python3 record_run.py credit_decision_gate --scenario "L-3039" --resume '{"decision":"approve","reason":"Two late payments are explained and the income is stable."}' --name 04_credit_decision_gate --quiet
python3 record_run.py live_digest_stream   --scenario "Lumen Plug 4-pack, newest 40"   --name 05_live_digest_stream   --quiet
python3 record_run.py conversation_memory  --scenario "Pasted 5-message"               --name 06_conversation_memory  --quiet
python3 record_run.py memory_concierge     --scenario "Customer 'ria'"                 --name 07_memory_concierge     --quiet
python3 - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob("runs/0*.json")):
    r = json.load(open(f)); fv = r.get("final_state") or {}
    rows.append((os.path.basename(f), r["graph"], r["input"].get("scenario", "(blank form)")[:70], r["seconds"], len(r["nodes"]), len(r["checkpoints"]), len(r["interrupts"]), fv.get("mode"), len(r["errors"])))
with open("runs/README.md", "w") as fh:
    fh.write("# Recorded runs\n\nEach file is one real run against a local `langgraph dev` server (see `record_run.py`): the input, thread id, every stream event "
             "with its time, node start/end, checkpoints, interrupts and the Command(resume) that answered them, and the final state. These are the same "
             "ready-made cases the video shows; each video scene is its own live run of that case, so wording of model output differs between them.\n\n"
             "| file | graph | case | seconds | node runs | checkpoints | interrupts | mode | errors |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write("| " + " | ".join(str(x) for x in r) + " |\n")
print(open("runs/README.md").read())
PY

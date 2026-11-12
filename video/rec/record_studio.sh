#!/usr/bin/env bash
# Re-record the three LangGraph Studio clips -> video/raw/studio_run.mp4, studio_inspect.mp4, studio_timetravel.mp4
# Needs: the repo's LangGraph dev server on :2125 (ticket_router, real OpenAI key in its env) and rec built (video/rec/rec).
# Starts the local HTTPS reverse proxy (tlsproxy.py, :2128 -> :2125) if it is not already up and stops it again at the end.
# usage: record_studio.sh [run] [inspect] [timetravel]   (default: all three; "inspect" continues the thread of the last "run" take)
# Real model calls: clip 1 runs ticket_router once, a thread is created for clip 3, clip 3 re-runs two nodes from a checkpoint.
set -euo pipefail
V="$(cd "$(dirname "$0")/.." && pwd)"
REC="$V/rec/rec"; PRE="$V/rec/studio_prejs.js"; DEV="${DEV_PORT:-2125}"; TLS="${TLS_PORT:-2128}"
[ -f "$V/rec/tls/cert.pem" ] || { mkdir -p "$V/rec/tls"; openssl req -x509 -newkey rsa:2048 -nodes -keyout "$V/rec/tls/key.pem" -out "$V/rec/tls/cert.pem" -days 30 -subj "/CN=127.0.0.1" -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" 2>/dev/null; }
PROXY=""
if ! curl -sk -m 3 "https://127.0.0.1:$TLS/ok" >/dev/null 2>&1; then python3 "$V/rec/tlsproxy.py" "$TLS" "$DEV" >/dev/null & PROXY=$!; sleep 1.5; fi
trap '[ -n "$PROXY" ] && kill "$PROXY" 2>/dev/null || true' EXIT
ASSISTANT=$(curl -s -X POST "http://127.0.0.1:$DEV/assistants/search" -H 'content-type: application/json' -d '{"limit":100}' | python3 -c "import sys,json; print(next(a['assistant_id'] for a in json.load(sys.stdin) if a['graph_id']=='ticket_router'))")
BASE="https://smith.langchain.com/studio/thread?baseUrl=https%3A%2F%2F127.0.0.1%3A${TLS}&render=interact&mode=graph&assistantId=${ASSISTANT}"
rec() { local name="$1" url="$2"; shift 2; "$REC" --url "$url" --css 1440x810 --pre-js "$PRE" --director "$V/rec/director_studio_$name.js" --out "$V/raw/studio_$name.mp4" --max-seconds 60 --tail 1.5 "$@"; }
mkdir -p "$V/raw"
CLIPS="${*:-run inspect timetravel}"; want() { [[ " $CLIPS " == *" $1 "* ]]; }

# thread for the time-travel clip: created now so it is ~1 minute old when clip 3 runs (original branch reads "a minute ago" next to the fresh fork)
if want timetravel; then TT_THREAD="$(python3 "$V/rec/studio_thread.py" "$DEV" T-2301)"; TT_T0=$(date +%s); echo "time-travel thread: $TT_THREAD"; fi

if want run; then rec run "$BASE"; fi                          # clip 1: pick the case in the dropdown, Submit, watch the run
if want inspect; then
  RUN_THREAD="$(python3 -c "import json,sys; print(next(m['data']['threadId'] for m in json.load(open('$V/raw/studio_run.mp4.markers.json')) if m['name']=='thread'))")"
  echo "clip 1 thread: $RUN_THREAD"
  rec inspect "$BASE&threadId=${RUN_THREAD}#panes=56"          # clip 2: continues clip 1's thread
fi
if want timetravel; then
  while [ $(( $(date +%s) - TT_T0 )) -lt 70 ]; do sleep 2; done
  rec timetravel "$BASE&threadId=${TT_THREAD}#panes=56"        # clip 3: Re-run from here -> Fork 2 of 2, flip between forks
fi
ls -l "$V"/raw/studio_*.mp4

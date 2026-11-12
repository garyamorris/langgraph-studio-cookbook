#!/bin/bash
# Record the animated, REAL-run scenes of the video (viz/live_graph.html driven by viz/scenarios.js) with rec.
# Needs: the dev server (`langgraph-dev-record`, :2125) and the static server (`video-static`, :8725) running; see README.md.
#   ./record_scenes.sh                 all scenes
#   ./record_scenes.sh ticket loop     only these scenes
# Each scene writes viz_out/<name>.mp4 plus .markers.json (captions with times, thread id) and .meta.json (fps stats).
cd "$(dirname "$0")"
BASE=${VIDEO_STATIC:-http://127.0.0.1:8725}
mkdir -p viz_out
declare -a ALL=(overview concept_ticket live_ticket live_orchestrator live_loop live_gate live_stream live_memory_short live_memory_long)
want=("$@"); [ ${#want[@]} -eq 0 ] && want=("${ALL[@]}")
for n in "${want[@]}"; do
  n=${n#live_}; case "$n" in
    overview) page=overview.html; q=""; name=overview;;
    concept_ticket) page=live_graph.html; q="scenario=ticket&concept=1"; name=concept_ticket;;
    ticket) page=live_graph.html; q="scenario=ticket&speed=0.65"; name=live_ticket;;   # slowed: it is narrated for ~24 s
    *) page=live_graph.html; q="scenario=$n"; name="live_$n";;
  esac
  echo "== $name ($q)"
  ./rec/rec --url "$BASE/viz/$page?$q" --out "viz_out/$name.mp4" --director viz/director_wait.js --tail 1.5 --max-seconds 180 2> "viz_out/$name.log"
  tail -n 2 "viz_out/$name.log" | cut -c1-200
  python3 -c "import json;m=json.load(open('viz_out/$name.mp4.meta.json'));print('   ',{k:m[k] for k in m if k in('duration','frames','repeated_frames','avg_grab_ms','console_errors','stop_reason')})" 2>/dev/null
done

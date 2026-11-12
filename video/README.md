# How the Deep Dive walkthrough was made

`../docs/demo/langgraph-deepdive-walkthrough.mp4` (1080p, 4:07, 55 MB; **not in git**, it is ignored because of its size, re-make it with the commands below) and `-720p.mp4` (26 MB, in git). Both carry an English caption track
(toggle it in QuickTime / VLC / iOS); the same cues are beside them as `.srt`, `.vtt` and a readable `.transcript.md`. It explains how LangGraph executes, using **real runs of this repo's examples**, not mock-ups.

| what you see | where it comes from |
|---|---|
| Animated graphs (nodes lighting up, Send fan-out chips, loop badges, interrupt pause, checkpoint timeline, state cards, stream tape) | `viz/live_graph.html` runs the example **for real** on the local server (`POST /threads`, `POST /threads/{id}/runs/stream`) and animates the stream. Structure comes from `GET /assistants/{id}/graph?xray=true`, never from a drawing. What each scene runs and says is `viz/scenarios.js`. |
| The "23 graphs" grid | `viz/overview.html`: the real structure of nine graphs, names of the other fourteen. |
| LangGraph Studio footage | `rec/director_studio_*.js` drive the real Studio UI against the local server (through `rec/tlsproxy.py`, see `rec/README.md`). |
| Narration | OpenAI `gpt-4o-mini-tts`, voice `marin` (`narration/tts_config.json`), from `narration/script.json`. |
| Captions / transcript | `narration/captions.py`: the narration is transcribed with word timestamps and aligned to the script, so the captions use the script's exact wording and the voice's real timing. It also **checks the voice says the script** (`captions.py check`). whisper-1 is the timing source and is occasionally wrong about a word (scene I: it
heard "LandGraph" and "pointed"); `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` both heard that take exactly as scripted, so it was kept. |
| Recorded runs | `runs/*.json`, see `runs/README.md`. One real run per featured example: input, thread id, every stream event with its time, node starts and ends, checkpoints, interrupts and the resume that answered them, final state. |

## Pieces

* `rec/rec.swift` → `rec/rec`: renders a page offscreen in a `WKWebView` and writes H.264 itself (30 fps, 1920x1080). `director_*.js` script a page
  like a user. `rec/README.md` documents it. Build: `swiftc -O -swift-version 5 -suppress-warnings rec/rec.swift -o rec/rec`.
* `record_scenes.sh`: records every animated scene into `viz_out/` (plus `.markers.json` with the caption times and the thread id).
* `record_run.py` / `record_all_runs.sh`: record a real run as JSON (`runs/`). Also the way to probe an example before scripting a scene.
* `make_manifest.py`: the storyboard. Footage cuts and callouts are anchored to **words of the narration** (using the transcript's word times),
  so they stay in place when a sentence changes. Writes `manifest.json`.
* `build_video.py`: assembles the manifest in FilmCraft through its localhost control server (see its docstring). It uses control port **7979**, not
  FilmCraft's default 7878, so it never touches another project's FilmCraft instance. `verify_mp4.py` decodes the export and checks joins and audio sync.

## Re-making it

Needs: this repo's `.venv` (`uv sync`), `OPENAI_API_KEY` in the environment (the dev server and the narration use it), FilmCraft, Xcode's `swiftc`.

```bash
# 1. servers (Claude Code preview configs in .claude/launch.json): the dev server WITHOUT hot reload on :2125, and a static server for video/ on :8725
uv run langgraph dev --no-browser --no-reload --port 2125 --n-jobs-per-worker 8
python3 -m http.server 8725 --bind 127.0.0.1 --directory video

# 2. record the animated scenes and the real runs
video/record_scenes.sh                      # -> video/viz_out/*.mp4
video/record_all_runs.sh                    # -> video/runs/*.json

# 3. narration, its check, and caption cues
python3 video/narration/tts.py all          # -> narration/audio/*.wav, durations.json
python3 video/narration/captions.py all     # transcribe, check (exit 1 on a real mismatch), cues

# 4. assemble, verify, captions
python3 video/make_manifest.py
python3 video/build_video.py video/manifest.json docs/demo/langgraph-deepdive-walkthrough.mp4 --small
python3 video/verify_mp4.py docs/demo/langgraph-deepdive-walkthrough.mp4 --joins --sync
python3 video/narration/captions.py export --plan docs/demo/langgraph-deepdive-walkthrough.plan.json --out docs/demo/langgraph-deepdive-walkthrough
# 5. soft caption track (no re-encode), with any ffmpeg that has mov_text; then move the result over the .mp4
ffmpeg -i X.mp4 -i X.srt -map 0:v -map 0:a -map 1:0 -c copy -c:s mov_text -metadata:s:s:0 language=eng -disposition:s:0 default -movflags +faststart X.captioned.mp4
```

Notes learned the hard way: run the recording server with `--no-reload` (hot reload restarts the server when files under the repo change, which cuts a
run in half); the scenes avoid cases that move money (`refund_approval` writes a payout ledger to the dev Store); the memory scene's case seeds two notes for the customer `ria`, idempotently; the model's wording
differs on every run, so scene narration never quotes model output or counts that a model decides (the number of report sections varies).

Studio footage: `rec/record_studio.sh` records the three clips (`raw/studio_*.mp4`) end to end; its directors drive the real Studio UI (the recorder has no real
mouse, so `rec/studio_prejs.js` forces Studio's hover-only buttons visible; the clicks still go through Studio's own handlers). `rec/tls/` holds a throw-away
self-signed certificate for the local HTTPS proxy and is git-ignored.

#!/usr/bin/env python3
"""Write manifest.json for build_video.py from a readable scene spec, placing every callout and every footage cut on a
WORD of the narration instead of a guessed second.

PROBLEM
    A callout that says "Send API" should appear when the voice says "Send API", not 3 s later because the scene was
    retimed. Hand-typed seconds drift every time a sentence is edited or the voice changes.

DESIGN
    * SPEC below is the storyboard: per scene, the footage segments (clip, source range) and the callouts, each callout
      anchored to a PHRASE of the narration (`"at": "Send API"`), and a segment may start when a phrase is spoken
      (`"from": "Long term"`), which sets how long the previous clip stays on screen.
    * The word times come from the speech recogniser's transcript of the real narration wavs
      (narration/cache/stt/<ID>.<sha>.json, written by `narration/captions.py transcribe`), so they are where the voice
      really is. If a scene has no transcript yet, times fall back to proportional placement by character position.
    * Scene timing in build_video.py: scene = pad_before + narration + pad_after, narration starts at pad_before.

USAGE   python3 make_manifest.py [--out manifest.json] [--print]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAD_BEFORE, PAD_AFTER = 0.4, 0.6
LEAD = 0.25  # a callout appears this long before its word

# --------------------------------------------------------------------------------------------------- storyboard
# clip(): one footage segment. src = source file; a:b = source seconds; weight = share of the scene (default = by 'from').
def clip(path, a=0.0, b=None, **kw):
    d = {"type": "video", "path": path, "speed": "fit", "in": a}
    if b is not None:
        d["out"] = b
    d.update(kw)
    return d


def co(at, text, dur=4.5, style="tech", anchor="bl", sub=None, nth=1):
    d = {"at": at, "nth": nth, "dur": dur, "text": text, "style": style, "anchor": anchor}
    if sub:
        d["sub"] = sub
    return d


def spec(have_studio: dict) -> dict:
    """The viz scenes carry their own live captions (viz/scenarios.js, keyed to real events), so they get no FilmCraft callouts
    or tag chip on top; only the Studio footage, which has no annotation of its own, is annotated here.
    Studio source ranges (seconds) come from each clip's markers (raw/studio_*.mp4.markers.json):
      studio_run:        0.7 loaded · 3.7 dropdown open · 8.7 case picked · 14.6 submit · 15.3 thread · run lights the graph until ~22
      studio_inspect:    3.5 hover load_ticket · 4.8 hover triage · 6.3 triage expanded · 10.4 lexicon_hits · 14.7 View state
      studio_timetravel: 3.4 hover triage · 4.8 checkpoint hover (View state / Re-run from here) · 7.1 re-run · 13.6 finished · 17.0 fork 1 of 2"""
    S = have_studio
    scenes = [
        {"id": "A", "tag": None, "segments": [clip("viz_out/overview.mp4", 0, None)], "callouts": []},
        {"id": "B", "tag": None,
         "segments": [clip("viz_out/concept_ticket.mp4", 0, None, speed=0.85)] + (
             [clip("raw/studio_run.mp4", 2.4, 11.0, **{"from": "Open any example"})] if "run" in S else []),
         "callouts": ([co("blank form", "Ready-made case dropdown · blank form runs the first", 5.0, "tech")] if "run" in S else [])},
        {"id": "C", "tag": None, "segments": [clip("viz_out/live_ticket.mp4", 0, None)], "callouts": []},
        {"id": "D", "tag": None, "segments": [clip("viz_out/live_orchestrator.mp4", 0, None)], "callouts": []},
        {"id": "E", "tag": None, "segments": [clip("viz_out/live_loop.mp4", 0, None)], "callouts": []},
        {"id": "F", "tag": None, "segments": [clip("viz_out/live_gate.mp4", 0, None)], "callouts": []},
        {"id": "G", "tag": None, "segments": [clip("viz_out/live_stream.mp4", 0, None)], "callouts": []},
        {"id": "H", "tag": None,
         "segments": [clip("viz_out/live_memory_short.mp4", 0, None),
                      clip("viz_out/live_memory_long.mp4", 0, None, **{"from": "Long term"})],
         "callouts": []},
    ]
    if len(S) == 3:
        scenes.append({"id": "I", "tag": None,
                       "segments": [clip("raw/studio_run.mp4", 0.7, 5.2),
                                    clip("raw/studio_run.mp4", 12.0, 21.5, **{"from": "Choose a ready"}),
                                    clip("raw/studio_inspect.mp4", 3.3, 12.5, **{"from": "Then click"}),
                                    clip("raw/studio_timetravel.mp4", 3.3, 17.5, **{"from": "Or go back"})],
                       "callouts": [co("every example appears", "Studio · every example is a graph", 3.8),
                                    co("click any step", "Inspect any step's state", 4.5),
                                    co("go back to any checkpoint", "Re-run from a checkpoint · fork the thread", 5.5)]})
    scenes.append({"id": "J", "tag": None, "segments": [clip("viz_out/overview.mp4", 0, None)], "callouts": []})
    return {"scenes": scenes}


# --------------------------------------------------------------------------------------------------- word times
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_words(sid: str, script_text: str, dur: float) -> list[dict]:
    """[{'w','start','end'}] for a scene, newest transcript of the CURRENT wav, else proportional guesses."""
    wav = os.path.join(HERE, "narration", "audio", f"{sid}.wav")
    cands = []
    if os.path.exists(wav):
        sha = hashlib.sha1(open(wav, "rb").read()).hexdigest()
        cands = glob.glob(os.path.join(HERE, "narration", "cache", "stt", f"{sid}.{sha}.json"))
    if cands:
        d = json.load(open(cands[0]))
        return [{"w": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in d.get("words", [])]
    toks = script_text.split()
    total = sum(len(t) + 1 for t in toks)
    out, pos = [], 0
    for t in toks:
        out.append({"w": t, "start": dur * pos / total, "end": dur * (pos + len(t)) / total})
        pos += len(t) + 1
    return out


def find_phrase(words: list[dict], phrase: str, nth: int = 1) -> float | None:
    """Time of the nth occurrence of a phrase; 'a|b' tries alternatives (the recogniser may write '23' for 'twenty-three')."""
    if "|" in phrase:
        for alt in phrase.split("|"):
            t = find_phrase(words, alt.strip(), nth)
            if t is not None:
                return t
        return None
    want = [norm(t) for t in phrase.split() if norm(t)]
    have = [norm(w["w"]) for w in words]
    hits = 0
    for i in range(len(have)):
        j, k = i, 0
        while k < len(want) and j < len(have):
            if have[j] == "":
                j += 1
                continue
            if have[j] != want[k]:
                break
            j += 1
            k += 1
        if k == len(want):
            hits += 1
            if hits == nth:
                return words[i]["start"]
    return None


def build(out_path: str, quiet: bool = False) -> dict:
    script = {s["id"]: s["text"] for s in json.load(open(os.path.join(HERE, "narration", "script.json")))["scenes"]}
    durs = json.load(open(os.path.join(HERE, "narration", "durations.json")))
    studio = {}
    for key, name in (("run", "studio_run"), ("inspect", "studio_inspect"), ("timetravel", "studio_timetravel")):
        p = os.path.join(HERE, "raw", f"{name}.mp4")
        if os.path.exists(p):
            meta = json.load(open(p + ".meta.json")) if os.path.exists(p + ".meta.json") else {}
            studio[key] = (f"raw/{name}.mp4", 0.0, None)
    sp = spec(studio)
    problems = []
    for sc in sp["scenes"]:
        sid = sc["id"]
        dur = float(durs[sid])
        words = load_words(sid, script[sid], dur)
        sc["voiceover_wav"] = f"narration/audio/{sid}.wav"
        sc["pad_before"], sc["pad_after"] = PAD_BEFORE, PAD_AFTER
        # callouts: times relative to the scene start
        for c in sc["callouts"]:
            t = find_phrase(words, c["at"], c.pop("nth", 1))
            if t is None:
                problems.append(f"{sid}: phrase {c['at']!r} not found in the narration; placed at 40%")
                t = 0.4 * dur
            c["at"] = round(max(0.3, PAD_BEFORE + t - LEAD), 2)
            c["dur"] = round(min(c["dur"], PAD_BEFORE + dur + PAD_AFTER - c["at"] - 0.2), 2)
        # segments: 'from' phrase -> that segment starts when the phrase is spoken; weights = on-screen seconds
        segs = sc["segments"]
        starts = [0.0]
        for s in segs[1:]:
            ph = s.pop("from", None)
            t = find_phrase(words, ph) if ph else None
            if t is None:
                t = dur * len(starts) / len(segs)
                if ph:
                    problems.append(f"{sid}: segment phrase {ph!r} not found")
            starts.append(t)
        starts.append(dur)
        for i, s in enumerate(segs):
            s.pop("from", None)
            if len(segs) > 1:
                s["weight"] = round(starts[i + 1] - starts[i], 3)
    manifest = {
        "width": 1920, "height": 1080, "fps": 30,
        "narration_dir": "narration/audio",
        "preset": "Match Source – Adaptive Low Bitrate", "max_mb": 60, "audio_kbps": 128,
        "small": {"width": 1280, "height": 720, "max_mb": 28, "audio_kbps": 96},
        "xfade": 0.35, "narration_db": -3.0,
        "title": {"text": "LangGraph Deep Dive", "sub": "how 23 LangGraph examples actually run", "seconds": 3.5,
                  "bg": {"type": "video", "path": "viz_out/overview.mp4", "in": 10.5, "freeze": True}, "dim": -90},
        "outro": {"text": "Run it yourself", "sub": "uv run langgraph dev  ·  pick a Ready-made case  ·  Submit", "seconds": 3.5,
                  "bg": {"type": "video", "path": "viz_out/live_loop.mp4", "in": 29.5, "freeze": True}, "dim": -90},
        "scenes": sp["scenes"],
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    if not quiet:
        for p in problems:
            print("  WARNING:", p)
        print(f"wrote {os.path.relpath(out_path)}: {len(manifest['scenes'])} scenes, studio clips: {sorted(studio) or 'none yet'}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "manifest.json"))
    a = ap.parse_args()
    build(a.out)

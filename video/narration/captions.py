#!/usr/bin/env python3
"""Accurate transcripts + captions for the narrated walkthrough, and a check that the synthesised voice says the script.

PROBLEM
    The voice-over is synthesised from script.json, so the *script* is the authoritative text: its capitalisation, hyphens
    and spelling are what a viewer should read.  But three things can go wrong, and tts.py's bag-of-words `transcribe`
    cannot see any of them: (1) the voice skips, repeats or mispronounces a word (a mispronounced word is transcribed as a
    different word, an omitted one is simply absent); (2) captions are timed by guesswork, so they drift against the voice;
    (3) a recogniser's transcript is spelled however the recogniser likes ("Lang Graph", "23", "ready made"), so using it
    as the caption text puts typos on screen.

DESIGN
    * Ground truth for *timing* is a speech recogniser run on the real per-scene wav: OpenAI whisper-1 with verbose_json and
      word + segment timestamps, cached on disk by the sha1 of the wav bytes, so a rerun costs nothing and an edited wav is
      re-transcribed automatically.  No prompt/hint is passed (a script-shaped prompt would bias the recogniser toward the
      script and hide exactly the errors we are looking for).  If whisper-1 cannot give word times for the key, gpt-4o-transcribe
      supplies the text and word times are DERIVED by proportional allocation inside each recognised segment (reported loudly).
    * Ground truth for *wording* is the script.  Both texts are normalised the same way (lowercase, punctuation stripped,
      hyphens split, integers spoken as words, apostrophes dropped, a small alias table for legitimate variants such as
      "lang graph"/"langgraph" or "ready-made"/"ready made") and aligned token by token with difflib.SequenceMatcher
      (autojunk=False).  Replace blocks whose letters are identical once spaces are removed ("run time"/"runtime") are accepted as
      spelling variants; everything else is a REAL difference: substitution (script words -> other heard words),
      omission (script words not heard) or insertion (heard words not in the script).
    * Every script token then gets a start/end time: matched tokens use the recognised word's times (a word split by the
      normaliser, or merged by an alias, is cut/joined proportionally); substituted tokens share the span of the words heard in
      their place; omitted tokens are interpolated between the nearest matched neighbours, never outside the spoken span.
    * whisper-1 word edges are only good to ~0.1 s inside a phrase and can be 0.5 s out beside a pause (it folds the pause into a
      neighbouring word).  So the seam between two cues is snapped onto the real pause found in the wav itself (the longest run of
      10 ms frames below -34 dBFS next to the recognised gap), and the first/last cue are tied to the first/last sound.
    * Cues are cut from the script's own text: sentence ends first; a sentence over 80 characters is split by a small dynamic
      programme that prefers comma, then 'and'/'then' boundaries, balances the parts and keeps every cue within 2 lines of 42
      characters and 6.5 s.  Final times are integer milliseconds: >= 1.2 s, <= 6.5 s, never overlapping, >= 40 ms apart, first cue at
      the first spoken word, last cue ending 0.25 s after the last spoken word (capped at the end of the wav, which only carries
      ~0.2 s of tail, so the cap is what binds).  A cue lingers 0.15 s past its last word when the next cue leaves room.
    * `export` shifts the per-scene cues (times relative to each wav) onto the video's timeline, from a build_video.py plan
      (key `vo_start` of each entry of "scenes"), from explicit --offsets, or end to end (--sequential), validates the result
      (monotonic, no overlaps, <= 2 lines x 42 chars) and writes .srt, .vtt and a readable .transcript.md.
    The tool cannot hear: a recogniser that "autocorrects" a mangled word from context will hide it.  CLEAN means "a
    context-aware recogniser heard exactly the script", which is strong evidence, not a proof of perfect pronunciation.
    Two alias families exist purely because the recogniser, not the voice, differs from the script: British/American spelling
    ("labelled"/"labeled") and whisper-1's "LanGraph" (gpt-4o-transcribe hears "LangGraph" in the very same audio).  Edit ALIASES
    to tighten or loosen the filter; the accepted variants of every scene are listed in transcript_check.json.

USAGE (run from anywhere; python3 + stdlib only; OPENAI_API_KEY is read by tts.api_key() and never printed, logged or written)
    python3 captions.py transcribe [--only A,B] [--force] [--model whisper-1]   wavs -> cache/stt/<ID>.<sha1 of wav>.json
    python3 captions.py check      [--only A,B]     script vs what was heard; exit 1 on any real difference; writes transcript_check.json
    python3 captions.py cues       [--only A,B]     -> cues/<ID>.json  {"scene","duration","cues":[{"start","end","text"}]}  (seconds, relative to the wav)
    python3 captions.py all        [--only A,B] [--force]               transcribe + check + cues (cues are written even if check fails)
    python3 captions.py export --out BASENAME ( --plan video.plan.json | --offsets A=0.4,B=20.1 | --sequential [--gap 1.0] [--lead 0] )
                                                    -> BASENAME.srt, BASENAME.vtt, BASENAME.transcript.md
    python3 captions.py selftest                    offline asserts on the normaliser, aligner, cue splitter and writers
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:  # only the network commands need tts.py (API key, SSL context); the selftest must run anywhere
    import tts
except Exception as _exc:  # noqa: BLE001
    tts = None
    _TTS_ERROR = _exc

SCRIPT = os.path.join(HERE, "script.json")
AUDIO = os.path.join(HERE, "audio")
STT_DIR = os.path.join(HERE, "cache", "stt")
CUES_DIR = os.path.join(HERE, "cues")
REPORT = os.path.join(HERE, "transcript_check.json")

STT_MODEL = "whisper-1"
FALLBACK_MODEL = "gpt-4o-transcribe"

# caption rules
MAX_LINE, MAX_LINES = 42, 2
SPLIT_OVER = 80          # a sentence longer than this many characters is split into several cues
MIN_CUE_MS, MAX_CUE_MS = 1200, 6500
GAP_MS = 40              # minimum gap between consecutive cues
LINGER_MS = 150          # a cue stays up this long after its last word (when the next cue leaves room)
TAIL_MS = 250            # the last cue of a scene ends this long after the last spoken word

# Legitimate spelling variants (applied to BOTH sides after tokenising): canonical -> variants as token tuples.
# A plural (last token + "s") is generated for each, so "sub graphs" == "subgraphs".
ALIASES: dict[str, list[tuple[str, ...]]] = {
    # "langraph": whisper-1 spells the brand "LanGraph" in places where gpt-4o-transcribe hears "LangGraph" (scenes A, I), i.e.
    # recogniser spelling noise on an unknown word, not a dropped "g" in the voice.
    "langgraph": [("lang", "graph"), ("langraph",)],
    "uv": [("u", "v")],
    "id": [("i", "d")],
    "readymade": [("ready", "made")],
    "longterm": [("long", "term")],
    "shortterm": [("short", "term")],
    "openai": [("open", "ai")],
    "subgraph": [("sub", "graph")],
    "checkpoint": [("check", "point")],
}
# British spellings: recognisers write the American form of the very same spoken word ("labelled" is transcribed "labeled")
for _uk, _us in {"labelled": "labeled", "colour": "color", "behaviour": "behavior", "favourite": "favorite", "centre": "center",
                 "grey": "gray", "programme": "program", "catalogue": "catalog", "cancelled": "canceled", "modelling": "modeling",
                 "analyse": "analyze", "summarise": "summarize", "organise": "organize", "optimise": "optimize",
                 "recognise": "recognize", "licence": "license", "defence": "defense", "travelled": "traveled"}.items():
    ALIASES.setdefault(_us, []).append((_uk,))


# ================================================================ plumbing
class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def redact(s: str) -> str:
    """Remove anything that looks like (or is) the API key from text that may be printed."""
    try:
        key = tts.api_key() if tts else ""
    except (Exception, SystemExit):  # noqa: BLE001
        key = ""
    if key:
        s = s.replace(key, "***")
    s = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", s)
    return re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1***", s)


def _need_tts():
    if tts is None:
        raise RuntimeError(f"cannot import tts.py from {HERE}: {_TTS_ERROR}")
    return tts


def post_multipart(path: str, fields: list[tuple[str, str]], filename: str, data: bytes, mime: str = "audio/wav",
                   timeout: float = 300, retries: int = 3) -> dict:
    """POST multipart/form-data (repeated field names allowed) -> parsed JSON.  Retries 429/5xx/network errors."""
    t = _need_tts()
    boundary = "----captions" + uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields]
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                  f"Content-Type: {mime}\r\n\r\n").encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    last: ApiError | None = None
    for attempt in range(retries):
        req = urllib.request.Request(t.API + path, method="POST", data=body)
        req.add_header("Authorization", "Bearer " + t.api_key())  # header only; never printed or stored
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=t.SSL_CTX) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except ValueError:
                raise ApiError(0, f"POST {path}: response was not JSON: {redact(raw[:200].decode('utf-8', 'replace'))}") from None
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read().decode("utf-8", "replace")[:400])
            last = ApiError(exc.code, f"HTTP {exc.code} on POST {path}: {detail}")
            if exc.code not in (429, 500, 502, 503, 504):
                raise last from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = ApiError(0, f"network error on POST {path}: {redact(str(exc))}")
        if attempt + 1 < retries:
            time.sleep(3 * (attempt + 1))
    assert last is not None
    raise last


def load_script() -> list[dict]:
    with open(SCRIPT) as fh:
        return json.load(fh)["scenes"]


def wav_path(sid: str) -> str:
    return os.path.join(AUDIO, f"{sid}.wav")


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def sha1_file(path: str) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def stt_cache_path(sid: str, path: str) -> str:
    return os.path.join(STT_DIR, f"{sid}.{sha1_file(path)}.json")


def speech_bounds(path: str, thresh_db: float = -45.0) -> tuple[float, float]:
    """(first, last) second at which the wav is above thresh_db; used only when word times have to be derived."""
    a, sr = _need_tts().read_wav(path)
    th = int(32768 * 10 ** (thresh_db / 20))
    first = next((i for i, s in enumerate(a) if s > th or s < -th), 0)
    last = next((i for i in range(len(a) - 1, -1, -1) if a[i] > th or a[i] < -th), len(a) - 1)
    return first / sr, (last + 1) / sr


def silent_runs(path: str, frame_ms: int = 10, thresh_db: float = -34.0, min_ms: int = 90) -> list[tuple[float, float]]:
    """(start, end) seconds of every stretch where all 10 ms frames peak below thresh_db (-34 dBFS: the voice-over is normalised
    to ~-16 LUFS, so speech peaks sit near -6 dBFS while the reverb tail between sentences is below -38).  Interior runs must last min_ms;
    the leading/trailing run counts whatever its length (it marks where the voice starts and stops)."""
    a, sr = _need_tts().read_wav(path)
    win, th = int(sr * frame_ms / 1000), int(32768 * 10 ** (thresh_db / 20))
    quiet = [max(a[i:i + win]) < th and min(a[i:i + win]) > -th for i in range(0, len(a) - win + 1, win)]
    runs, i = [], 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if (j - i) * frame_ms >= min_ms or i == 0 or j == len(quiet):
            runs.append((i * frame_ms / 1000, len(a) / sr if j == len(quiet) else j * frame_ms / 1000))
        i = j
    return runs


def select(scenes: list[dict], only: str | None) -> list[dict]:
    if not only:
        return scenes
    want = {x.strip() for x in only.split(",") if x.strip()}
    unknown = want - {s["id"] for s in scenes}
    if unknown:
        raise SystemExit(f"unknown scene id(s): {', '.join(sorted(unknown))}")
    return [s for s in scenes if s["id"] in want]


# ================================================================ heard words (cache + derivation)
@dataclass
class Word:
    text: str
    start: float
    end: float


def derive_words(text: str, segments: list[dict] | None, lo: float, hi: float) -> list[Word]:
    """Word times without a word-level recogniser: share the words among the recognised segments by character share, then
    allocate each segment's span to its words in proportion to their length.  No segments -> one segment [lo, hi]."""
    words = text.split()
    if not words:
        return []
    segs = [(float(s["start"]), float(s["end"]), s.get("text", "")) for s in (segments or []) if float(s["end"]) > float(s["start"])]
    if not segs:
        segs = [(lo, hi, text)]
    weights = [max(1, len(re.sub(r"\s+", "", t))) for _, _, t in segs]
    total_w, total_c = float(sum(weights)), float(sum(len(w) for w in words))
    bounds, acc = [], 0.0  # cumulative character share at the end of each segment
    for w in weights:
        acc += w / total_w
        bounds.append(acc)
    groups: list[list[str]] = [[] for _ in segs]
    done = 0.0
    for w in words:
        mid = (done + len(w) / 2) / total_c
        done += len(w)
        k = next((i for i, b in enumerate(bounds) if mid <= b + 1e-9), len(segs) - 1)
        groups[k].append(w)
    out: list[Word] = []
    for (s, e, _), grp in zip(segs, groups):
        tot = float(sum(len(w) + 1 for w in grp))
        t = s
        for w in grp:
            d = (e - s) * (len(w) + 1) / tot
            out.append(Word(w, t, t + d))
            t += d
    return out


def heard_words(obj: dict, path: str) -> tuple[list[Word], str]:
    """Words (with times) from a cached transcription, and 'native' or 'derived'."""
    if obj.get("words"):
        return [Word(str(w["word"]).strip(), float(w["start"]), float(w["end"])) for w in obj["words"]], "native"
    segs = obj.get("segments") or (obj.get("_meta") or {}).get("segments")
    lo, hi = speech_bounds(path)
    return derive_words(obj.get("text", ""), segs, lo, hi), "derived"


# ================================================================ transcribe
def transcribe_one(path: str, model: str) -> dict:
    """One scene -> raw API response (+ "_meta").  whisper-1 first; gpt-4o-transcribe text + derived word times if it cannot."""
    data = open(path, "rb").read()
    name = os.path.basename(path)
    whisper_segments = None
    if model == STT_MODEL:
        fields = [("model", STT_MODEL), ("response_format", "verbose_json"), ("timestamp_granularities[]", "word"),
                  ("timestamp_granularities[]", "segment"), ("language", "en"), ("temperature", "0")]
        try:
            obj = post_multipart("/audio/transcriptions", fields, name, data)
            if obj.get("words"):
                obj["_meta"] = {"model": STT_MODEL, "word_timestamps": "native"}
                return obj
            whisper_segments = obj.get("segments")
            print(f"  note: {STT_MODEL} returned no word timestamps for {name}; falling back to {FALLBACK_MODEL}")
        except ApiError as exc:
            if exc.status not in (400, 403, 404):
                raise
            print(f"  note: {STT_MODEL} unavailable for word timestamps on this key ({redact(str(exc))[:160]}); "
                  f"falling back to {FALLBACK_MODEL}")
        model = FALLBACK_MODEL
    fields = [("model", model), ("response_format", "json"), ("language", "en"), ("temperature", "0")]
    obj = post_multipart("/audio/transcriptions", fields, name, data)
    obj["_meta"] = {"model": model, "word_timestamps": "derived", "segments": whisper_segments,
                    "note": "word times derived by proportional allocation inside each recognised segment"}
    return obj


def cmd_transcribe(a) -> int:
    os.makedirs(STT_DIR, exist_ok=True)
    rc = 0
    for sc in select(load_script(), a.only):
        sid, path = sc["id"], wav_path(sc["id"])
        if not os.path.exists(path):
            print(f"{sid}: MISSING {path}")
            rc = 1
            continue
        cache = stt_cache_path(sid, path)
        status = "cached"
        if a.force or not os.path.exists(cache):
            try:
                obj = transcribe_one(path, a.model)
            except ApiError as exc:
                print(f"{sid}: transcription failed: {redact(str(exc))}", file=sys.stderr)
                rc = 1
                continue
            with open(cache + ".tmp", "w") as fh:
                json.dump(obj, fh, indent=1)
            os.replace(cache + ".tmp", cache)
            status = "new"
        obj = json.load(open(cache))
        meta = obj.get("_meta") or {"model": STT_MODEL, "word_timestamps": "native"}
        words, how = heard_words(obj, path)
        nseg = len(obj.get("segments") or [])
        print(f"{sid}: {status:6} {meta['model']}  {len(words)} words, {nseg} segments, word times {how.upper()}"
              f"{'  (derived: proportional allocation inside each segment, less precise)' if how == 'derived' else ''}")
    return rc


# ================================================================ normalisation
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> list[str]:
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    for base, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= base:
            head, rest = divmod(n, base)
            return number_words(head) + [name] + (number_words(rest) if rest else [])
    raise AssertionError(n)


def pieces(surface: str) -> list[str]:
    """One raw word ("Twenty-three,", "23", "ready-made") -> normalised word tokens."""
    s = unicodedata.normalize("NFKD", surface)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[’‘`´']", "", s)           # it's -> its, don't -> dont (same sound, no apostrophe noise)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)         # 1,000 -> 1000
    s = s.replace("%", " percent ").replace("&", " and ")
    out: list[str] = []
    for m in re.finditer(r"[^\W\d_]+|\d+", s):             # letters / digits; everything else (punctuation, hyphens) separates
        t = m.group()
        out.extend(number_words(int(t)) if t.isdigit() else [t])
    return out


@dataclass
class Tok:
    """A normalised token and the raw items (script chunks / heard words) it came from."""
    text: str
    i0: int          # first raw item
    f0: float        # fraction of that item at which the token starts (a raw word can yield several tokens)
    i1: int          # last raw item
    f1: float        # fraction of that item at which the token ends
    aliased: bool = False


def _alias_table() -> dict[tuple[str, ...], str]:
    table: dict[tuple[str, ...], str] = {}
    for canon, variants in ALIASES.items():
        for v in variants:
            table[tuple(v)] = canon
            table[tuple(v[:-1]) + (v[-1] + "s",)] = canon + "s"
    return table


_ALIAS = _alias_table()
_ALIAS_MAX = max(len(k) for k in _ALIAS)


def merge_aliases(toks: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(toks):
        for n in range(min(_ALIAS_MAX, len(toks) - i), 0, -1):
            canon = _ALIAS.get(tuple(t.text for t in toks[i:i + n]))
            if canon:
                out.append(Tok(canon, toks[i].i0, toks[i].f0, toks[i + n - 1].i1, toks[i + n - 1].f1, True))
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return out


def tokenise(items: list[str]) -> list[Tok]:
    toks: list[Tok] = []
    for idx, surf in enumerate(items):
        ps = pieces(surf)
        tot = float(sum(len(p) for p in ps)) or 1.0
        acc = 0.0
        for p in ps:
            toks.append(Tok(p, idx, acc / tot, idx, (acc + len(p)) / tot))
            acc += len(p)
    return merge_aliases(toks)


def normalise_text(text: str) -> list[str]:
    """Normalised tokens of a free text (both sides of the comparison go through exactly this)."""
    return [t.text for t in tokenise(text.split())]


# ================================================================ alignment
def spread(lens: list[int], s: float, e: float) -> list[tuple[float, float]]:
    """Split [s, e] among items in proportion to their lengths."""
    tot = float(sum(lens)) or 1.0
    out, t = [], s
    for n in lens:
        d = (e - s) * n / tot
        out.append((t, t + d))
        t += d
    return out


def align(stoks: list[Tok], htoks: list[Tok]) -> list[tuple[str, int, int, int, int]]:
    """difflib opcodes over token text.  A 'replace' whose letters agree once spaces are removed is a 'variant' (accepted)."""
    a, b = [t.text for t in stoks], [t.text for t in htoks]
    blocks = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and "".join(a[i1:i2]) == "".join(b[j1:j2]):
            tag = "variant"
        blocks.append((tag, i1, i2, j1, j2))
    return blocks


def token_times(stoks: list[Tok], htoks: list[Tok], hwords: list[Word], blocks, lo: float, hi: float) -> list[tuple[float, float]]:
    """A (start, end) for every script token: heard times where matched/substituted, interpolated elsewhere, always in [lo, hi]."""
    def htime(t: Tok) -> tuple[float, float]:
        w0, w1 = hwords[t.i0], hwords[t.i1]
        return w0.start + t.f0 * (w0.end - w0.start), w1.start + t.f1 * (w1.end - w1.start)

    n = len(stoks)
    T: list[tuple[float, float] | None] = [None] * n
    for tag, i1, i2, j1, j2 in blocks:
        if tag == "equal":
            for k in range(i2 - i1):
                T[i1 + k] = htime(htoks[j1 + k])
        elif tag in ("variant", "replace") and j2 > j1:   # spoken words took the place of the script words: share their span
            s, e = htime(htoks[j1])[0], htime(htoks[j2 - 1])[1]
            for k, span in enumerate(spread([len(t.text) for t in stoks[i1:i2]], s, e)):
                T[i1 + k] = span
    # keep known times inside the speech and monotonic (recognisers occasionally overlap neighbouring words)
    prev_s, prev_e, last = lo, lo, None
    for k in range(n):
        if T[k] is None:
            continue
        s, e = max(lo, min(hi, T[k][0])), max(lo, min(hi, T[k][1]))
        s = max(s, prev_s)
        e = max(e, s)
        if last is not None and T[last][1] > s:
            T[last] = (T[last][0], max(T[last][0], s))
        T[k], prev_s, prev_e, last = (s, e), s, e, k
    # interpolate runs of untimed tokens between the nearest timed neighbours (edges: the spoken span)
    k = 0
    while k < n:
        if T[k] is not None:
            k += 1
            continue
        j = k
        while j < n and T[j] is None:
            j += 1
        a = T[k - 1][1] if k > 0 else lo
        b = T[j][0] if j < n else hi
        for m, span in enumerate(spread([len(t.text) for t in stoks[k:j]], a, max(a, b))):
            T[k + m] = span
        k = j
    return T  # type: ignore[return-value]


# ================================================================ sentences / chunks
ABBREV = {"e.g.", "i.e.", "vs.", "mr.", "mrs.", "ms.", "dr.", "no.", "approx."}
SENT_END = re.compile(r"[.!?][\"'”’)\]]*$")


def script_chunks(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def split_sentences(chunks: list[str]) -> list[tuple[int, int]]:
    out, start = [], 0
    for k, c in enumerate(chunks):
        if SENT_END.search(c) and c.lower() not in ABBREV:
            out.append((start, k + 1))
            start = k + 1
    if start < len(chunks):
        out.append((start, len(chunks)))
    return out


# ================================================================ analysis of one scene
@dataclass
class Analysis:
    sid: str
    text: str
    duration: float
    source: str                          # 'native' / 'derived' word times
    model: str
    chunks: list[tuple[str, int, int]]   # (text, char0, char1)
    stoks: list[Tok]
    hwords: list[Word]
    htoks: list[Tok]
    blocks: list
    times: list[tuple[float, float]]     # per script token
    speech: tuple[float, float]          # first / last heard word
    silences: list[tuple[float, float]] | None = None   # silent runs of the wav (None: unknown, no snapping)


def analyse(sid: str, text: str, hwords: list[Word], duration: float, source: str = "native", model: str = STT_MODEL,
            silences: list[tuple[float, float]] | None = None) -> Analysis:
    chunks = script_chunks(text)
    stoks = tokenise([c[0] for c in chunks])
    htoks = tokenise([w.text for w in hwords])
    blocks = align(stoks, htoks)
    lo = max(0.0, hwords[0].start) if hwords else 0.0
    hi = min(duration, hwords[-1].end) if hwords else duration
    times = token_times(stoks, htoks, hwords, blocks, lo, hi) if stoks else []
    return Analysis(sid, text, duration, source, model, chunks, stoks, hwords, htoks, blocks, times, (lo, hi), silences)


def load_analysis(sc: dict) -> Analysis | None:
    sid, path = sc["id"], wav_path(sc["id"])
    if not os.path.exists(path):
        return None
    cache = stt_cache_path(sid, path)
    if not os.path.exists(cache):
        return None
    obj = json.load(open(cache))
    words, how = heard_words(obj, path)
    return analyse(sid, sc["text"], words, wav_duration(path), how, (obj.get("_meta") or {}).get("model", STT_MODEL), silent_runs(path))


# ================================================================ check
def diffs_of(an: Analysis) -> list[dict]:
    sents = split_sentences([c[0] for c in an.chunks])
    chunk_sent = {k: si for si, (a, b) in enumerate(sents) for k in range(a, b)}
    out = []
    for tag, i1, i2, j1, j2 in an.blocks:
        if tag in ("equal", "variant"):
            continue
        kind = {"replace": "substitution", "delete": "omission", "insert": "insertion"}[tag]
        s_toks, h_toks = an.stoks[i1:i2], an.htoks[j1:j2]
        if s_toks:
            c0, c1 = s_toks[0].i0, s_toks[-1].i1
            anchor = c0
        else:  # insertion: heard words sit before script token i1
            anchor = an.stoks[i1].i0 if i1 < len(an.stoks) else len(an.chunks) - 1
            c0, c1 = anchor, anchor - 1
        h_surface = " ".join(w.text for w in an.hwords[h_toks[0].i0:h_toks[-1].i1 + 1]) if h_toks else ""
        s_surface = " ".join(c[0] for c in an.chunks[c0:c1 + 1]) if s_toks else ""
        if h_toks:
            t0, t1 = an.hwords[h_toks[0].i0].start, an.hwords[h_toks[-1].i1].end
        else:
            t0 = an.times[i1][0] if i1 < len(an.times) else an.speech[1]
            t1 = an.times[i2 - 1][1] if i2 - 1 < len(an.times) and i2 > i1 else t0
        sa, sb = sents[chunk_sent[min(anchor, len(an.chunks) - 1)]]
        left = " ".join(c[0] for c in an.chunks[sa:c0])
        right = " ".join(c[0] for c in an.chunks[max(c1 + 1, c0):sb])
        marked = f"{s_surface or '(nothing)'} -> {h_surface or '(nothing)'}"
        out.append({
            "kind": kind,
            "position": i1 + 1,                              # 1-based index of the first script word involved (insertion: the word after)
            "script_words": [t.text for t in s_toks],
            "heard_words": [t.text for t in h_toks],
            "script_surface": s_surface,
            "heard_surface": h_surface,
            "time": [round(t0, 2), round(t1, 2)],
            "char": [an.chunks[anchor][1] if an.chunks else 0, an.chunks[c1][2] if s_toks else (an.chunks[anchor][1] if an.chunks else 0)],
            "sentence": " ".join(c[0] for c in an.chunks[sa:sb]),
            "marked": " ".join(x for x in (left, f"[[{marked}]]", right) if x),
        })
    return out


def accepted_variants(an: Analysis) -> list[str]:
    """Differences in spelling that were accepted (alias table / compound words), for transparency."""
    out = []
    for tag, i1, i2, j1, j2 in an.blocks:
        if tag == "variant":
            out.append(f"{' '.join(t.text for t in an.stoks[i1:i2])} ~ {' '.join(t.text for t in an.htoks[j1:j2])}")
        elif tag == "equal":
            for k in range(i2 - i1):
                s, h = an.stoks[i1 + k], an.htoks[j1 + k]
                if s.aliased or h.aliased:
                    sur_s = " ".join(c[0] for c in an.chunks[s.i0:s.i1 + 1])
                    sur_h = " ".join(w.text for w in an.hwords[h.i0:h.i1 + 1])
                    out.append(f"{sur_s} ~ {sur_h}")
    return out


def scene_report(an: Analysis) -> dict:
    diffs = diffs_of(an)
    matched = sum(i2 - i1 for tag, i1, i2, _, _ in an.blocks if tag in ("equal", "variant"))
    return {
        "status": "CLEAN" if not diffs else "DIFFERENCES",
        "duration": round(an.duration, 3),
        "model": an.model,
        "word_times": an.source,
        "script_words": len(an.stoks),
        "heard_words": len(an.htoks),
        "matched_words": matched,
        "counts": {k: sum(1 for d in diffs if d["kind"] == k) for k in ("substitution", "omission", "insertion")},
        "accepted_variants": accepted_variants(an),
        "differences": diffs,
        "heard_text": " ".join(w.text for w in an.hwords),
        "script_text": an.text,
    }


def cmd_check(a) -> int:
    scenes = select(load_script(), a.only)
    report: dict = {}
    if a.only and os.path.exists(REPORT):
        try:
            report = json.load(open(REPORT)).get("scenes", {})
        except ValueError:
            report = {}
    rows, bad = [], 0
    for sc in scenes:
        an = load_analysis(sc)
        if an is None:
            print(f"{sc['id']}: NO TRANSCRIPT for the current {os.path.basename(wav_path(sc['id']))} (run: python3 captions.py transcribe)")
            report[sc["id"]] = {"status": "NO-TRANSCRIPT"}
            rows.append((sc["id"], 0.0, 0, 0, 0, (0, 0, 0), "NO-TRANSCRIPT"))
            bad += 1
            continue
        rep = scene_report(an)
        report[sc["id"]] = rep
        c = rep["counts"]
        rows.append((sc["id"], an.duration, rep["script_words"], rep["heard_words"], rep["matched_words"],
                     (c["substitution"], c["omission"], c["insertion"]), rep["status"]))
        acc = len(rep["accepted_variants"])
        if rep["status"] == "CLEAN":
            print(f"{sc['id']}: CLEAN        {rep['script_words']} words, all heard as scripted" + (f" ({acc} spelling variant{'s' if acc != 1 else ''} accepted)" if acc else ""))
        else:
            bad += 1
            n = len(rep["differences"])
            print(f"{sc['id']}: DIFFERENCES  {n} real difference{'s' if n != 1 else ''} "
                  f"({c['substitution']} substitution, {c['omission']} omission, {c['insertion']} insertion)")
            for d in rep["differences"]:
                print(f"     {d['kind']:12} word {d['position']:>3} @ {d['time'][0]:5.2f}s: "
                      f"\"{d['script_surface'] or '(nothing)'}\" -> \"{d['heard_surface'] or '(nothing)'}\"")
                print(f"                  {d['marked']}")
            if rep["matched_words"] < 0.85 * max(1, rep["script_words"]):
                print(f"     hint: only {rep['matched_words']} of {rep['script_words']} script words were heard: this audio is probably older than the "
                      f"script (script.json edited after synthesis?)  ->  python3 tts.py scenes --only {sc['id']} && python3 tts.py normalise")
    ordered = {k: report[k] for k in [s["id"] for s in load_script()] if k in report}
    with open(REPORT, "w") as fh:
        json.dump({"tool": "captions.py check", "alias_table": {k: [" ".join(v) for v in vs] for k, vs in ALIASES.items()},
                   "scenes": ordered,
                   "summary": {"scenes": len(ordered), "clean": sum(1 for r in ordered.values() if r.get("status") == "CLEAN"),
                               "with_differences": sum(1 for r in ordered.values() if r.get("status") == "DIFFERENCES")}},
                  fh, indent=1)
    print(f"\n{'scene':5} {'dur':>6} {'script':>6} {'heard':>6} {'match':>6} {'sub':>4} {'omit':>5} {'ins':>4}  verdict")
    for sid, dur, ns, nh, nm, (s, o, i), verdict in rows:
        print(f"{sid:5} {dur:5.1f}s {ns:6} {nh:6} {nm:6} {s:4} {o:5} {i:4}  {verdict}")
    print(f"\n{len(rows) - bad}/{len(rows)} scenes CLEAN -> {os.path.relpath(REPORT)}")
    return 1 if bad else 0


# ================================================================ cues
def wrap_lines(text: str, max_line: int = MAX_LINE, max_lines: int = MAX_LINES) -> list[str] | None:
    """One line if it fits, else two lines broken at the space nearest the middle (both <= max_line); None if impossible."""
    text = " ".join(text.split())
    if len(text) <= max_line:
        return [text]
    if max_lines < 2:
        return None
    best: tuple[float, int] | None = None
    for i, ch in enumerate(text):
        if ch == " " and i <= max_line and len(text) - i - 1 <= max_line:
            d = abs(i - len(text) / 2)
            if best is None or d < best[0]:
                best = (d, i)
    return None if best is None else [text[:best[1]], text[best[1] + 1:]]


def fits(text: str) -> bool:
    return len(text) <= SPLIT_OVER and wrap_lines(text) is not None


_STRONG_CONJ = {"and", "then"}
_WEAK_CONJ = {"but", "so", "or", "which", "because", "while", "when", "where", "if", "that"}
_PHRASE_START = {"for", "with", "to", "in", "on", "at", "by", "from", "into", "over", "under", "through", "about", "inside", "without"}
_WEAK_END = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "and", "or", "but",
             "into", "that", "this", "its", "their", "your", "every", "each", "any"}


def boundary_cost(prev: str, nxt: str) -> float:
    """Price of ending a cue after chunk `prev` and starting the next with `nxt` (lower = better place to break)."""
    if prev.rstrip("\"')”")[-1:] in (",", ";", ":"):
        return 0.0
    n = re.sub(r"\W+", "", nxt.lower())
    if n in _STRONG_CONJ:
        return 1.5
    if n in _WEAK_CONJ:
        return 2.5
    if n in _PHRASE_START and re.sub(r"\W+", "", prev.lower()) not in _WEAK_END:
        return 12.0
    return 14.0 + (4.0 if re.sub(r"\W+", "", prev.lower()) in _WEAK_END else 0.0)


def split_sentence(words: list[str], starts: list[float], ends: list[float], glued: list[bool]) -> list[tuple[int, int]]:
    """Partition one sentence (chunk list) into cue-sized runs: fewest, best-placed, balanced cuts that all fit.
    glued[k] True = chunk k and k+1 must stay together.  Minimises sum(5 per cue + imbalance) + boundary costs by DP: a cue is
    cheap, a break that is not at a comma/'and'/'then' is dear, so commas win even when that means one more cue."""
    n = len(words)
    txt = lambda i, j: " ".join(words[i:j])
    total = len(txt(0, n))
    target = total / max(1, math.ceil(total / SPLIT_OVER))

    def solve(max_dur: float | None, strict: bool) -> list[tuple[int, int]] | None:
        INF = float("inf")
        best = [INF] * (n + 1)
        prev = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j):
                if best[i] == INF or (i > 0 and glued[i - 1]):
                    continue
                t = txt(i, j)
                if strict and not fits(t):
                    continue
                if max_dur is not None and (ends[j - 1] - starts[i]) > max_dur:
                    continue
                c = best[i] + 5.0 + 0.004 * (len(t) - target) ** 2 + (boundary_cost(words[i - 1], words[i]) if i > 0 else 0.0)
                if len(t) < 14 or (ends[j - 1] - starts[i]) < MIN_CUE_MS / 1000:
                    c += 10.0
                if c < best[j]:
                    best[j], prev[j] = c, i
        if best[n] == INF:
            return None
        cuts, j = [], n
        while j > 0:
            cuts.append((prev[j], j))
            j = prev[j]
        return cuts[::-1]

    return (solve(MAX_CUE_MS / 1000, True) or solve(None, True) or solve(None, False) or [(0, n)])


@dataclass
class Part:
    text: str
    start: int        # ms (first word)
    end: int          # ms (last word)
    first_end: int    # ms: end of the first word / start of the last word (bounds for acoustic snapping)
    last_start: int


SNAP_EARLY_MS, SNAP_LATE_MS = 150, 150


MIN_PAUSE_MS, WIDE_PAUSE_MS = 100, 150


def snap_boundary(p: "Part", q: "Part", runs: list[tuple[int, int]]) -> bool:
    """Recognisers put word edges near, not on, pauses (a sentence start can be 0.5 s late).  Move the seam between consecutive
    cues p, q onto the real pause between them: the longest silent run overlapping the recognised gap (then a wider window),
    adjacent to p's last word and q's first word.  Continuous speech with no pause leaves the recognised times alone."""
    lo, hi = min(p.end, q.start), max(p.end, q.start)
    for slack, minlen in ((100, MIN_PAUSE_MS), (350, WIDE_PAUSE_MS)):
        cand = [(b - a, -abs((a + b) / 2 - (lo + hi) / 2), a, b) for a, b in runs
                if b - a >= minlen and b > lo - slack and a < hi + slack
                and a >= p.last_start - 200 and a > p.start + 300 and b <= q.first_end + 200 and b < q.end - 300]
        if cand:
            _, _, a, b = max(cand)
            p.end, q.start = a, b
            return True
    return False


def snap_start(s: int, runs: list[tuple[int, int]], first_end: int) -> int:
    """Recognisers fold pauses into word edges.  A start inside silence moves to where the voice resumes; a start a little after
    the voice resumed (late word) moves back to the onset.  Never past the end of the first word."""
    for a, b in runs:
        if a <= s <= b and b <= first_end:
            return b
    before = [b for a, b in runs if s - SNAP_EARLY_MS <= b < s]
    return max(before) if before else s


def snap_end(e: int, runs: list[tuple[int, int]], last_start: int) -> int:
    """An end inside silence moves back to where the voice stops; an end just before the voice stops moves forward to it."""
    for a, b in runs:
        if a <= e <= b and a >= last_start:
            return a
    after = [a for a, b in runs if e < a <= e + SNAP_LATE_MS and a >= last_start]
    return min(after) if after else e


def settle(parts: list[Part], duration_ms: int, speech_end_ms: int) -> list[dict]:
    """Final cue times in integer ms: merge cues that cannot reach 1.2 s, then linger/extend within the limits."""
    def final_end(ps: list[Part], i: int) -> int:
        p = ps[i]
        last = i == len(ps) - 1
        cap = duration_ms if last else ps[i + 1].start - GAP_MS
        want = (max(p.end, speech_end_ms) + TAIL_MS) if last else (p.end + LINGER_MS)
        return min(max(want, p.start + MIN_CUE_MS), cap, p.start + MAX_CUE_MS)

    changed = True
    while changed and len(parts) > 1:
        changed = False
        for i in range(len(parts)):
            if final_end(parts, i) - parts[i].start >= MIN_CUE_MS:
                continue
            for j in (i + 1, i - 1):
                if not 0 <= j < len(parts):
                    continue
                a, b = (parts[i], parts[j]) if j > i else (parts[j], parts[i])
                text = a.text + " " + b.text
                if fits(text) and b.end - a.start <= MAX_CUE_MS:
                    parts[min(i, j):max(i, j) + 1] = [Part(text, a.start, b.end, a.first_end, b.last_start)]
                    changed = True
                    break
            if changed:
                break
    cues = []
    for i, p in enumerate(parts):
        cues.append({"start": p.start, "end": final_end(parts, i), "text": p.text})
    for i, c in enumerate(cues):
        if c["end"] <= c["start"]:
            raise RuntimeError(f"cannot place cue {i + 1} ({c['text'][:30]!r}): no room between its neighbours")
    return cues


def make_cues(an: Analysis) -> dict:
    """Script text + per-token times -> {"scene", "duration", "cues": [...]} (seconds, relative to the wav)."""
    chunks = [c[0] for c in an.chunks]
    n = len(chunks)
    if not n or not an.hwords:
        return {"scene": an.sid, "duration": round(an.duration, 3), "cues": []}
    cs: list[float | None] = [None] * n
    ce: list[float | None] = [None] * n
    glued = [False] * max(0, n - 1)
    for tok, (s, e) in zip(an.stoks, an.times):
        ks = list(range(tok.i0, tok.i1 + 1))
        if len(ks) > 1:                                         # one token spanning several chunks ("Lang Graph")
            for k in ks[:-1]:
                glued[k] = True
        for k, (a, b) in zip(ks, spread([len(chunks[k]) for k in ks], s, e)):
            cs[k] = a if cs[k] is None else min(cs[k], a)
            ce[k] = b if ce[k] is None else max(ce[k], b)
    last_end = an.speech[0]
    for k in range(n):                                          # chunks without tokens (a lone dash): zero-width at the neighbour
        if cs[k] is None:
            cs[k] = ce[k] = last_end
        last_end = ce[k]
    runs = [(round(a * 1000), round(b * 1000)) for a, b in (an.silences or [])]
    parts: list[Part] = []
    for a, b in split_sentences(chunks):
        for i, j in split_sentence(chunks[a:b], cs[a:b], ce[a:b], glued[a:b - 1] + [False]):  # type: ignore[arg-type]
            k0, k1 = a + i, a + j - 1
            p = Part(" ".join(chunks[k0:k1 + 1]), round(cs[k0] * 1000), round(ce[k1] * 1000), round(ce[k0] * 1000), round(cs[k1] * 1000))
            parts.append(p)
    if runs and parts:
        for p, q in zip(parts, parts[1:]):
            snap_boundary(p, q, runs)
        parts[0].start = snap_start(parts[0].start, runs, parts[0].first_end)      # first sound of the scene
        parts[-1].end = snap_end(parts[-1].end, runs, parts[-1].last_start)        # last sound of the scene
    dur_ms = int(round(an.duration * 1000))
    cues = settle(parts, dur_ms, round(an.speech[1] * 1000))
    out = []
    for c in cues:
        lines = wrap_lines(c["text"])
        if lines is None:
            raise RuntimeError(f"scene {an.sid}: cue does not fit 2 x {MAX_LINE}: {c['text']!r}")
        out.append({"start": round(c["start"] / 1000, 3), "end": round(c["end"] / 1000, 3), "text": "\n".join(lines)})
    return {"scene": an.sid, "duration": round(an.duration, 3), "cues": out}


def cmd_cues(a) -> int:
    os.makedirs(CUES_DIR, exist_ok=True)
    rc = 0
    for sc in select(load_script(), a.only):
        an = load_analysis(sc)
        if an is None:
            print(f"{sc['id']}: NO TRANSCRIPT for the current wav (run: python3 captions.py transcribe)")
            rc = 1
            continue
        doc = make_cues(an)
        with open(os.path.join(CUES_DIR, f"{sc['id']}.json"), "w") as fh:
            json.dump(doc, fh, indent=1)
        cues = doc["cues"]
        ds = [c["end"] - c["start"] for c in cues]
        note = ""
        n_diff = sum(i2 - i1 + (j2 - j1) > 0 for tag, i1, i2, j1, j2 in an.blocks if tag in ("replace", "delete", "insert"))
        if n_diff:
            note = f"  (script and voice differ in {n_diff} place{'s' if n_diff != 1 else ''}: times near them are approximate; see check)"
        if an.source == "derived":
            note += "  [word times derived]"
        print(f"{sc['id']}: {len(cues)} cues, {min(ds):.2f}-{max(ds):.2f}s each, first {cues[0]['start']:.2f}s, "
              f"last ends {cues[-1]['end']:.2f}s of {doc['duration']:.2f}s{note}")
    return rc


def cmd_all(a) -> int:
    rc = cmd_transcribe(a)
    print()
    rc_check = cmd_check(a)
    print()
    rc_cues = cmd_cues(a)
    return 1 if (rc or rc_check or rc_cues) else 0


# ================================================================ export
def fmt_ts(ms: int, sep: str) -> str:
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def parse_offsets(spec: str) -> dict[str, float]:
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--offsets: expected ID=seconds, got {item!r}")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--offsets: {item!r} is not ID=seconds") from None
    return out


def load_plan_offsets(path: str, have: set[str]) -> dict[str, float]:
    """Narration start (key `vo_start`) of every plan scene that has captions; cards/scenes without narration are skipped."""
    plan = json.load(open(path))
    out: dict[str, float] = {}
    for sc in plan["scenes"]:
        if sc.get("vo_start") is None:
            continue
        if sc["id"] not in have:
            print(f"warning: plan scene {sc['id']} has narration but no cues/{sc['id']}.json (skipped)", file=sys.stderr)
            continue
        out[sc["id"]] = float(sc["vo_start"])
    return out


def validate_timeline(cues: list[dict]) -> None:
    """Raise if the cue list would make an invalid / unreadable caption file."""
    prev_end = None
    for i, c in enumerate(cues):
        if c["end"] <= c["start"]:
            raise ValueError(f"cue {i + 1}: end <= start")
        if prev_end is not None and c["start"] < prev_end + GAP_MS:
            raise ValueError(f"cue {i + 1} starts {c['start'] - prev_end} ms after the previous one ends (need >= {GAP_MS})")
        lines = c["text"].split("\n")
        if len(lines) > MAX_LINES or any(len(x) > MAX_LINE or not x for x in lines):
            raise ValueError(f"cue {i + 1} breaks the 2 x {MAX_LINE} rule: {c['text']!r}")
        prev_end = c["end"]


def build_timeline(offsets: dict[str, float], docs: dict[str, dict]) -> list[dict]:
    """Per-scene cues (seconds in the wav) -> one chronological list in integer ms on the video timeline."""
    out = []
    for sid, off in sorted(offsets.items(), key=lambda kv: kv[1]):
        for c in docs[sid]["cues"]:
            out.append({"start": round((off + c["start"]) * 1000), "end": round((off + c["end"]) * 1000), "text": c["text"], "scene": sid})
    out.sort(key=lambda c: c["start"])
    for i in range(len(out) - 1):  # scenes placed back to back: keep the 40 ms gap across the seam
        if out[i + 1]["start"] - out[i]["end"] < GAP_MS:
            out[i]["end"] = out[i + 1]["start"] - GAP_MS
    validate_timeline(out)
    return out


def render_srt(cues: list[dict]) -> str:
    return "".join(f"{i}\n{fmt_ts(c['start'], ',')} --> {fmt_ts(c['end'], ',')}\n{c['text']}\n\n" for i, c in enumerate(cues, 1))


def render_vtt(cues: list[dict]) -> str:
    body = "".join(f"{i}\n{fmt_ts(c['start'], '.')} --> {fmt_ts(c['end'], '.')}\n{html.escape(c['text'], quote=False)}\n\n"
                   for i, c in enumerate(cues, 1))
    return "WEBVTT\n\n" + body


def render_transcript(offsets: dict[str, float], script: dict[str, str]) -> str:
    lines = ["# Transcript", "", "Narration start times (mm:ss) on the video timeline; text is the exact script.", ""]
    for sid, off in sorted(offsets.items(), key=lambda kv: kv[1]):
        m, s = divmod(int(off), 60)
        lines += [f"## Scene {sid} - {m:02d}:{s:02d}", "", script[sid], ""]
    return "\n".join(lines)


def cmd_export(a) -> int:
    script = {s["id"]: s["text"] for s in load_script()}
    docs: dict[str, dict] = {}
    for sid in script:
        p = os.path.join(CUES_DIR, f"{sid}.json")
        if os.path.exists(p):
            docs[sid] = json.load(open(p))
    if not docs:
        raise SystemExit("no cues found: run `python3 captions.py cues` first")
    if a.plan:
        if not os.path.exists(a.plan):
            raise SystemExit(f"plan file {a.plan} does not exist yet; use --offsets A=0.4,B=20.1,... or --sequential instead")
        offsets = load_plan_offsets(a.plan, set(docs))
        plan = json.load(open(a.plan))
        for sc in plan["scenes"]:
            if sc["id"] in offsets and sc.get("vo_len") and abs(float(sc["vo_len"]) - docs[sc["id"]]["duration"]) > 0.05:
                print(f"warning: scene {sc['id']}: plan narration is {sc['vo_len']:.2f}s but cues were made for "
                      f"{docs[sc['id']]['duration']:.2f}s (wav changed? rerun `all`)", file=sys.stderr)
    elif a.offsets:
        offsets = parse_offsets(a.offsets)
        missing = sorted(set(offsets) - set(docs))
        if missing:
            raise SystemExit(f"--offsets names scene(s) without cues: {', '.join(missing)}")
    else:
        t, offsets = a.lead, {}
        for sid in script:
            if sid in docs:
                offsets[sid] = t
                t += docs[sid]["duration"] + a.gap
    if not offsets:
        raise SystemExit("no scene with narration to caption")
    skipped = sorted(set(docs) - set(offsets))
    if skipped:
        print(f"note: no start time for scene(s) {', '.join(skipped)}; not in the output", file=sys.stderr)
    for sid in offsets:  # stale cues: the wav was re-synthesised after `cues` ran
        p = wav_path(sid)
        if os.path.exists(p) and abs(wav_duration(p) - docs[sid]["duration"]) > 0.05:
            print(f"warning: cues/{sid}.json were made for {docs[sid]['duration']:.2f}s but the wav is now {wav_duration(p):.2f}s "
                  f"(rerun `python3 captions.py all`)", file=sys.stderr)
    cues = build_timeline(offsets, docs)
    base = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext, payload in ((".srt", render_srt(cues)), (".vtt", render_vtt(cues)), (".transcript.md", render_transcript(offsets, script))):
        with open(base + ext, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"wrote {base + ext}")
    print(f"{len(cues)} cues from {len(offsets)} scenes, {fmt_ts(cues[0]['start'], '.')} -> {fmt_ts(cues[-1]['end'], '.')}")
    return 0


# ================================================================ selftest (offline)
def cmd_selftest(_a=None) -> int:
    # normaliser: case, punctuation, hyphens, digits, aliases, apostrophes
    assert normalise_text("Twenty-three LangGraph examples, U V run; ready-made!") == \
        "twenty three langgraph examples uv run readymade".split()
    assert normalise_text("23 Lang graph UV ready made") == "twenty three langgraph uv readymade".split()
    assert normalise_text("thread I D") == normalise_text("thread ID") == normalise_text("thread I.D.") == ["thread", "id"]
    assert normalise_text("Short-term and long term sub graphs, check points, Open AI's") == \
        ["shortterm", "and", "longterm", "subgraphs", "checkpoints", "openais"]
    assert normalise_text("100 1,000 7 99 It's") == ["one", "hundred", "one", "thousand", "seven", "ninety", "nine", "its"]
    assert normalise_text("sixteen words, in a 10-word limit") == normalise_text("16 words in a ten word limit")

    # aligner: classification, aliases accepted, compound variants accepted, real errors not hidden
    def run(script: str, heard: str):
        hw = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(heard.split())]
        return analyse("T", script, hw, len(hw) * 0.5 + 0.3)

    an = run("The quick brown fox jumps over the lazy dog.", "the quick fox jumps over the the lazy dog")
    d = diffs_of(an)
    assert [(x["kind"], x["script_words"], x["heard_words"]) for x in d] == \
        [("omission", ["brown"], []), ("insertion", [], ["the"])], d
    d = diffs_of(run("A node returns an update.", "A note returns an update"))
    assert [(x["kind"], x["script_words"], x["heard_words"]) for x in d] == [("substitution", ["node"], ["note"])]
    assert diffs_of(run("Start with U V run, then sub-graphs.", "Start with UV run then subgraphs")) == []
    assert diffs_of(run("The long term plan is ready-made.", "the Long-term plan is ready made")) == []
    assert diffs_of(run("Honestly labelled, in LangGraph.", "honestly labeled in LanGraph")) == []   # UK/US spelling, recogniser spelling
    assert diffs_of(run("Calls the run time.", "calls the runtime")) == []            # compound variant
    assert [x["kind"] for x in diffs_of(run("Calls the run time.", "calls the runtimes"))] == ["substitution"]
    assert diffs_of(run("Twenty-three graphs.", "23 graphs")) == []
    assert [x["kind"] for x in diffs_of(run("Calls interrupt now.", "calls in to rupt now"))] == ["substitution"]

    # token times: matched = heard times; omitted word interpolated between neighbours and inside the speech
    an = run("The quick brown fox jumps.", "the quick fox jumps")      # words at 0.0-0.4, .5-.9, 1.0-1.4, 1.5-1.9
    t = dict(zip([x.text for x in an.stoks], an.times))
    assert t["the"] == (0.0, 0.4) and t["jumps"] == (1.5, 1.9)
    assert 0.9 <= t["brown"][0] <= t["brown"][1] <= 1.0 and len(an.times) == 5
    assert all(an.speech[0] <= s <= e <= an.speech[1] for s, e in an.times)

    # alias spanning two heard words gets the joined span; a split digit word is cut proportionally
    an = run("LangGraph rocks.", "Lang graph rocks")
    assert an.times[0] == (0.0, 0.9)
    an = run("Twenty-three graphs.", "23 graphs")
    assert an.times[0][0] == 0.0 and abs(an.times[0][1] - 0.4 * 6 / 11) < 1e-9 and abs(an.times[1][1] - 0.4) < 1e-9
    assert abs(an.times[2][0] - 0.5) < 1e-9

    # derived word times: proportional, inside segments, in order
    w = derive_words("aa bbbb cc dd", [{"start": 0, "end": 3, "text": "aa bbbb"}, {"start": 5, "end": 7, "text": "cc dd"}], 0, 9)
    assert [x.text for x in w] == ["aa", "bbbb", "cc", "dd"] and w[0].start == 0 and abs(w[1].end - 3) < 1e-9
    assert w[2].start == 5 and abs(w[3].end - 7) < 1e-9 and w[1].end <= w[2].start
    assert derive_words("x y", None, 1.0, 2.0)[0].start == 1.0

    # acoustic snapping: a start inside silence moves to the onset, an end inside silence to the offset; limits are respected
    runs = [(4670, 5270), (16050, 16200)]
    assert snap_start(4900, runs, 6000) == 5270 and snap_start(5380, runs, 6000) == 5270      # early inside pause / slightly late
    assert snap_start(5600, runs, 6000) == 5600 and snap_start(4900, runs, 5000) == 4900      # too late to trust / word ends first
    assert snap_end(5100, runs, 4000) == 4670 and snap_end(16000, runs, 15000) == 16050       # inside pause / just before the voice stops
    assert snap_end(15000, runs, 14000) == 15000 and snap_end(5100, runs, 5200) == 5100

    # seam snapping: the recogniser puts the next sentence 0.5 s late; the real pause (3.09-3.47) wins, a lone 100 ms dip does not
    p, q = Part("a", 0, 3000, 500, 2500), Part("b", 3960, 7000, 4200, 6500)
    assert snap_boundary(p, q, [(3090, 3470), (4000, 4180)]) and (p.end, q.start) == (3090, 3470)
    p, q = Part("a", 0, 3000, 500, 2500), Part("b", 3040, 7000, 3300, 6500)
    assert not snap_boundary(p, q, [(2200, 2250)]) and (p.end, q.start) == (3000, 3040)   # continuous speech: untouched

    # line wrapping
    assert wrap_lines("short text") == ["short text"]
    two = wrap_lines("This sentence is deliberately a little longer than one line allows")
    assert two and len(two) == 2 and all(len(x) <= MAX_LINE for x in two) and abs(len(two[0]) - len(two[1])) <= 12
    assert wrap_lines("x" * 43) is None

    # cue splitter: two long sentences + one short; words 0.3 s apart with pauses at sentence ends
    sent1 = ("In the report orchestrator, one node plans the sections, and the Send API launches a separate task for each of "
             "them at the same time, then a reducer merges the drafts.")
    sent2 = "Short one here."
    sent3 = "A final sentence that needs no split at all, really."
    text = f"{sent1} {sent2} {sent3}"
    words, t = [], 0.1
    for ch in text.split():
        words.append(Word(ch, t, t + 0.28))
        t += 0.3 + (0.35 if ch.endswith(".") else 0.0)
    an = analyse("S", text, words, t + 0.22)
    assert diffs_of(an) == []
    doc = make_cues(an)
    cues = doc["cues"]
    assert len(cues) >= 4
    joined = " ".join(c["text"].replace("\n", " ") for c in cues)
    assert joined == text, "cue text must be the script, verbatim"
    for i, c in enumerate(cues):
        lines = c["text"].split("\n")
        assert len(lines) <= 2 and all(len(x) <= 42 for x in lines), c
        assert 1.2 - 1e-9 <= c["end"] - c["start"] <= 6.5 + 1e-9, c
        if i:
            assert c["start"] - cues[i - 1]["end"] >= 0.04 - 1e-9, (cues[i - 1], c)
    assert cues[0]["start"] == 0.1 and abs(cues[-1]["end"] - min(words[-1].end + 0.25, doc["duration"])) < 2e-3
    assert any(c["text"].rstrip().endswith("Short one here.") for c in cues), "sentence ends must end a cue"
    assert any(c["text"].replace("\n", " ").endswith("sections,") or c["text"].replace("\n", " ").endswith("time,") for c in cues), \
        "long sentences should break at commas"
    # a very short sentence between two others is merged rather than shown for < 1.2 s
    words = [Word(w, i * 0.2, i * 0.2 + 0.15) for i, w in enumerate("Yes. No. Maybe so, we will see.".split())]
    doc = make_cues(analyse("Q", "Yes. No. Maybe so, we will see.", words, 2.0))
    assert all(c["end"] - c["start"] >= 1.2 - 1e-9 for c in doc["cues"]) and len(doc["cues"]) <= 2, doc

    # writers: SRT/VTT syntax, sequential numbering, comma vs dot, round trip
    tl = build_timeline({"S": 3600.5, "Q": 0.0}, {"S": {"cues": [{"start": 0.1, "end": 2.0, "text": "one\ntwo"}]},
                                                  "Q": {"cues": [{"start": 1.0, "end": 2.5, "text": "a & b"}]}})
    srt, vtt = render_srt(tl), render_vtt(tl)
    assert srt == "1\n00:00:01,000 --> 00:00:02,500\na & b\n\n2\n01:00:00,600 --> 01:00:02,500\none\ntwo\n\n", srt
    assert vtt.startswith("WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\na &amp; b\n\n") and "," not in vtt.split("\n")[3]
    assert fmt_ts(3723456, ",") == "01:02:03,456"
    assert parse_offsets("A=0.4, B=20.1") == {"A": 0.4, "B": 20.1}
    try:
        validate_timeline([{"start": 0, "end": 1000, "text": "x"}, {"start": 1010, "end": 2000, "text": "y"}])
        raise AssertionError("overlap/gap rule not enforced")
    except ValueError:
        pass
    try:
        validate_timeline([{"start": 0, "end": 1000, "text": "x" * 43}])
        raise AssertionError("line length rule not enforced")
    except ValueError:
        pass
    assert "api" not in redact("sk-abcdefghijklmnop1234 failed").lower() and "sk-***" in redact("sk-abcdefghijklmnop1234 failed")
    print("selftest: all assertions passed")
    return 0


# ================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="captions.py", description="Accurate transcripts/captions + voice-vs-script check.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("transcribe", "check", "cues", "all"):
        s = sub.add_parser(name)
        s.add_argument("--only", help="comma-separated scene ids, e.g. A,B")
        if name in ("transcribe", "all"):
            s.add_argument("--force", action="store_true", help="ignore the cache and transcribe again")
            s.add_argument("--model", default=STT_MODEL, choices=[STT_MODEL, FALLBACK_MODEL],
                           help=f"{FALLBACK_MODEL} = text only, word times derived (default {STT_MODEL}, which falls back by itself; "
                                f"the cache is shared, so add --force to replace an existing transcript)")
    s = sub.add_parser("export")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", help="plan json written by build_video.py (<out>.plan.json; key vo_start of each scene)")
    g.add_argument("--offsets", help="narration start times in seconds, e.g. A=0.4,B=20.1")
    g.add_argument("--sequential", action="store_true", help="scenes end to end, with --gap between them")
    s.add_argument("--gap", type=float, default=1.0, help="seconds between scenes for --sequential (default 1.0)")
    s.add_argument("--lead", type=float, default=0.0, help="seconds before the first scene for --sequential (default 0)")
    s.add_argument("--out", required=True, help="output basename: writes .srt, .vtt and .transcript.md")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    try:
        return {"transcribe": cmd_transcribe, "check": cmd_check, "cues": cmd_cues, "all": cmd_all,
                "export": cmd_export, "selftest": cmd_selftest}[a.cmd](a)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

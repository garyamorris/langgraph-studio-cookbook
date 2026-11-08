"""Classifier bake-off: keyword rules vs an LLM with structured output vs the Decisions API, measured on labelled headlines.

PROBLEM
    A trading-signal team wants to classify market headlines as positive, negative or neutral and is arguing about how:
    cheap keyword rules, an LLM that returns structured labels, or the Decisions API, which returns calibrated
    probabilities. Opinions are not evidence. The labelled file ``news_headlines.csv`` (180 headlines, about 15% deliberately
    tricky or mixed, such as "beats estimates but slashes outlook") lets the team measure instead: accuracy, accuracy on the
    hard headlines, how trustworthy each method's confidence is, how often the methods agree, and what each costs in time
    and tokens. This graph is that measurement as a repeatable harness.

DESIGN
    * One stratified, deterministic sample (``seed``). The strata are label x "hard", so the hard headlines (those containing
      "but", "despite", "yet", "only after", ";" and similar) are represented in the same proportion as in the full file even
      in a small sample. The same ``seed`` always gives the same headlines.
    * Three classifiers run as PARALLEL BRANCHES from the sample node and write to a shared list through a reducer. Each one
      receives only the headline texts, never the labels: the nodes declare an ``input_schema`` with a single field, so
      leakage is impossible by construction rather than by discipline.
        - ``rules_baseline``: a small lexicon of everyday finance words (written from general vocabulary, not tuned to this
          file). It is free and instant, has no notion of confidence, and by construction cannot read "good news, but...".
        - ``chat_structured``: an OpenAI chat model, 10 headlines per call, structured output
          ``{headline_id, label, confidence}`` per headline. Confidence is the model's own claim.
        - ``decisions_api``: ONE request with a ``choice`` question per headline (up to 400 are allowed; the sample is capped
          at 90). The probability of the chosen option is the confidence. Both LLM methods get the same label definitions, so
          the comparison is about the method, not about different instructions.
    * ``score`` is plain code: accuracy, macro-F1, confusion matrix, accuracy on the hard and easy subsets, and for the methods
      that give a confidence a top-label Brier score and a 5-bin expected calibration error (ECE), plus pairwise agreement.
      Everything is hand-coded (no sklearn) and unit-tested against hand-computed examples. A headline a method failed to
      answer counts as wrong, never silently dropped.
    * ``report`` renders a deterministic comparison table and asks a chat model for a short verdict. The verdict must quote the
      table: a guard checks that it repeats at least three of the table's numbers and otherwise replaces it with a verdict built
      by code. A branch that crashes (for example an API outage) is reported as ``failed`` and does not stop the others.

    Calibration definitions: top-label Brier = mean over answered headlines of (confidence - 1[correct])^2; ECE = sum over 5
    equal-width confidence bins of (bin share) x |bin accuracy - bin mean confidence|. Lower is better for both. The rules
    baseline has no confidence, so it has neither.

    IMPORTANT, about offline runs: offline, the chat and Decisions API stand-ins return arbitrary but stable labels, so the
    accuracy numbers you see offline are MEANINGLESS. The pipeline (sampling, batching, parallel branches, scoring, the
    report) is exercised; the numbers are not. A live run, with ``OPENAI_API_KEY`` set, is what answers the question; the
    output says ``mode`` and prints a caveat when it is offline. Also: the headlines are generated from about two dozen
    templates, so plain headlines are easy for every method; the hard subset is where methods differ, and with a sample of 30
    it holds only 5 or 6 headlines, so differences of a few points are noise. Use ``sample_size`` 90 for a firmer comparison.

GRAPH
    START -> sample --+--> rules_baseline --+
                      +--> chat_structured -+--> score -> report -> END
                      +--> decisions_api ---+
                        (each sees only the headline texts)

LANGGRAPH FEATURES
    * Parallel branches from one node and a join (``add_edge([...], "score")``) that waits for all three.
    * A reducer (``operator.add``) so parallel branches can append to the same state key.
    * Per-node ``input_schema`` to hide the labels from the classifiers.
    * The Decisions API at scale: one batched request, one question per headline, probabilities read per question.
    * Batching: 10 headlines per chat call, run concurrently, with usage tokens summed from the responses.
    * An evaluation harness expressed as a graph: sampling, runs, metrics and report as separate, testable nodes.

MODELS AND DATA
    OpenAI chat (fast tier) for ``chat_structured`` and for the verdict; the Decisions API for ``decisions_api``. Data:
    ``news_headlines.csv`` (180 rows: id, date, ticker, headline, label). Token counts come from the providers' usage fields;
    cost is not computed because prices change. Latency is wall-clock per branch, measured inside the node.

TRY IT IN STUDIO
    Input ``{"sample_size": 30, "seed": 0}``. Open ``score`` to see the metrics per classifier and ``report`` for the
    comparison table (markdown) and the verdict. Then:
      * ``{"sample_size": 90, "seed": 0}``: the largest run (90 Decisions questions, 9 chat calls).
      * the same ``sample_size`` with ``"seed": 1``: a different, equally stratified sample, to see how much the ranking moves.
    With a live key, read the ``hard acc`` column first: that is where the methods separate. Offline, ignore the numbers.

EXTEND IT
    Add a fourth branch (a small fine-tuned classifier, or the chat model with few-shot examples); weight the hard subset
    by what a wrong signal costs; or add a bootstrap confidence interval to ``score`` so the table says how big the noise is.
"""

from __future__ import annotations

import math
import operator
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from deepdive.common.config import is_offline, mode_label
from deepdive.common.data import read_csv
from deepdive.common.decisions import choice, decide
from deepdive.common.fakes import stable_int
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm

META = ExampleMeta(
    id="classifier_bakeoff",
    title="Classifier bake-off: rules vs LLM vs Decisions API",
    category="Decisions API",
    problem="Measure which approach (keyword rules, a structured-output LLM, or the Decisions API) should classify market "
    "headlines, on accuracy, hard cases, calibration, agreement, latency and tokens.",
    features=["parallel branches + reducer", "per-node input_schema", "batching", "Decisions API at scale", "evaluation harness"],
    models=["OpenAI chat", "OpenAI Decisions API"],
    datasets=["news_headlines.csv"],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"sample_size": 30, "seed": 0}

LABELS = ("positive", "negative", "neutral")
CLASSIFIERS = ("rules_baseline", "chat_structured", "decisions_api")
MAX_SAMPLE = 90  # also keeps the Decisions request far below its 400-question limit
MIN_SAMPLE = 3
CHAT_BATCH = 10
ECE_BINS = 5
OFFLINE_STANDINS = True  # offline only: label headlines from a stable hash when the stub returns no usable ids (tests turn it off)

LABEL_GUIDE = {
    "positive": "The news is likely to push the company's share price up.",
    "negative": "The news is likely to push the company's share price down.",
    "neutral": "No clear net direction: routine corporate news, or good and bad news that roughly offset.",
}

# "Hard" headlines mix good and bad news. Markers are generic connectives, not the labels.
_HARD = re.compile(r"\b(?:but|despite|yet|only after|although|though)\b|;", re.I)


def is_hard(headline: str) -> bool:
    return bool(_HARD.search(headline))


# --------------------------------------------------------------------------- state
class BakeoffIn(TypedDict):
    sample_size: NotRequired[int]  # default 30, max 90
    seed: NotRequired[int]  # default 0


class ClassifierInput(TypedDict):
    headlines: list[dict]  # [{headline_id, headline}] and nothing else: no labels, no hard flags


class BakeoffOut(TypedDict):
    mode: str
    caveat: str
    sample: dict
    metrics: dict
    table: str
    verdict: str
    verdict_source: str


class BakeoffState(TypedDict, total=False):
    sample_size: int
    seed: int
    headlines: list[dict]
    truth: dict  # headline_id -> label (only `score` reads it)
    hard_ids: list[str]
    sample: dict
    results: Annotated[list[dict], operator.add]  # one entry per classifier branch
    metrics: dict
    caveat: str
    table: str
    verdict: str
    verdict_source: str
    mode: str


# --------------------------------------------------------------------------- sampling
def stratified_sample(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample of ``n`` rows, proportional per (label, hard) stratum (largest-remainder allocation)."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["label"], r["hard"])].append(r)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(rows) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (quota[k] - alloc[k], k), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    chosen: list[dict] = []
    for k in keys:
        pool = sorted(strata[k], key=lambda r: r["headline_id"])
        chosen += random.Random(f"{seed}|{k}").sample(pool, alloc[k])
    return sorted(chosen, key=lambda r: r["headline_id"])


def sample(state: BakeoffState) -> dict:
    requested = state.get("sample_size")
    n = max(MIN_SAMPLE, min(30 if requested is None else int(requested), MAX_SAMPLE))
    seed = int(state.get("seed") or 0)
    df = read_csv("news_headlines.csv")
    rows = [
        {"headline_id": r["headline_id"], "headline": r["headline"], "label": r["label"], "hard": is_hard(r["headline"])}
        for r in df.to_dict("records")
    ]
    chosen = stratified_sample(rows, n, seed)
    return {
        "headlines": [{"headline_id": r["headline_id"], "headline": r["headline"]} for r in chosen],
        "truth": {r["headline_id"]: r["label"] for r in chosen},
        "hard_ids": [r["headline_id"] for r in chosen if r["hard"]],
        "sample": {
            "n": len(chosen),
            "seed": seed,
            "n_hard": sum(r["hard"] for r in chosen),
            "label_counts": {lab: sum(r["label"] == lab for r in chosen) for lab in LABELS},
            "population": len(rows),
        },
        "results": Overwrite([]),  # a reused thread must not accumulate results across runs
        "mode": mode_label(),
    }


# --------------------------------------------------------------------------- classifier 1: keyword rules
POSITIVE_WORDS = frozenset(
    "beat beats upgrade upgrades upgraded raises raised lifts record surge surges soar soars wins contract buyback dividend "
    "outperform growth profit strong expansion accelerates boost improve improves approval outpaces gains higher".split()
)
NEGATIVE_WORDS = frozenset(
    "miss misses missed downgrade downgrades downgraded cuts cut slashes warns warning recall recalls probe investigation lawsuit "
    "resigns resignation plunge plunges falls fall drop drops loss losses weak weakening shortage shortages restatement fraud "
    "bankruptcy layoffs charge weighs decline abruptly".split()
)


def rules_label(headline: str) -> str:
    """Count lexicon hits; the larger side wins, a tie (including 0-0) is neutral. No negation, no clause structure."""
    tokens = re.findall(r"[a-z']+", headline.lower())
    pos, neg = sum(t in POSITIVE_WORDS for t in tokens), sum(t in NEGATIVE_WORDS for t in tokens)
    return "positive" if pos > neg else "negative" if neg > pos else "neutral"


def rules_baseline(state: ClassifierInput) -> dict:
    started = time.perf_counter()
    preds = [{"headline_id": h["headline_id"], "label": rules_label(h["headline"]), "confidence": None} for h in state["headlines"]]
    return {"results": [{"name": "rules_baseline", "predictions": preds, "latency_s": time.perf_counter() - started, "calls": 0,
                         "usage": {}, "notes": ["no confidence: a lexicon has none"]}]}


# --------------------------------------------------------------------------- classifier 2: chat model, structured output
class LabelItem(BaseModel):
    headline_id: str = Field(description="The id exactly as given, e.g. H-0123")
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(description="Your probability, from 0.0 to 1.0, that the label is correct")


class BatchLabels(BaseModel):
    items: list[LabelItem] = Field(description="Exactly one entry per input headline, in any order")


CHAT_PROMPT = (
    "You label financial news headlines for a trading-signal pipeline. For each headline decide its likely net effect on the "
    "company's share price, judging only from the headline text:\n"
    + "\n".join(f"- {lab}: {text}" for lab, text in LABEL_GUIDE.items())
    + "\nReturn exactly one entry per headline, using its headline_id unchanged. Do not skip headlines and do not add any."
)


def _label_batch(batch: list[dict]) -> tuple[list[dict], dict, str | None]:
    """One structured chat call for up to CHAT_BATCH headlines -> (items, usage, error)."""
    listing = "\n".join(f"{h['headline_id']} | {h['headline']}" for h in batch)
    try:
        out = get_llm("fast").with_structured_output(BatchLabels, include_raw=True).invoke(
            [SystemMessage(CHAT_PROMPT), HumanMessage(f"Headlines:\n{listing}")]
        )
    except Exception as e:  # noqa: BLE001 - one failed batch must not discard the others
        return [], {}, f"{type(e).__name__}: {e}"[:200]
    parsed = out.get("parsed") if isinstance(out, dict) else out
    raw = out.get("raw") if isinstance(out, dict) else None
    usage = dict(getattr(raw, "usage_metadata", None) or {})
    if parsed is None:
        return [], usage, f"unparseable output: {out.get('parsing_error') if isinstance(out, dict) else ''}"[:200]
    return [i.model_dump() for i in parsed.items], usage, None


def chat_structured(state: ClassifierInput) -> dict:
    started = time.perf_counter()
    headlines = state["headlines"]
    batches = [headlines[i : i + CHAT_BATCH] for i in range(0, len(headlines), CHAT_BATCH)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_label_batch, batches))

    preds: list[dict] = []
    notes: list[str] = []
    usage_total: dict[str, int] = {}
    for n, (batch, (items, usage, error)) in enumerate(zip(batches, outcomes), start=1):
        wanted = {h["headline_id"] for h in batch}
        got: dict[str, dict] = {}
        for it in items:
            if it["headline_id"] in wanted and it["headline_id"] not in got:  # unknown and duplicate ids are ignored
                got[it["headline_id"]] = it
        if error:
            notes.append(f"batch {n} failed: {error}")
        elif not got and is_offline() and OFFLINE_STANDINS:
            # The offline stub returns placeholder ids. Label from a stable hash so the metrics below can be exercised.
            for h in batch:
                got[h["headline_id"]] = {"label": LABELS[stable_int(h["headline"]) % 3], "confidence": 0.5}
            notes.append(f"batch {n}: OFFLINE stand-in labels (stable hash of the headline)")
        if len(got) < len(wanted) and not error:
            notes.append(f"batch {n}: {len(wanted) - len(got)} of {len(wanted)} headlines missing from the model output")
        for h in batch:
            it = got.get(h["headline_id"])
            preds.append(
                {
                    "headline_id": h["headline_id"],
                    "label": it["label"] if it else None,
                    "confidence": min(max(float(it["confidence"]), 0.0), 1.0) if it else None,
                }
            )
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            usage_total[k] = usage_total.get(k, 0) + int(usage.get(k, 0) or 0)
    return {"results": [{"name": "chat_structured", "predictions": preds, "latency_s": time.perf_counter() - started,
                         "calls": len(batches), "usage": usage_total, "notes": notes}]}


# --------------------------------------------------------------------------- classifier 3: Decisions API, one request
def question_name(headline_id: str) -> str:
    """Letters, digits and underscores only: the safest shape for a question identifier. 'H-0062' -> 'q_H_0062'."""
    return "q_" + re.sub(r"[^A-Za-z0-9]+", "_", headline_id)


def decisions_api(state: ClassifierInput) -> dict:
    started = time.perf_counter()
    headlines = state["headlines"]
    document = "Market headlines to classify:\n" + "\n".join(f"{h['headline_id']} | {h['headline']}" for h in headlines)
    questions = [
        choice(
            question_name(h["headline_id"]),
            f"What is the likely net effect of headline {h['headline_id']} ('{h['headline']}') on the company's share price?",
            LABEL_GUIDE,
        )
        for h in headlines
    ]
    res = decide(document, questions)  # ONE request, one question per headline
    preds, notes = [], []
    for h in headlines:
        name = question_name(h["headline_id"])
        label = res.pick(name)
        if res.refused(name) or label not in LABELS:
            preds.append({"headline_id": h["headline_id"], "label": None, "confidence": None})
            continue
        probs = res.answers[name].probabilities
        preds.append({"headline_id": h["headline_id"], "label": label, "confidence": probs.get(label, res.conf(name)),
                      "probabilities": {k: round(v, 4) for k, v in probs.items()}})
    if (missing := sum(p["label"] is None for p in preds)):
        notes.append(f"{missing} questions were refused or unanswered")
    return {"results": [{"name": "decisions_api", "predictions": preds, "latency_s": time.perf_counter() - started, "calls": 1,
                         "usage": res.usage, "notes": notes + [f"{len(questions)} questions in one request; model {res.model}"]}]}


# --------------------------------------------------------------------------- metrics (hand-coded, no sklearn)
def accuracy(y_true: Sequence[str], y_pred: Sequence[str | None]) -> float | None:
    """Share of exact matches; a missing prediction (None) is wrong. None if there is nothing to score."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else None


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str] = LABELS) -> dict[str, dict[str, int]]:
    """Rows = true label, columns = predicted label plus ``none`` for unanswered items."""
    cols = [*labels, "none"]
    m = {t: {c: 0 for c in cols} for t in labels}
    for t, p in zip(y_true, y_pred):
        m[t][p if p in labels else "none"] += 1
    return m


def per_class_scores(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str] = LABELS) -> dict[str, dict[str, float]]:
    out = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        predicted = sum(p == lab for p in y_pred)
        actual = sum(t == lab for t in y_true)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[lab] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return out


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str | None], labels: Sequence[str] = LABELS) -> float:
    """Unweighted mean of per-class F1 over ``labels``; a class with undefined precision or recall contributes 0."""
    scores = per_class_scores(y_true, y_pred, labels)
    return sum(s["f1"] for s in scores.values()) / len(labels)


def brier_top(confidence: Sequence[float], correct: Sequence[bool]) -> float | None:
    """Top-label Brier score: mean of (confidence - correct)^2. 0 is perfect; always guessing 0.5 scores 0.25."""
    return sum((c - float(ok)) ** 2 for c, ok in zip(confidence, correct)) / len(confidence) if confidence else None


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS) -> dict[str, Any] | None:
    """ECE with equal-width bins on [0, 1] (the last bin includes 1.0). Returns the value and the per-bin table."""
    if not confidence:
        return None
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for c, ok in zip(confidence, correct):
        grouped[min(int(c * bins), bins - 1)].append((c, ok))
    total, ece, table = len(confidence), 0.0, []
    for b, items in enumerate(grouped):
        lo, hi = b / bins, (b + 1) / bins
        if items:
            acc, conf = sum(ok for _, ok in items) / len(items), sum(c for c, _ in items) / len(items)
            ece += len(items) / total * abs(acc - conf)
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(items), "accuracy": round(acc, 3), "mean_confidence": round(conf, 3)})
        else:
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0, "accuracy": None, "mean_confidence": None})
    return {"ece": ece, "bins": table}


def pairwise_agreement(preds: dict[str, dict[str, str | None]]) -> dict[str, float | None]:
    """Share of items on which two classifiers give the SAME label (both must have answered)."""
    names = sorted(preds)
    out: dict[str, float | None] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = [k for k in preds[a] if preds[a][k] is not None and preds.get(b, {}).get(k) is not None]
            out[f"{a} vs {b}"] = sum(preds[a][k] == preds[b][k] for k in ids) / len(ids) if ids else None
    return out


def classifier_metrics(result: dict, ids: list[str], truth: dict[str, str], hard_ids: set[str]) -> dict:
    by_id = {p["headline_id"]: p for p in result["predictions"]}
    y_true = [truth[i] for i in ids]
    y_pred = [by_id.get(i, {}).get("label") for i in ids]
    hard_idx = [k for k, i in enumerate(ids) if i in hard_ids]
    easy_idx = [k for k, i in enumerate(ids) if i not in hard_ids]
    sub = lambda idx: accuracy([y_true[k] for k in idx], [y_pred[k] for k in idx])  # noqa: E731
    scored = [(by_id[i]["confidence"], by_id[i]["label"] == truth[i]) for i in ids if i in by_id and by_id[i].get("confidence") is not None and by_id[i]["label"] is not None]
    conf, ok = [c for c, _ in scored], [o for _, o in scored]
    cal = expected_calibration_error(conf, ok)
    return {
        "status": "ok",
        "n": len(ids),
        "answered": sum(p is not None for p in y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": per_class_scores(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred),
        "hard": {"n": len(hard_idx), "accuracy": sub(hard_idx)},
        "easy": {"n": len(easy_idx), "accuracy": sub(easy_idx)},
        "calibration": None if cal is None else {
            "n_scored": len(conf), "brier_top": brier_top(conf, ok), "ece": cal["ece"], "bins": cal["bins"],
            "mean_confidence": sum(conf) / len(conf),
        },
        "latency_s": result["latency_s"],
        "calls": result["calls"],
        "usage": result["usage"],
        "notes": result["notes"],
    }


def score(state: BakeoffState) -> dict:
    ids = [h["headline_id"] for h in state["headlines"]]
    hard_ids = set(state["hard_ids"])
    by_name = {r["name"]: r for r in state["results"]}  # branch order is not deterministic; the lookup by name is
    metrics: dict[str, Any] = {}
    labels_by_clf: dict[str, dict[str, str | None]] = {}
    for name in CLASSIFIERS:
        r = by_name.get(name)
        if r is None or r.get("error"):
            metrics[name] = {"status": "failed", "error": (r or {}).get("error", "branch produced no result")}
            continue
        metrics[name] = classifier_metrics(r, ids, state["truth"], hard_ids)
        labels_by_clf[name] = {p["headline_id"]: p["label"] for p in r["predictions"]}
    agree = pairwise_agreement(labels_by_clf)
    unanimous = [i for i in ids if len(labels_by_clf) > 1 and len({m.get(i) for m in labels_by_clf.values()}) == 1 and all(m.get(i) is not None for m in labels_by_clf.values())]
    metrics["agreement"] = {
        "pairwise": agree,
        "unanimous_share": len(unanimous) / len(ids) if labels_by_clf and len(labels_by_clf) > 1 else None,
        "accuracy_when_unanimous": accuracy([state["truth"][i] for i in unanimous], [next(iter(labels_by_clf.values()))[i] for i in unanimous]) if unanimous else None,
    }
    return {"metrics": metrics}


# --------------------------------------------------------------------------- report
def _f(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_table(metrics: dict, sample_info: dict, mode: str) -> str:
    n_hard = sample_info["n_hard"]
    lines = [
        f"Bake-off on {sample_info['n']} headlines (seed {sample_info['seed']}, {n_hard} hard) - mode: {mode}",
        "",
        f"| classifier | accuracy | macro-F1 | hard acc (n={n_hard}) | easy acc | Brier | ECE | latency s | calls | tokens in/out |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in CLASSIFIERS:
        m = metrics[name]
        if m["status"] != "ok":
            lines.append(f"| {name} | failed: {m['error'][:60]} | | | | | | | | |")
            continue
        cal = m["calibration"] or {}
        lines.append(
            f"| {name} | {_f(m['accuracy'])} | {_f(m['macro_f1'])} | {_f(m['hard']['accuracy'])} | {_f(m['easy']['accuracy'])} | "
            f"{_f(cal.get('brier_top'))} | {_f(cal.get('ece'))} | {m['latency_s']:.2f} | {m['calls']} | "
            f"{m['usage'].get('input_tokens', 0)}/{m['usage'].get('output_tokens', 0)} |"
        )
    lines += ["", "Pairwise agreement (same label):"]
    for pair, v in metrics["agreement"]["pairwise"].items():
        lines.append(f"- {pair}: {_f(v)}")
    lines.append(f"- all classifiers agree on {_f(metrics['agreement']['unanimous_share'])} of headlines "
                 f"(accuracy when they do: {_f(metrics['agreement']['accuracy_when_unanimous'])})")
    for name in CLASSIFIERS:
        m = metrics[name]
        if m["status"] != "ok":
            continue
        lines += ["", f"Confusion matrix, {name} (rows = true, columns = predicted):", f"| true \\ pred | {' | '.join([*LABELS, 'none'])} |", "|---|---|---|---|---|"]
        for t, row in m["confusion"].items():
            lines.append(f"| {t} | " + " | ".join(str(row[c]) for c in [*LABELS, "none"]) + " |")
    return "\n".join(lines)


def table_numbers(table: str) -> set[str]:
    """Decimal figures that appear in the main comparison table (what a verdict may quote)."""
    head = table.split("Pairwise agreement")[0]
    return set(re.findall(r"\b\d+\.\d+\b", head))


def quotes_table(verdict: str, table: str, minimum: int = 3) -> bool:
    return len(table_numbers(table) & set(re.findall(r"\b\d+\.\d+\b", verdict))) >= minimum


def _best(metrics: dict, key, higher: bool = True) -> tuple[list[str], float | None]:
    vals = {n: key(metrics[n]) for n in CLASSIFIERS if metrics[n]["status"] == "ok" and key(metrics[n]) is not None}
    if not vals:
        return [], None
    top = max(vals.values()) if higher else min(vals.values())
    return [n for n, v in vals.items() if math.isclose(v, top, abs_tol=1e-9)], top


def deterministic_verdict(metrics: dict, sample_info: dict, mode: str) -> str:
    acc, acc_v = _best(metrics, lambda m: m["accuracy"])
    hard, hard_v = _best(metrics, lambda m: m["hard"]["accuracy"])
    cal, cal_v = _best(metrics, lambda m: (m["calibration"] or {}).get("ece"), higher=False)
    parts = [
        f"On {sample_info['n']} headlines ({sample_info['n_hard']} hard): best accuracy {' / '.join(acc) or 'n/a'} ({_f(acc_v)}); "
        f"best on hard headlines {' / '.join(hard) or 'n/a'} ({_f(hard_v)}); lowest calibration error {' / '.join(cal) or 'n/a'} ({_f(cal_v)})."
    ]
    if sample_info["n_hard"] < 10:
        parts.append(f"Only {sample_info['n_hard']} hard headlines: treat gaps of a few points as noise.")
    if mode == "offline":
        parts.append("Offline stand-ins: these numbers are meaningless.")
    return " ".join(parts)


VERDICT_PROMPT = (
    "You write the verdict of a classifier bake-off for a trading-signal team. You are given the results table. Write at most 110 "
    "words of plain prose. Quote at least three numbers EXACTLY as printed in the table. Say which classifier is best on overall "
    "accuracy, which on the hard headlines, and which has the lowest calibration error (ECE); mention latency or tokens only if "
    "they change the recommendation. State the sample size, and if there are fewer than 10 hard headlines say that small gaps are "
    "noise. If the mode is offline, say the numbers are meaningless. Never invent a number that is not in the table."
)


def report(state: BakeoffState) -> dict:
    mode = state.get("mode", mode_label())
    table = render_table(state["metrics"], state["sample"], mode)
    source = "model"
    try:
        verdict = str(get_llm("fast").invoke([SystemMessage(VERDICT_PROMPT), HumanMessage(table)]).text).strip()
    except Exception:  # noqa: BLE001 - the table is the deliverable; the prose is a convenience
        verdict = ""
    if not verdict or not quotes_table(verdict, table):
        verdict, source = deterministic_verdict(state["metrics"], state["sample"], mode), "template"
    caveat = (
        "OFFLINE run: the chat and Decisions API stand-ins give arbitrary labels, so accuracy and calibration numbers are "
        "meaningless. The pipeline is exercised; run with OPENAI_API_KEY set to answer the question."
        if mode == "offline"
        else "Live run. The headlines come from about two dozen templates and the hard subset is small: read hard-subset gaps of a few points as noise."
    )
    return {"table": table, "verdict": verdict, "verdict_source": source, "caveat": caveat, "mode": mode}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(BakeoffState, input_schema=BakeoffIn, output_schema=BakeoffOut)
    g.add_node("sample", sample)
    for name, fn in (("rules_baseline", rules_baseline), ("chat_structured", chat_structured), ("decisions_api", decisions_api)):
        g.add_node(name, _guard(name, fn), input_schema=ClassifierInput)  # the classifiers see the texts, never the labels
    g.add_node("score", score)
    g.add_node("report", report)
    g.add_edge(START, "sample")
    for name in CLASSIFIERS:
        g.add_edge("sample", name)  # parallel branches
    g.add_edge(list(CLASSIFIERS), "score")  # join: waits for all three
    g.add_edge("score", "report")
    g.add_edge("report", END)
    return g.compile(**compile_kwargs)


def _guard(name: str, fn):
    """Turn a crash in one branch into a ``failed`` result so the other two still get scored and reported."""

    def run(state: ClassifierInput) -> dict:
        started = time.perf_counter()
        try:
            return fn(state)
        except Exception as e:  # noqa: BLE001
            return {"results": [{"name": name, "error": f"{type(e).__name__}: {e}"[:200], "predictions": [],
                                 "latency_s": time.perf_counter() - started, "calls": 0, "usage": {}, "notes": []}]}

    run.__name__ = name
    return run


graph = build_graph()

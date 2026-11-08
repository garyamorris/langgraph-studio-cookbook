"""Thin, typed wrapper around OpenAI's Decisions API (``client.decisions.create``).

The Decisions API answers *many classification / scoring questions about one input in a single call* and
returns calibrated probabilities instead of prose. Three question types exist:

* ``predicate`` - "how likely is this statement true?"      -> ``probability``
* ``choice``    - "which of these options fits best?"       -> ``choice`` + ``confidence`` + ``probabilities``
* ``score``     - "rate against these ordered levels"       -> ``score`` + ``confidence`` + ``probabilities``

A request may carry 1-400 questions (this wrapper transparently chunks larger batches) and any single
answer may come back as a ``refusal`` - ``DecisionResult`` makes that safe to consume.

Build questions with ``predicate()``, ``choice()`` and ``score()``, send them with ``decide()``:

    q = [
        predicate("urgent", "Does the customer describe a safety risk?"),
        choice("team", "Which team should own this?", {"billing": "payments", "tech": "devices"}),
        score("anger", "How angry is the customer?", ["calm", "annoyed", "furious"]),
    ]
    res = decide(ticket_text, q)
    if res.p("urgent") > 0.8: ...
    res.pick("team"), res.conf("team"), res.score("anger"), res.answers["team"].probabilities

Offline mode (``DEEPDIVE_OFFLINE=1`` or no API key) returns deterministic, *arbitrary but stable* answers so
graphs run end to end for free. Tests pin exact answers with ``set_decision_override``.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Literal, Sequence

from langsmith import traceable
from pydantic import BaseModel, Field

from deepdive.common.config import decisions_model, is_offline

MAX_QUESTIONS_PER_REQUEST = 400


# --------------------------------------------------------------------------- question builders
def predicate(name: str, instructions: str) -> dict[str, Any]:
    """A yes/no statement; the answer is the probability that it is true."""
    return {"name": name, "type": "predicate", "instructions": instructions}


def choice(name: str, instructions: str, options: dict[str, str] | Sequence[str]) -> dict[str, Any]:
    """Pick one of 2-255 options. ``options`` is ``{value: description}`` or a plain list of values."""
    opts = options if isinstance(options, dict) else {o: "" for o in options}
    choices = [{"value": v, **({"description": d} if d else {})} for v, d in opts.items()]
    return {"name": name, "type": "choice", "instructions": instructions, "choices": choices}


def score(name: str, instructions: str, levels: dict[str, str] | Sequence[str]) -> dict[str, Any]:
    """Rate against ordered ``levels`` (lowest first). ``levels`` is ``{label: description}`` or a list of labels."""
    lv = levels if isinstance(levels, dict) else {o: "" for o in levels}
    return {
        "name": name,
        "type": "score",
        "instructions": instructions,
        "levels": [{"label": k, **({"description": d} if d else {})} for k, d in lv.items()],
    }


# --------------------------------------------------------------------------- results
class Answer(BaseModel):
    """One answer, normalised across question types."""

    name: str
    type: Literal["predicate", "choice", "score", "refusal"]
    probability: float | None = None  # predicate
    choice: str | None = None  # choice
    score: float | None = None  # score (expected level value)
    confidence: float | None = None  # choice / score
    probabilities: dict[str, float] = Field(default_factory=dict)  # choice value / score label -> probability


class DecisionResult(BaseModel):
    """All answers of one ``decide()`` call, keyed by question name."""

    model: str
    mode: Literal["live", "offline"]
    answers: dict[str, Answer]
    usage: dict[str, Any] = Field(default_factory=dict)

    def p(self, name: str, default: float = 0.0) -> float:
        """Probability of a predicate (``default`` if it was refused / missing)."""
        a = self.answers.get(name)
        return float(a.probability) if a is not None and a.probability is not None else default

    def pick(self, name: str, default: str | None = None) -> str | None:
        """The chosen option of a choice question."""
        a = self.answers.get(name)
        return a.choice if a is not None and a.choice is not None else default

    def conf(self, name: str, default: float = 0.0) -> float:
        a = self.answers.get(name)
        return float(a.confidence) if a is not None and a.confidence is not None else default

    def level(self, name: str, default: str | None = None) -> str | None:
        """The most probable *label* of a score question."""
        a = self.answers.get(name)
        if a is None or not a.probabilities:
            return default
        return max(a.probabilities, key=a.probabilities.get)  # type: ignore[arg-type]

    def score(self, name: str, default: float = 0.0) -> float:
        a = self.answers.get(name)
        return float(a.score) if a is not None and a.score is not None else default

    def refused(self, name: str) -> bool:
        a = self.answers.get(name)
        return a is None or a.type == "refusal"

    def compact(self) -> dict[str, Any]:
        """A small JSON-friendly dict for graph state / Studio (drops None fields, rounds floats)."""
        out: dict[str, Any] = {}
        for k, a in self.answers.items():
            d = a.model_dump(exclude_none=True, exclude={"name"})
            d["probabilities"] = {kk: round(vv, 3) for kk, vv in d.get("probabilities", {}).items()} or None
            out[k] = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items() if vv is not None}
        return out


# --------------------------------------------------------------------------- offline stub
_OVERRIDES: dict[str, tuple[Any, float]] = {}


def set_decision_override(name: str, value: float | str, confidence: float = 0.9) -> None:
    """Pin the offline answer of question ``name``. float -> predicate probability; str -> chosen option / score label."""
    _OVERRIDES[name] = (value, confidence)


def clear_decision_overrides() -> None:
    _OVERRIDES.clear()


def _h01(*parts: Any) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:10], 16) / float(16**10)


def _normalise(weights: list[float]) -> list[float]:
    t = sum(weights) or 1.0
    return [w / t for w in weights]


def _offline_answer(text: str, q: dict[str, Any]) -> Answer:
    name, qtype = q["name"], q["type"]
    override = _OVERRIDES.get(name)
    if qtype == "predicate":
        p = float(override[0]) if override else round(0.1 + 0.8 * _h01(text, name), 4)
        return Answer(name=name, type="predicate", probability=p)

    labels = [c["value"] for c in q["choices"]] if qtype == "choice" else [lv["label"] for lv in q["levels"]]
    labels = [str(x) for x in labels]
    if override and str(override[0]) in labels:
        top, conf = str(override[0]), override[1]
        rest = (1 - conf) / max(len(labels) - 1, 1)
        probs = {lab: (conf if lab == top else rest) for lab in labels}
    else:
        weights = _normalise([(0.15 + _h01(text, name, lab)) ** 2 for lab in labels])
        probs = dict(zip(labels, weights))
        top = max(probs, key=probs.get)  # type: ignore[arg-type]
    probs = {k: round(v, 4) for k, v in probs.items()}
    if qtype == "choice":
        return Answer(name=name, type="choice", choice=top, confidence=probs[top], probabilities=probs)
    expected = sum(i * probs[lab] for i, lab in enumerate(labels))
    return Answer(name=name, type="score", score=round(expected, 4), confidence=probs[top], probabilities=probs)


# --------------------------------------------------------------------------- live call
def _parse_live(resp: Any, questions: list[dict[str, Any]]) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for idx, ans in enumerate(getattr(resp, "answers", [])):
        name = getattr(ans, "name", None) or questions[idx]["name"]
        t = getattr(ans, "type", "refusal")
        if t == "predicate":
            out[name] = Answer(name=name, type="predicate", probability=float(ans.probability))
        elif t == "choice":
            probs = {str(p.value): float(p.probability) for p in ans.probabilities}
            out[name] = Answer(name=name, type="choice", choice=str(ans.choice), confidence=float(ans.confidence), probabilities=probs)
        elif t == "score":
            probs = {str(p.label): float(p.probability) for p in ans.probabilities}
            out[name] = Answer(name=name, type="score", score=float(ans.score), confidence=float(ans.confidence), probabilities=probs)
        else:
            out[name] = Answer(name=name, type="refusal")
    return out


@traceable(run_type="llm", name="openai.decisions.create")
def _live_call(model: str, input: Any, questions: list[dict[str, Any]]) -> dict[str, Any]:
    import openai

    # An explicit timeout matters: the SDK default is 10 minutes (times retries), so one stalled request would hang a
    # whole run for half an hour. Override with DECISIONS_TIMEOUT (seconds).
    client = openai.OpenAI(timeout=float(os.getenv("DECISIONS_TIMEOUT", "60")), max_retries=2)
    resp = client.decisions.create(model=model, input=input, questions=questions)  # type: ignore[arg-type]
    usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
    answers = _parse_live(resp, questions)
    return {"model": resp.model, "answers": {k: v.model_dump() for k, v in answers.items()}, "usage": usage}


def _validate(questions: list[dict[str, Any]]) -> None:
    if not questions:
        raise ValueError("decide() needs at least one question")
    names = [q.get("name") for q in questions]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("every question needs a unique 'name'")
    for q in questions:
        if q["type"] == "choice" and not 2 <= len(q["choices"]) <= 255:
            raise ValueError(f"choice {q['name']!r} needs 2-255 options")


def decide(input: str | list[dict[str, Any]], questions: list[dict[str, Any]], *, model: str | None = None) -> DecisionResult:
    """Ask ``questions`` about ``input`` in as few Decisions API calls as possible."""
    _validate(questions)
    used_model = model or decisions_model()
    text = input if isinstance(input, str) else re.sub(r"\s+", " ", str(input))

    if is_offline():
        answers = {q["name"]: _offline_answer(text, q) for q in questions}
        return DecisionResult(model=f"{used_model} (offline stub)", mode="offline", answers=answers)

    answers_all: dict[str, Answer] = {}
    usage_total: dict[str, int] = {}
    resp_model = used_model
    for i in range(0, len(questions), MAX_QUESTIONS_PER_REQUEST):
        raw = _live_call(used_model, input, questions[i : i + MAX_QUESTIONS_PER_REQUEST])
        resp_model = raw["model"]
        answers_all.update({k: Answer(**v) for k, v in raw["answers"].items()})
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            usage_total[k] = usage_total.get(k, 0) + int(raw["usage"].get(k, 0) or 0)
    return DecisionResult(model=resp_model, mode="live", answers=answers_all, usage=usage_total)

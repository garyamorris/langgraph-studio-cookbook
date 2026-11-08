"""Subject-line tournament: diverse drafts, hard guardrails, one calibrated judge.

PROBLEM
    Email subject lines drive open rates, and a single LLM draft is a coin flip: the model's first idea is rarely its
    best, and asking it to "pick the best of these" returns a confident answer with no way to see why. The team wants
    several genuinely different candidates, a hard filter for the mistakes that hurt deliverability, and a pick they can
    inspect and argue with.

DESIGN
    A best-of-N tournament with the model used where it is good and code used where rules are rules:

      1. **Diversity by construction.** Six copywriter workers run in parallel (``Send``), each with a different angle:
         urgency, curiosity, social proof, benefit-first, question, minimalist. Asking one model for six variants gives
         six near-copies; six distinct instructions give six different ideas.
      2. **Guardrail in code.** Candidates over 60 characters, with an ALL-CAPS word, with a spam-trigger word or
         pattern, or that nearly duplicate an earlier candidate are rejected, with the reason recorded. These are rules
         you can state exactly, so a model never gets a vote on them.
      3. **One judge call for all survivors.** A single Decisions API request puts the brief and every surviving
         candidate in the same context and asks, per candidate, an ordered ``score`` for appeal (poor..excellent) and a
         ``predicate`` "reads as spam or clickbait", plus one ``choice`` over all candidates for "best overall". Judging
         together makes the comparison relative (not six independent absolute ratings), and the answers are calibrated
         probabilities, not prose. Candidates are shown in a fixed pseudo-random order seeded by the brief, because LLM
         judges favour early items; that reduces, not removes, the bias.
      4. **Ranking in code.** ``adjusted = expected_appeal - SPAM_PENALTY * P(spam)``, with expected appeal rescaled to
         0..1. If the top two are within ``TIE_MARGIN`` the ``choice`` probabilities break the tie; otherwise the
         ``choice`` is shown but does not override the scores. Every probability is in the output table.
      5. A chat model writes a two-sentence rationale from the table only (code trims it to two sentences).

    Edge cases are explicit: with fewer than two survivors there is nothing to compare, so no judge call is made
    (one survivor wins by default, zero survivors returns the rejection reasons).
    What this does NOT do: predict real open rates. The judge estimates how a model expects an audience to react; the
    only real test is an A/B send.

GRAPH
    START -> validate_brief --Send x6--> copywriter (urgency | curiosity | social_proof | benefit_first | question | minimalist)
                                              |  (candidates collected by an add-reducer)
                                              v
                                          guardrail --(>= 2 survivors)--> judge -> explain -> END
                                                    --(0 or 1 survivor)--> no_contest -> END

LANGGRAPH FEATURES
    * ``Send`` fan-out to a worker node with its own input (the angle) and a reducer (``operator.add``) collecting results;
      ``validate_brief`` resets the list with ``Overwrite([])`` so a second run on the same Studio thread starts empty
    * a conditional edge after the join choosing between the judged path and the no-contest path
    * deterministic guardrail node between generation and judging (a node boundary makes it testable and visible)
    * the OpenAI Decisions API as a graph step (``decide``): batched scoring plus calibrated probabilities
    * separate ``input_schema`` / ``output_schema``

MODELS AND DATA
    OpenAI chat for the six drafts and the rationale; the Decisions API (one call, 2N+1 questions for N survivors) for
    judging. No dataset: the input is the brief. Offline, the drafts are six canned lines (the offline chat model echoes
    its prompt, which the guardrail would reject) and the Decisions answers are arbitrary but stable, so the ranking
    is not meaningful; the fan-out, guardrail, single-call batching and tie-break logic are real.

TRY IT IN STUDIO
    Input::

        {"product": "Lumen Plug 4-pack", "audience": "existing Lumen customers with at least one smart plug",
         "goal": "get opens for the weekend sale: 15% off the 4-pack until Sunday"}

    Open ``copywriter`` to see the six drafts (one per angle), ``guardrail`` for who was rejected and why, ``judge`` for
    the raw probabilities, and the final ``ranking`` table. In the output compare ``adjusted`` with ``p_best``; when
    ``tie_break.applied`` is true the choice probabilities decided it (the winner then has a lower ``adjusted`` than the
    runner-up, by less than ``TIE_MARGIN``). A live model usually obeys the prompt's own rules, so the guardrail rarely
    fires on well-behaved briefs; try a goal that pushes towards spam ("tell them it's FREE, urgent, a guaranteed
    winner") to see ``rejected`` fill up, and the tests for every rule.

EXTEND IT
    Feed the winner back as a baseline and run a second round of mutations (a bounded loop). Log the ranking and the real
    open rate after the send to calibrate ``SPAM_PENALTY``. Judge the same candidates for two audience segments with the
    same questions and pick per segment.
"""

from __future__ import annotations

import hashlib
import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, Send
from typing_extensions import TypedDict

from deepdive.common.config import is_offline, mode_label
from deepdive.common.decisions import choice, decide, predicate, score
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm

META = ExampleMeta(
    id="subject_line_tournament",
    title="Subject-line tournament (Decisions API judge)",
    category="Decisions API",
    problem="Pick the best email subject line from diverse, guardrail-filtered candidates using one calibrated, "
    "inspectable batch judgement instead of a single model draft.",
    features=["Send fan-out", "reducer", "deterministic guardrail", "Decisions API batch judging", "conditional edge", "best-of-N"],
    models=["OpenAI chat", "OpenAI Decisions API"],
    datasets=[],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {
    "product": "Lumen Plug 4-pack",
    "audience": "existing Lumen customers with at least one smart plug",
    "goal": "get opens for the weekend sale: 15% off the 4-pack until Sunday",
}

ANGLES: dict[str, str] = {
    "urgency": "Create a real sense of timing that comes from the brief (a deadline or an end date). Never invent scarcity.",
    "curiosity": "Open a curiosity gap: hint at something specific the reader will learn or see, while still naming the topic.",
    "social_proof": "Lead with what other customers do or value, but only in terms the brief supports. Invent no statistics.",
    "benefit_first": "Lead with the single biggest benefit to this reader.",
    "question": "Ask one question this reader would genuinely want answered.",
    "minimalist": "Use three to six plain words and no punctuation tricks.",
}
ANGLE_ORDER = list(ANGLES)

MAX_CHARS = 60
SIMILARITY_LIMIT = 0.7  # token Jaccard similarity at which a later candidate counts as a near-duplicate
SPAM_PENALTY = 0.5  # a line judged certain to be spam loses half of the 0..1 appeal scale
TIE_MARGIN = 0.03  # adjusted scores this close are a tie, decided by the "best overall" choice probabilities
APPEAL_LEVELS = {
    "poor": "Would be ignored or deleted: vague, off-putting or irrelevant to this audience and goal",
    "weak": "Acceptable but forgettable; the reader has little reason to open it",
    "fair": "Clear and relevant, but not distinctive",
    "good": "Clear, specific to this audience and goal, and gives a reason to open it",
    "excellent": "Specific, relevant and compelling without hype; a strong candidate to win the A/B test",
}
ALLOWED_ACRONYMS = {"USB", "LED", "GPS", "NFC", "PIN", "FAQ", "USA"}
SPAM_PATTERNS = [
    r"\bfree\b", r"\bwinners?\b", r"\bwon\b", r"\bguarantee[ds]?\b", r"\bact now\b", r"\bclick here\b", r"\bcash\b",
    r"\bno obligation\b", r"\b100\s?%", r"\bmake money\b", r"\burgent(ly)?\b", r"\bbuy now\b", r"\bonce in a lifetime\b",
    r"!{2,}", r"\${2,}",
]


# --------------------------------------------------------------------------- state
class TournamentIn(TypedDict, total=False):
    product: str
    audience: str
    goal: str


class TournamentOut(TypedDict, total=False):
    status: Literal["judged", "single_survivor", "no_survivors"]
    winner: dict | None
    ranking: list[dict]  # one row per survivor, best first, with the calibrated probabilities
    rejected: list[dict]  # candidates the guardrail removed, with reasons
    tie_break: dict
    rationale: str
    decision: dict
    mode: str


class TournamentState(TournamentIn, total=False):
    candidates: Annotated[list[dict], operator.add]  # filled by the six parallel copywriters
    survivors: list[dict]
    rejected: list[dict]
    ranking: list[dict]
    winner: dict | None
    tie_break: dict
    rationale: str
    status: str
    decision: dict
    mode: str


class WriterInput(TypedDict):
    """What each Send hands a copywriter (not the graph state)."""

    product: str
    audience: str
    goal: str
    angle: str


# --------------------------------------------------------------------------- nodes: generate
def validate_brief(state: TournamentState) -> dict:
    cleaned = {k: re.sub(r"\s+", " ", str(state.get(k) or "")).strip() for k in ("product", "audience", "goal")}
    missing = [k for k, v in cleaned.items() if not v]
    if missing:
        raise ValueError("Provide product, audience and goal; missing: " + ", ".join(missing))
    # Studio reuses the thread: the copywriters APPEND to ``candidates`` (operator.add), so start each run from an empty
    # list, or the previous six lines would reach the guardrail, where the near-duplicate rule rejects the new ones.
    # ``decision`` is written only by the judge, so a no-contest run would otherwise show the last judged run's.
    return {**cleaned, "candidates": Overwrite([]), "decision": {}}


def fan_out(state: TournamentState) -> list[Send]:
    """One Send per angle: six parallel copywriters, each with its own input."""
    return [Send("copywriter", {"product": state["product"], "audience": state["audience"], "goal": state["goal"], "angle": a})
            for a in ANGLE_ORDER]


COPY_PROMPT = (
    "You are an email copywriter. Write ONE subject line for the brief you are given, using the angle below. "
    "Rules: at most 55 characters; no emoji; no ALL-CAPS words; no exclamation marks; none of the words free, winner, "
    "guarantee, urgent; do not invent discounts, numbers, deadlines or customer counts that the brief does not state. "
    "Reply with the subject line only - no quotes, no label, no explanation.\n"
    "ANGLE: {angle}\n{instruction}"
)


def clean_line(raw: str) -> str:
    """Models sometimes add a label, quotes or a second line; keep just the subject line."""
    text = raw.strip()
    line = text.splitlines()[0] if text else ""
    line = re.sub(r"^\s*subject(\s*line)?\s*:\s*", "", line, flags=re.I)
    return line.strip().strip("\"'“”‘’").strip()


# Offline stand-ins. The offline chat model echoes the prompt, which the guardrail would (rightly) reject as far too
# long, so an offline run would never reach the judge. Only the model's default "[offline] ..." echo is replaced; a
# scripted reply (tests) or a real model's answer always passes through untouched.
OFFLINE_DRAFTS = {
    "urgency": "Ends Sunday: {product}",
    "curiosity": "What {product} owners do first",
    "social_proof": "Why customers pick {product}",
    "benefit_first": "A simpler home with {product}",
    "question": "Is your home fully plugged in?",
    "minimalist": "{product}, on sale",
}


def copywriter(state: WriterInput) -> dict:
    angle = state["angle"]
    system = COPY_PROMPT.format(angle=angle, instruction=ANGLES[angle])
    brief = f"Product: {state['product']}\nAudience: {state['audience']}\nGoal: {state['goal']}"
    reply = get_llm(temperature=0.9).invoke([SystemMessage(system), HumanMessage(brief)])
    text = str(reply.text)
    if is_offline() and text.startswith("[offline]"):
        text = OFFLINE_DRAFTS[angle].format(product=state["product"])
    return {"candidates": [{"angle": angle, "text": clean_line(text)}]}


# --------------------------------------------------------------------------- nodes: guardrail
def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def check_line(text: str, kept: list[dict]) -> list[dict]:
    """Every rule a subject line must pass, as ``{"code", "detail"}`` reasons (empty list = passes). Pure function."""
    reasons: list[dict] = []
    if not text:
        return [{"code": "empty", "detail": "no text"}]
    if len(text) > MAX_CHARS:
        reasons.append({"code": "too_long", "detail": f"{len(text)} characters (max {MAX_CHARS})"})
    shouting = [w for w in re.findall(r"\b[A-Z]{3,}\b", text) if w not in ALLOWED_ACRONYMS]
    if shouting:
        reasons.append({"code": "all_caps", "detail": ", ".join(shouting)})
    spam = [m.group(0) for p in SPAM_PATTERNS for m in [re.search(p, text, re.I)] if m]
    if spam:
        reasons.append({"code": "spam_trigger", "detail": ", ".join(spam)})
    for k in kept:
        sim = similarity(text, k["text"])
        if sim >= SIMILARITY_LIMIT:
            reasons.append({"code": "near_duplicate", "detail": f"{k['id']} (similarity {sim:.2f})"})
            break
    return reasons


def guardrail(state: TournamentState) -> dict:
    """Join point: order the candidates, give them stable ids, and drop the ones that break a rule."""
    ordered = sorted(state.get("candidates", []), key=lambda c: ANGLE_ORDER.index(c["angle"]))
    survivors: list[dict] = []
    rejected: list[dict] = []
    for i, c in enumerate(ordered, start=1):
        cand = {"id": f"C{i}", "angle": c["angle"], "text": c["text"]}
        reasons = check_line(cand["text"], survivors)
        if reasons:
            rejected.append({**cand, "reasons": reasons})
        else:
            survivors.append(cand)
    return {"survivors": survivors, "rejected": rejected}


def after_guardrail(state: TournamentState) -> Literal["judge", "no_contest"]:
    return "judge" if len(state["survivors"]) >= 2 else "no_contest"


# --------------------------------------------------------------------------- nodes: judge
def present_in_blind_order(survivors: list[dict], brief: str) -> list[dict]:
    """Fixed pseudo-random order seeded by the brief: reproducible, and not the angle order the ids suggest."""
    return sorted(survivors, key=lambda c: hashlib.sha256(f"{brief}|{c['id']}".encode()).hexdigest())


def build_judge_request(state: TournamentState) -> tuple[str, list[dict]]:
    brief = f"Product: {state['product']}\nAudience: {state['audience']}\nGoal: {state['goal']}"
    shown = present_in_blind_order(state["survivors"], brief)
    text = (
        "BRIEF\n" + brief + "\n\nCANDIDATE SUBJECT LINES (listed in arbitrary order)\n"
        + "\n".join(f"{c['id']}: {c['text']}" for c in shown)
    )
    questions: list[dict] = []
    for c in state["survivors"]:
        questions.append(score(
            f"appeal_{c['id']}",
            f"How likely is it that a typical member of the audience in the brief opens an email with subject line {c['id']}? "
            "Judge only that subject line against the brief. Reward specificity to this audience and goal; penalise vagueness, "
            "hype and anything the brief does not support.",
            APPEAL_LEVELS,
        ))
        questions.append(predicate(
            f"spam_{c['id']}",
            f"Subject line {c['id']} reads as spam or clickbait, or promises something the brief does not support.",
        ))
    questions.append(choice(
        "best",
        "Which single subject line is most likely to achieve the goal in the brief with this audience?",
        {c["id"]: c["text"] for c in state["survivors"]},
    ))
    return text, questions


def expected_appeal(probabilities: dict[str, float]) -> float | None:
    """Expected appeal on 0..1 from the probability of each ordered level (poor=0 ... excellent=1).

    Computed here from the level probabilities rather than read from the API's own ``score`` field, so the scale is
    explicit and identical live and offline. ``None`` when the judge gave no probabilities (a refusal).
    """
    total = sum(probabilities.get(level, 0.0) for level in APPEAL_LEVELS)
    if total <= 0:
        return None
    index = sum(i * probabilities.get(level, 0.0) for i, level in enumerate(APPEAL_LEVELS))
    return index / total / (len(APPEAL_LEVELS) - 1)


def rank_candidates(survivors: list[dict], res) -> tuple[list[dict], dict]:
    """Turn the Decisions answers into a ranking and a tie-break record. Pure function of ``survivors`` and ``res``."""
    best_probs = res.answers["best"].probabilities if "best" in res.answers else {}
    rows = []
    for c in survivors:
        appeal_name, spam_name = f"appeal_{c['id']}", f"spam_{c['id']}"
        a = res.answers.get(appeal_name)
        expected = expected_appeal(a.probabilities) if a is not None and a.type == "score" else None
        appeal = expected if expected is not None else 0.0
        spam_p = res.p(spam_name)
        rows.append({
            **c,
            "chars": len(c["text"]),
            "appeal": round(appeal, 3),
            "appeal_level": res.level(appeal_name),
            "appeal_probs": {k: round(v, 3) for k, v in (a.probabilities if a else {}).items()},
            "spam_p": round(spam_p, 3),
            "p_best": round(float(best_probs.get(c["id"], 0.0)), 3),
            "adjusted": round(appeal - SPAM_PENALTY * spam_p, 4),
            "unscored": expected is None,  # the judge declined to score it: it ranks as the lowest appeal
        })
    rows.sort(key=lambda r: (-r["adjusted"], r["id"]))
    first, second = rows[0], rows[1]
    gap = round(first["adjusted"] - second["adjusted"], 4)
    tie = gap <= TIE_MARGIN
    swapped = tie and second["p_best"] > first["p_best"]
    if swapped:
        rows[0], rows[1] = rows[1], rows[0]
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows, {
        "applied": tie,
        "between": [first["id"], second["id"]],
        "gap": gap,
        "margin": TIE_MARGIN,
        "decided_by": ("p_best" if swapped else "adjusted score order") if tie else "score gap",
        "swapped": swapped,
    }


def judge(state: TournamentState) -> dict:
    """ONE Decisions API call for every survivor, then the ranking is computed in code."""
    text, questions = build_judge_request(state)
    res = decide(text, questions)
    ranking, tie = rank_candidates(state["survivors"], res)
    w = ranking[0]
    return {
        "status": "judged",
        "ranking": ranking,
        "tie_break": tie,
        "winner": {"id": w["id"], "angle": w["angle"], "text": w["text"]},
        "decision": {"model": res.model, "mode": res.mode, "questions": len(questions), "usage": res.usage},
        "mode": mode_label(),
    }


EXPLAIN_PROMPT = (
    "Write EXACTLY two sentences for a marketing manager explaining why this subject line won. Use only the numbers and "
    "angles in the table. Mention the winner's expected appeal and its spam probability; if a tie-break was applied, say "
    "that the 'best overall' probability decided it. No hype, no advice beyond the table."
)


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return " ".join(parts[:n])


def explain(state: TournamentState) -> dict:
    rows = state["ranking"][:3]
    table = "\n".join(
        f"#{r['rank']} {r['id']} [{r['angle']}] \"{r['text']}\" appeal={r['appeal']} spam_p={r['spam_p']} "
        f"adjusted={r['adjusted']} p_best={r['p_best']}" for r in rows)
    tie = state["tie_break"]
    note = (f"Tie-break applied between {tie['between'][0]} and {tie['between'][1]} (gap {tie['gap']} <= {tie['margin']}); "
            f"decided by {tie['decided_by']}." if tie["applied"] else f"No tie: the winner leads by {tie['gap']}.")
    reply = get_llm().invoke([SystemMessage(EXPLAIN_PROMPT), HumanMessage(f"TABLE (best first)\n{table}\n\n{note}")])
    return {"rationale": first_sentences(str(reply.text))}


def no_contest(state: TournamentState) -> dict:
    """Fewer than two candidates passed the guardrail: nothing to compare, so no judge call and no model-written text."""
    survivors, rejected = state["survivors"], state["rejected"]
    reasons = "; ".join(f"{r['id']} ({r['angle']}): " + ", ".join(x["code"] for x in r["reasons"]) for r in rejected)
    if survivors:
        w = survivors[0]
        return {
            "status": "single_survivor",
            "winner": {"id": w["id"], "angle": w["angle"], "text": w["text"]},
            "ranking": [{**w, "rank": 1, "chars": len(w["text"])}],
            "tie_break": {"applied": False},
            "rationale": f"Only {w['id']} ({w['angle']}) passed the guardrail, so it wins by default without being judged. Rejected: {reasons}.",
            "mode": mode_label(),
        }
    return {
        "status": "no_survivors",
        "winner": None,
        "ranking": [],
        "tie_break": {"applied": False},
        "rationale": f"No candidate passed the guardrail. Rejected: {reasons}. Rewrite the brief or relax the rules.",
        "mode": mode_label(),
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(TournamentState, input_schema=TournamentIn, output_schema=TournamentOut)
    g.add_node("validate_brief", validate_brief)
    g.add_node("copywriter", copywriter)
    g.add_node("guardrail", guardrail)
    g.add_node("judge", judge)
    g.add_node("explain", explain)
    g.add_node("no_contest", no_contest)
    g.add_edge(START, "validate_brief")
    g.add_conditional_edges("validate_brief", fan_out, ["copywriter"])
    g.add_edge("copywriter", "guardrail")  # all six Sends finish in one step, so the join needs no extra machinery
    g.add_conditional_edges("guardrail", after_guardrail, ["judge", "no_contest"])
    g.add_edge("judge", "explain")
    g.add_edge("explain", END)
    g.add_edge("no_contest", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

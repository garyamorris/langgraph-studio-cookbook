from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.config import data_dir
from deepdive.common.data import read_csv
from deepdive.common.decisions import Answer, DecisionResult, set_decision_override
from deepdive.common.fakes import set_chat_handler
from deepdive.common.testing import check_example, normalise, run_twice
from deepdive.examples import subject_line_tournament as ex

BRIEF = {"product": "Lumen Plug 4-pack", "audience": "existing customers", "goal": "open the weekend sale email"}

GOOD = {
    "urgency": "Weekend sale ends Sunday night",
    "curiosity": "What plug owners do before Sunday",
    "social_proof": "Why customers reorder their plugs",
    "benefit_first": "Cut standby waste with smart plugs",
    "question": "Is every lamp on a schedule yet?",
    "minimalist": "Plugs, 15% off",
}
RATIONALE = "The winner scored highest. Its spam risk is low. A third sentence that must be trimmed."


class Writers:
    """Chat stand-in keyed on the angle named in the prompt; records the system prompt each copywriter received."""

    def __init__(self, lines=None):
        self.lines = {**GOOD, **(lines or {})}
        self.prompts: dict[str, str] = {}
        self.explain_calls = 0
        set_chat_handler(self)

    def __call__(self, messages, tools):
        system = str(messages[0].content)
        m = re.search(r"ANGLE: (\w+)", system)
        if m:
            self.prompts[m.group(1)] = system
            return AIMessage(content=self.lines[m.group(1)])
        self.explain_calls += 1
        return AIMessage(content=RATIONALE)


def pin(appeal: dict[str, str] | None = None, spam: float = 0.1, best: str = "C1"):
    """Pin every Decisions answer. ``appeal`` maps candidate id -> level; the rest are 'poor'."""
    for i in range(1, 7):
        cid = f"C{i}"
        set_decision_override(f"appeal_{cid}", (appeal or {}).get(cid, "poor"))
        set_decision_override(f"spam_{cid}", spam)
    set_decision_override("best", best)


def run(**over):
    return ex.build_graph().invoke({**BRIEF, **over})


# --------------------------------------------------------------------------- generation
def test_contract():
    check_example(ex)


def test_six_workers_run_in_parallel_with_distinct_angles_and_distinct_candidates():
    w = Writers()
    pin(best="C1")
    ran = [(tuple(ns), node) for ns, upd in ex.build_graph().stream({**BRIEF}, stream_mode="updates", subgraphs=True) for node in upd]
    assert sum(1 for _, n in ran if n == "copywriter") == 6
    assert set(w.prompts) == set(ex.ANGLES) and len(w.prompts) == 6
    for angle, prompt in w.prompts.items():  # each worker got its own instruction and nobody else's
        assert ex.ANGLES[angle] in prompt
        assert all(ex.ANGLES[other] not in prompt for other in ex.ANGLES if other != angle)
    out = run()
    shown = {r["angle"]: r["text"] for r in out["ranking"]}
    assert shown == GOOD and len(set(shown.values())) == 6
    assert [r["id"] for r in sorted(out["ranking"], key=lambda r: r["id"])] == [f"C{i}" for i in range(1, 7)]
    assert [r["angle"] for r in sorted(out["ranking"], key=lambda r: r["id"])] == ex.ANGLE_ORDER, "ids follow angle order, not completion order"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"Weekend sale ends Sunday"', "Weekend sale ends Sunday"),
        ("Subject: Weekend sale ends Sunday\nThis line works because...", "Weekend sale ends Sunday"),
        ("  subject line:  “Plugs, 15% off”  ", "Plugs, 15% off"),
        ("", ""),
    ],
)
def test_clean_line_strips_labels_quotes_and_extra_lines(raw, expected):
    assert ex.clean_line(raw) == expected


# --------------------------------------------------------------------------- guardrail
@pytest.mark.parametrize(
    "text,code",
    [
        ("x" * 61, "too_long"),
        ("Plugs on sale NOW", "all_caps"),
        ("Get your plugs for free", "spam_trigger"),
        ("A guaranteed saving", "spam_trigger"),
        ("Act now to save", "spam_trigger"),
        ("Big savings!!", "spam_trigger"),
        ("This is urgent", "spam_trigger"),
        ("", "empty"),
    ],
)
def test_check_line_flags_each_rule(text, code):
    assert [r["code"] for r in ex.check_line(text, [])][0] == code


@pytest.mark.parametrize("text", ["x" * 60, "USB plugs for every room", "Is every lamp on a schedule yet?", "Plugs, 15% off", "Save 100 minutes a week"])
def test_check_line_accepts_clean_lines(text):
    assert ex.check_line(text, []) == []


def test_near_duplicates_are_detected_against_earlier_survivors_only():
    kept = [{"id": "C1", "angle": "urgency", "text": "Weekend sale ends Sunday night"}]
    dup = ex.check_line("Weekend sale ends Sunday", kept)
    assert dup[0]["code"] == "near_duplicate" and dup[0]["detail"].startswith("C1")
    assert ex.check_line("Why customers reorder their plugs", kept) == []
    assert ex.similarity("a b c", "a b c") == 1.0 and ex.similarity("a b", "c d") == 0.0


def test_guardrail_removes_bad_candidates_before_the_judge_sees_them():
    w = Writers({
        "curiosity": "What plug owners do before Sunday and why it matters so much to the whole household now",  # too long
        "social_proof": "FREE plugs for every winner",  # caps + spam
        "question": "Weekend sale ends Sunday",  # near-duplicate of urgency
    })
    pin(best="C4")
    asked = []
    real = ex.decide
    ex.decide = lambda text, qs, **kw: (asked.append((text, qs)), real(text, qs, **kw))[1]
    try:
        out = run()
    finally:
        ex.decide = real
    rejected = {r["id"]: [x["code"] for x in r["reasons"]] for r in out["rejected"]}
    assert rejected == {"C2": ["too_long"], "C3": ["all_caps", "spam_trigger"], "C5": ["near_duplicate"]}
    assert {r["id"] for r in out["ranking"]} == {"C1", "C4", "C6"}
    text, questions = asked[0]
    assert "FREE plugs" not in text and "winner" not in text, "rejected candidates never reach the judge"
    assert len(questions) == 2 * 3 + 1
    assert w.explain_calls == 1


# --------------------------------------------------------------------------- judging
def test_one_decisions_call_judges_all_survivors_together():
    Writers()
    pin(best="C3")
    calls = []
    real = ex.decide
    ex.decide = lambda text, qs, **kw: (calls.append((text, qs)), real(text, qs, **kw))[1]
    try:
        out = run()
    finally:
        ex.decide = real
    assert len(calls) == 1, "exactly one Decisions API request, however many candidates"
    text, qs = calls[0]
    by_type = {t: [q for q in qs if q["type"] == t] for t in ("score", "predicate", "choice")}
    assert (len(by_type["score"]), len(by_type["predicate"]), len(by_type["choice"])) == (6, 6, 1)
    assert [lv["label"] for lv in by_type["score"][0]["levels"]] == list(ex.APPEAL_LEVELS), "levels are ordered lowest first"
    assert {q["name"] for q in qs} == {f"{k}_C{i}" for k in ("appeal", "spam") for i in range(1, 7)} | {"best"}
    assert [c["value"] for c in by_type["choice"][0]["choices"]] == [f"C{i}" for i in range(1, 7)]
    for brief_part in (BRIEF["product"], BRIEF["audience"], BRIEF["goal"], *GOOD.values()):
        assert brief_part in text
    assert out["decision"]["questions"] == 13 and out["decision"]["mode"] == "offline"


def test_candidates_are_shown_in_a_fixed_blind_order_not_in_angle_order():
    Writers()
    pin()
    seen = []
    real = ex.decide
    ex.decide = lambda text, qs, **kw: (seen.append(text), real(text, qs, **kw))[1]
    try:
        run()
        run()
    finally:
        ex.decide = real
    order = lambda t: re.findall(r"^(C\d): ", t, re.M)  # noqa: E731
    assert order(seen[0]) == order(seen[1]), "deterministic for a given brief"
    assert order(seen[0]) != sorted(order(seen[0])) and sorted(order(seen[0])) == [f"C{i}" for i in range(1, 7)]


def test_pinned_decisions_pin_the_winner_and_the_table_shows_the_probabilities():
    Writers()
    pin(appeal={"C4": "excellent", "C1": "good"}, best="C4")
    out = run()
    assert out["status"] == "judged" and out["winner"]["id"] == "C4" and out["winner"]["text"] == GOOD["benefit_first"]
    assert [r["id"] for r in out["ranking"]][:2] == ["C4", "C1"] and [r["rank"] for r in out["ranking"]] == list(range(1, 7))
    top = out["ranking"][0]
    assert top["appeal_level"] == "excellent" and top["appeal"] == pytest.approx(0.9375, abs=1e-3)  # (4*.9+(0+1+2+3)*.025)/4
    assert top["p_best"] == pytest.approx(0.9) and top["spam_p"] == pytest.approx(0.1)
    assert sum(top["appeal_probs"].values()) == pytest.approx(1.0, abs=0.01) and top["appeal_probs"]["excellent"] == pytest.approx(0.9)
    assert top["adjusted"] == pytest.approx(top["appeal"] - ex.SPAM_PENALTY * top["spam_p"], abs=1e-3)
    assert out["tie_break"]["applied"] is False and out["tie_break"]["decided_by"] == "score gap"
    assert out["rationale"] == "The winner scored highest. Its spam risk is low.", "trimmed to exactly two sentences"


def test_spam_probability_can_overturn_a_higher_appeal():
    Writers()
    pin(appeal={"C1": "excellent", "C2": "good"}, best="C1")
    set_decision_override("spam_C1", 0.95)
    set_decision_override("spam_C2", 0.05)
    out = run()
    assert [r["id"] for r in out["ranking"]][:2] == ["C2", "C1"]
    assert out["ranking"][1]["appeal"] > out["ranking"][0]["appeal"], "C1 appeals more but is penalised for reading as spam"
    assert out["winner"]["id"] == "C2"


def test_close_scores_are_decided_by_the_best_overall_probabilities():
    Writers()
    pin(appeal={"C2": "excellent", "C5": "excellent"}, best="C5")  # C2 and C5 tie on score; the judge prefers C5
    out = run()
    assert out["tie_break"] == {"applied": True, "between": ["C2", "C5"], "gap": 0.0, "margin": ex.TIE_MARGIN,
                                "decided_by": "p_best", "swapped": True}
    assert out["winner"]["id"] == "C5" and [r["id"] for r in out["ranking"]][:2] == ["C5", "C2"]
    assert out["ranking"][0]["p_best"] > out["ranking"][1]["p_best"]
    # control: the same tie with the judge preferring C2 keeps C2 first - the choice, not the id order, decides
    pin(appeal={"C2": "excellent", "C5": "excellent"}, best="C2")
    again = run()
    assert again["winner"]["id"] == "C2" and again["tie_break"]["swapped"] is False


def test_a_clear_score_lead_is_not_overridden_by_the_choice_answer():
    Writers()
    pin(appeal={"C3": "excellent", "C5": "good"}, best="C5")  # the choice favours C5, but C3 leads by far more than the margin
    out = run()
    assert out["winner"]["id"] == "C3" and out["tie_break"]["applied"] is False
    assert next(r for r in out["ranking"] if r["id"] == "C5")["p_best"] > next(r for r in out["ranking"] if r["id"] == "C3")["p_best"]


def _level_probs(expected_level: float) -> dict[str, float]:
    """Level probabilities whose expected level index (poor=0 ... excellent=4) is exactly ``expected_level``."""
    levels = list(ex.APPEAL_LEVELS)
    lo = int(expected_level)
    probs = {lv: 0.0 for lv in levels}
    probs[levels[lo]] = 1 - (expected_level - lo)
    if lo + 1 < len(levels):
        probs[levels[lo + 1]] = expected_level - lo
    return probs


def _result(appeal: dict[str, float], spam: dict[str, float], best: dict[str, float]) -> DecisionResult:
    """Handcrafted Decisions answers. NB the API's own ``score`` field is deliberately wrong: it must be ignored."""
    answers = {}
    for cid in appeal:
        answers[f"appeal_{cid}"] = Answer(name=f"appeal_{cid}", type="score", score=99.0, confidence=0.5,
                                          probabilities=_level_probs(appeal[cid]))
        answers[f"spam_{cid}"] = Answer(name=f"spam_{cid}", type="predicate", probability=spam[cid])
    answers["best"] = Answer(name="best", type="choice", choice=max(best, key=best.get), confidence=max(best.values()), probabilities=best)
    return DecisionResult(model="m", mode="offline", answers=answers)


def _surv(*ids):
    return [{"id": i, "angle": "urgency", "text": f"line {i}"} for i in ids]


def test_ranking_formula_and_margin_boundary():
    # expected levels 3.0 and 2.88 -> appeal 0.75 and 0.72 -> adjusted gap 0.03 exactly: still a tie (<=)
    res = _result({"C1": 3.0, "C2": 2.88}, {"C1": 0.0, "C2": 0.0}, {"C1": 0.4, "C2": 0.6})
    rows, tie = ex.rank_candidates(_surv("C1", "C2"), res)
    assert tie["applied"] and tie["swapped"] and rows[0]["id"] == "C2"
    # one hundredth wider: no tie, the higher adjusted score wins whatever the choice says
    res = _result({"C1": 3.0, "C2": 2.84}, {"C1": 0.0, "C2": 0.0}, {"C1": 0.4, "C2": 0.6})
    rows, tie = ex.rank_candidates(_surv("C1", "C2"), res)
    assert not tie["applied"] and rows[0]["id"] == "C1"
    assert rows[0]["adjusted"] == pytest.approx(3.0 / 4 - 0.5 * 0.0)


def test_expected_appeal_comes_from_the_level_probabilities_not_the_api_score_field():
    assert ex.expected_appeal(_level_probs(0.0)) == 0.0 and ex.expected_appeal(_level_probs(4.0)) == 1.0
    assert ex.expected_appeal(_level_probs(3.0)) == pytest.approx(0.75) and ex.expected_appeal(_level_probs(2.5)) == pytest.approx(0.625)
    assert ex.expected_appeal({"poor": 0.3, "excellent": 0.3}) == pytest.approx(0.5), "probabilities are renormalised"
    assert ex.expected_appeal({}) is None
    rows, _ = ex.rank_candidates(_surv("C1", "C2"), _result({"C1": 3.0, "C2": 1.0}, {"C1": 0.0, "C2": 0.0}, {"C1": 0.5, "C2": 0.5}))
    assert rows[0]["appeal"] == 0.75, "score=99 in the answers was ignored"


def test_a_refused_appeal_answer_ranks_last_and_is_flagged():
    res = _result({"C1": 3.0, "C2": 3.2, "C3": 1.0}, {"C1": 0.1, "C2": 0.1, "C3": 0.1}, {"C1": 0.3, "C2": 0.3, "C3": 0.4})
    res.answers["appeal_C2"] = Answer(name="appeal_C2", type="refusal")
    rows, _ = ex.rank_candidates(_surv("C1", "C2", "C3"), res)
    assert [r["id"] for r in rows] == ["C1", "C3", "C2"] and rows[-1]["unscored"] is True and rows[-1]["appeal"] == 0.0


# --------------------------------------------------------------------------- edge cases
def forbid_decide(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the Decisions API must not be called with fewer than two survivors")

    monkeypatch.setattr(ex, "decide", boom)


def test_a_single_survivor_wins_by_default_without_a_judge_call(monkeypatch):
    forbid_decide(monkeypatch)
    bad = {a: GOOD[a].upper() + " FREE" for a in ex.ANGLE_ORDER if a != "benefit_first"}
    w = Writers(bad)
    out = run()
    assert out["status"] == "single_survivor" and out["winner"] == {"id": "C4", "angle": "benefit_first", "text": GOOD["benefit_first"]}
    assert len(out["rejected"]) == 5 and [r["rank"] for r in out["ranking"]] == [1] and out["tie_break"] == {"applied": False}
    assert "wins by default" in out["rationale"] and "C1 (urgency)" in out["rationale"]
    assert w.explain_calls == 0, "no model-written rationale when nothing was judged"


def test_no_survivors_returns_the_reasons_and_no_winner(monkeypatch):
    forbid_decide(monkeypatch)
    Writers({a: "x" * 80 for a in ex.ANGLE_ORDER})
    out = run()
    assert out["status"] == "no_survivors" and out["winner"] is None and out["ranking"] == []
    assert len(out["rejected"]) == 6 and all(r["reasons"][0]["code"] == "too_long" for r in out["rejected"])
    assert out["rationale"].startswith("No candidate passed the guardrail") and "too_long" in out["rationale"]


@pytest.mark.parametrize("missing", ["product", "audience", "goal"])
def test_brief_must_be_complete(missing):
    with pytest.raises(ValueError, match=missing):
        ex.build_graph().invoke({**BRIEF, missing: "   "})
    with pytest.raises(ValueError, match="missing: audience, goal"):
        ex.build_graph().invoke({"product": "Lumen Plug 4-pack"})  # a half-typed brief is a bad value, not a blank form
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        ex.build_graph().invoke({"scenario": "not a case"})


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
def writers_per_product():
    """Chat stand-in whose lines name the product in the brief, so two briefs never produce the same six lines.
    A product called "Spam ..." gets lines that the guardrail rejects, so that brief ends in no_contest."""

    def handler(messages, tools):
        angle = re.search(r"ANGLE: (\w+)", str(messages[0].content))
        if not angle:
            return AIMessage(content=RATIONALE)
        product = re.search(r"Product: (.*)", str(messages[1].content)).group(1)
        if product.startswith("Spam"):
            return AIMessage(content="FREE guaranteed winner, act now")
        return AIMessage(content=f"{GOOD[angle.group(1)]} ({product})")

    set_chat_handler(handler)


SECOND_BRIEF = {"product": "Lumen Therm", "audience": "new homeowners", "goal": "promote the autumn heating offer"}


def test_a_second_brief_on_the_same_thread_is_the_same_as_on_a_fresh_thread():
    writers_per_product()
    pin()
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), BRIEF, SECOND_BRIEF)
    assert normalise(reused.output) == normalise(fresh.output)
    out = reused.output
    assert len(out["ranking"]) + len(out["rejected"]) == 6, "exactly the six new candidates were considered"
    assert all("Lumen Therm" in r["text"] for r in out["ranking"]), "no line from the first brief survived"
    assert not any(x["code"] == "near_duplicate" for r in out["rejected"] for x in r["reasons"])


def test_the_same_brief_twice_on_one_thread_does_not_make_every_candidate_a_near_duplicate():
    Writers()
    pin()
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    first, again = g.invoke(BRIEF, cfg), g.invoke(BRIEF, cfg)
    assert first["rejected"] == [] and again["rejected"] == []
    assert normalise(again) == normalise(first)


def test_a_no_contest_run_after_a_judged_run_does_not_show_the_old_judgement():
    writers_per_product()
    pin()
    spam_brief = {**SECOND_BRIEF, "product": "Spam Plug"}
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), BRIEF, spam_brief)
    assert reused.output["status"] == "no_survivors"
    assert "decision" not in reused.output or reused.output["decision"] == {}, "the judge never ran for this brief"
    assert normalise(reused.output) == normalise(fresh.output)


# --------------------------------------------------------------------------- ready-made cases (Studio's dropdown)
def spy_on_decide(monkeypatch):
    asked: list[str] = []
    real = ex.decide
    monkeypatch.setattr(ex, "decide", lambda text, qs, **kw: (asked.append(text), real(text, qs, **kw))[1])
    return asked


def test_a_blank_form_runs_the_first_ready_made_brief(monkeypatch):
    Writers()
    pin()
    asked = spy_on_decide(monkeypatch)
    out = ex.build_graph().invoke({})  # Studio's Submit on an empty form: no dead end
    first = next(iter(ex.CASES.values()))
    assert out["status"] == "judged" and first == ex.EXAMPLE_INPUT, "the first case is the standard brief"
    assert all(first[k] in asked[0] for k in ("product", "audience", "goal")), "the judge saw the first case's brief"


def test_every_ready_made_brief_is_complete_and_rests_on_real_data():
    sales = read_csv("sales.csv")
    prices = {p: g.sort_values("order_date").iloc[-1]["unit_price"] for p, g in sales.groupby("product")}
    firmware = " ".join((data_dir() / "kb" / "support" / "firmware-updates.md").read_text().split())
    for label, c in ex.CASES.items():
        assert set(c) == {"product", "audience", "goal"} and all(c[k].strip() for k in c), f"{label}: a complete triple"
        assert c["product"] in prices and label.startswith(c["product"]), f"{label}: lead with a product that is in the data"
        for dollars in re.findall(r"\$(\d+)", label):  # a price in a label is a price in the data (the Therm's old one too)
            known = set(sales[sales["product"] == c["product"]]["unit_price"])
            assert float(dollars) in known, f"{label}: ${dollars} is not a {c['product']} price"
        if "3.0.2" in label:
            assert "Firmware 3.0.2 fixes" in firmware and "pairing timeout (E-317) on Cam Pro" in firmware
    therm = next(c for c in ex.CASES.values() if c["product"] == "Lumen Therm")
    assert "$149, down from $159" in therm["goal"] and prices["Lumen Therm"] == 149.0
    assert len({c["audience"] for c in ex.CASES.values()}) == len(ex.CASES), "each case has its own audience"


def test_the_spam_pushing_case_asks_for_exactly_what_the_guardrail_rejects():
    """The label only says 'watch for rejected'; what is certain is that the words its goal asks for are rejected."""
    label = next(k for k in ex.CASES if "spam" in k)
    goal = ex.CASES[label]["goal"]
    for phrase in ("FREE", "urgent", "guaranteed winner"):
        assert phrase in goal
        assert any(r["code"] in ("spam_trigger", "all_caps") for r in ex.check_line(f"Open this: {phrase}", [])), phrase
    Writers({a: "FREE gift, urgent, a guaranteed winner" for a in ex.ANGLE_ORDER})  # writers that obey the goal
    out = ex.build_graph().invoke({"scenario": label})
    assert out["status"] in ("single_survivor", "no_survivors") and len(out["rejected"]) >= 5
    assert all(r["reasons"][0]["code"] in ("spam_trigger", "all_caps", "near_duplicate") for r in out["rejected"])


def test_a_second_ready_made_brief_on_the_same_thread_is_the_same_as_on_a_fresh_thread():
    writers_per_product()
    pin()
    labels = list(ex.CASES)
    reused, fresh = run_twice(
        lambda: ex.build_graph(checkpointer=InMemorySaver()), {"scenario": labels[0]}, {"scenario": labels[1]}
    )
    assert normalise(reused.output) == normalise(fresh.output)
    assert all("Lumen Therm" in r["text"] for r in reused.output["ranking"]), "nothing from the first brief survived"

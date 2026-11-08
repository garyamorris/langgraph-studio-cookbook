from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import draft_critic_loop as ex

FACTS = ["$149", "24-month warranty", "30-day returns"]
GOOD = "Meet the Lumen Therm at $149. It learns your schedule and keeps rooms comfortable. Includes a 24-month warranty and 30-day returns."
BAD = (
    "The best thermostat ever, guaranteed to revolutionize your home! " + "It is simply wonderful and everyone loves it so much. " * 3
)  # banned claims, no facts, and long enough to break a small limit


def draft_source(*drafts):
    """Chat handler that returns the k-th draft on the k-th writer call, and records every prompt it was given."""
    calls: list[str] = []
    items = list(drafts)

    def handler(messages, tools):
        calls.append(messages[-1].content)
        return AIMessage(items[min(len(calls) - 1, len(items) - 1)])

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def critiques(*scores_and_notes):
    return scripted([ex.Critique(score=s, feedback=list(n)) for s, n in scores_and_notes])


def run(extra=None):
    return ex.graph.invoke({"product": "Lumen Therm", "required_facts": FACTS, **(extra or {})})


def test_contract_and_offline_run():
    out = check_example(ex).output
    assert out["status"] in ("approved", "best_effort") and out["iterations"] <= ex.DEFAULT_MAX_ITERATIONS


# --------------------------------------------------------------------------- the deterministic checker
def rules(draft, facts=FACTS, max_words=120):
    return [v["rule"] for v in ex.check_draft(draft, facts, max_words)]


def test_checker_accepts_a_compliant_draft():
    assert ex.check_draft(GOOD, FACTS, 120) == []
    assert ex.word_count(GOOD) <= 120


def test_checker_counts_words_exactly_at_the_limit():
    ten = " ".join(["word"] * 8) + " $149 returns"
    assert ex.word_count(ten) == 10
    assert rules(ten, ["$149"], max_words=10) == []
    assert rules(ten, ["$149"], max_words=9) == ["length"]
    message = ex.check_draft(ten, ["$149"], 9)[0]["message"]
    assert "10 words" in message and "limit is 9" in message and "at least 1" in message


def test_checker_requires_each_fact_verbatim_but_ignores_case():
    assert rules("price $149, 24-MONTH WARRANTY and 30-day returns") == []
    assert rules("Costs $149 with a 2-year warranty and 30-day returns") == ["missing_fact"]
    assert rules("Nothing useful here at all") == ["missing_fact"] * 3
    assert "24-month warranty" in ex.check_draft("Costs $149 and 30-day returns", FACTS, 120)[0]["message"]


@pytest.mark.parametrize(
    "text,label",
    [
        ("Simply the best choice", "best"),
        ("A guaranteed fit", "guaranteed"),
        ("Guarantee included", "guaranteed"),
        ("The #1 thermostat", "#1 / number one"),
        ("Number one in comfort", "#1 / number one"),
        ("A revolutionary design", "revolutionary"),
        ("It will revolutionize heating", "revolutionary"),
        ("An unbeatable price", "unbeatable"),
        ("The world's first", "world's first"),
        ("A risk-free trial", "risk-free"),
    ],
)
def test_checker_flags_banned_claims(text, label):
    found = [v for v in ex.check_draft(text + " $149 24-month warranty 30-day returns", FACTS, 120) if v["rule"] == "banned_claim"]
    assert len(found) == 1 and f"({label})" in found[0]["message"]


@pytest.mark.parametrize("text", ["Bestow comfort on every room", "Our bestseller returns", "Rated 1 in a small survey", "Hash #10 is fine"])
def test_checker_does_not_flag_look_alikes(text):
    assert not [v for v in ex.check_draft(text, [], 120) if v["rule"] == "banned_claim"]


def test_checker_reports_every_problem_at_once_and_flags_empty_drafts():
    assert sorted(set(rules(BAD, max_words=30))) == ["banned_claim", "length", "missing_fact"]
    assert rules("   ") == ["empty"]


def test_best_attempt_ranking_prefers_fewer_violations_then_score_then_later():
    mk = lambda n, v, s: {"attempt": n, "violations": [{}] * v, "score": s}  # noqa: E731
    assert ex.pick_best([mk(1, 0, 6), mk(2, 1, 10)])["attempt"] == 1, "a rule violation outweighs a high score"
    assert ex.pick_best([mk(1, 0, 6), mk(2, 0, 8), mk(3, 0, 5)])["attempt"] == 2
    assert ex.pick_best([mk(1, 0, 6), mk(2, 0, 6)])["attempt"] == 2, "ties go to the later attempt"


# --------------------------------------------------------------------------- the loop
def test_violating_first_draft_is_revised_once_and_then_approved():
    source = draft_source(BAD, GOOD)
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((5, ["Open with the benefit, not a slogan."]), (8, [])))
    out = run()
    assert out["status"] == "approved" and out["iterations"] == 2
    assert out["draft"] == GOOD and out["unmet"] == []
    first, second = out["history"]
    assert [a["attempt"] for a in out["history"]] == [1, 2] and not first["passed"] and second["passed"]
    assert {v["rule"] for v in first["violations"]} == {"banned_claim", "missing_fact"} and second["violations"] == []
    # the revision prompt carries the exact failures, the editor's note and the draft to fix
    revision = source.calls[1]
    assert "PROBLEMS TO FIX" in revision and 'Missing required fact: "$149"' in revision and "Banned claim" in revision
    assert "Open with the benefit" in revision and BAD[:30] in revision
    assert len(source.calls) == 2, "no third draft once the second passes"


def test_first_draft_that_passes_costs_one_iteration():
    set_chat_handler(draft_source(GOOD))
    set_structured_handler(ex.Critique, critiques((9, [])))
    out = run()
    assert (out["status"], out["iterations"], len(out["history"])) == ("approved", 1, 1)


def test_rules_pass_but_editor_unhappy_still_revises():
    source = draft_source(GOOD, GOOD + " Set it once and forget it.")
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((6, ["Add a concrete benefit."]), (8, [])))
    out = run()
    assert out["iterations"] == 2 and out["status"] == "approved"
    assert "Add a concrete benefit." in source.calls[1]


def test_unsatisfiable_editor_stops_at_the_cap_and_returns_the_best_attempt_not_the_last():
    drafts = [GOOD + " One.", GOOD + " Two.", GOOD + " Three."]
    source = draft_source(*drafts)
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((5, ["Be livelier."]), (6, ["Shorter sentences."]), (4, ["Warmer."])))
    out = run()
    assert len(source.calls) == 3 == out["iterations"], "exactly max_iterations drafts, then stop"
    assert out["status"] == "best_effort"
    assert out["draft"] == drafts[1], "attempt 2 scored highest (6); the last attempt (score 4) must not be returned"
    assert any("6/10" in u and "below the bar" in u for u in out["unmet"]) and "Editor: Shorter sentences." in out["unmet"]
    assert [a["score"] for a in out["history"]] == [5, 6, 4], "history keeps every attempt, in order"
    assert [a["draft"] for a in out["history"]] == drafts


def test_rule_violations_outrank_editor_score_when_choosing_the_best_attempt():
    drafts = [BAD, GOOD.replace("30-day returns", "returns"), BAD]  # attempt 2 has the fewest violations (1)
    set_chat_handler(draft_source(*drafts))
    set_structured_handler(ex.Critique, critiques((9, []), (3, ["Weak."]), (9, [])))
    out = run()
    assert out["status"] == "best_effort" and out["draft"] == drafts[1]
    assert out["unmet"][0].startswith('Missing required fact: "30-day returns"')


@pytest.mark.parametrize("cap", [1, 2, 5])
def test_max_iterations_is_an_input_and_is_honoured(cap):
    source = draft_source(BAD)
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((2, ["Everything."])))
    out = run({"max_iterations": cap})
    assert len(source.calls) == cap == out["iterations"] == len(out["history"])


def test_word_limit_is_an_input():
    set_chat_handler(draft_source(GOOD))
    set_structured_handler(ex.Critique, critiques((9, [])))
    out = run({"max_words": 12, "max_iterations": 2})
    assert out["status"] == "best_effort" and any("Too long" in u and "limit is 12" in u for u in out["unmet"])


def test_both_critics_see_the_same_draft_and_run_for_every_attempt():
    seen = []

    def spy(prompt: str):
        seen.append(prompt)
        return ex.Critique(score=5, feedback=["x"])

    set_chat_handler(draft_source("First draft $149 24-month warranty 30-day returns.", GOOD))
    set_structured_handler(ex.Critique, spy)
    out = run({"max_iterations": 2})
    assert len(seen) == 2 and "First draft" in seen[0] and GOOD in seen[1]
    assert [a["draft"] for a in out["history"]] == ["First draft $149 24-month warranty 30-day returns.", GOOD]


def test_editor_scores_are_clamped_to_the_valid_range():
    set_chat_handler(draft_source(GOOD))
    set_structured_handler(ex.Critique, critiques((99, [])))
    assert run()["history"][0]["score"] == 10
    set_structured_handler(ex.Critique, critiques((-4, ["x"])))
    assert run({"max_iterations": 1})["history"][0]["score"] == 1


def test_history_does_not_carry_over_between_runs_on_the_same_thread():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    set_chat_handler(draft_source(GOOD))
    set_structured_handler(ex.Critique, critiques((9, [])))
    first = g.invoke({"product": "Lumen Therm", "required_facts": FACTS}, cfg)
    second = g.invoke({"product": "Lumen Therm", "required_facts": FACTS}, cfg)
    assert len(first["history"]) == len(second["history"]) == 1 and second["iterations"] == 1


def test_inputs_are_validated():
    for bad, msg in (
        ({}, "Provide a product"),
        ({"product": "Lumen Therm", "max_iterations": 0}, "max_iterations must be"),
        ({"product": "Lumen Therm", "max_iterations": 99}, "max_iterations must be"),
        ({"product": "Lumen Therm", "max_words": 3}, "max_words must be"),
    ):
        with pytest.raises(ValueError, match=msg):
            ex.graph.invoke(bad)


def test_contradictory_rules_can_never_pass_and_end_best_effort_at_the_cap():
    """The Studio demo from the docstring: a required phrase that is itself a banned claim."""
    drafts = ["Meet the Lumen Therm at $149, best-in-class comfort.", "Lumen Therm: $149 and best-in-class.", "Best-in-class Lumen Therm, $149."]
    source = draft_source(*drafts)
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((9, [])))
    out = ex.graph.invoke({"product": "Lumen Therm", "required_facts": ["$149", "best-in-class"]})
    assert out["status"] == "best_effort" and out["iterations"] == 3 == len(source.calls)
    assert all(a["violations"] and a["violations"][0]["rule"] == "banned_claim" for a in out["history"])
    assert any(u.startswith("Banned claim") and "(best)" in u for u in out["unmet"])


def test_writer_prompt_lists_the_banned_claims_from_the_checker():
    source = draft_source(GOOD)
    set_chat_handler(source)
    set_structured_handler(ex.Critique, critiques((9, [])))
    run()
    system = ex.WRITER_PROMPT.format(max_words=120, banned=", ".join(ex.BANNED_CLAIMS))
    assert "revolutionary" in system and "#1 / number one" in system and "at most 120 words" in system

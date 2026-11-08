from __future__ import annotations

import re
import threading
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.decisions import set_decision_override
from deepdive.common.fakes import set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import classifier_bakeoff as ex

NEXT = {"positive": "negative", "negative": "neutral", "neutral": "positive"}  # a deterministic wrong answer
PREV = {v: k for k, v in NEXT.items()}  # a different deterministic wrong answer


def drawn(n=30, seed=0):
    """The sample the graph will use (the `sample` node is a plain function)."""
    return ex.sample({"sample_size": n, "seed": seed})


def pin_chat(truth, correct, *, confidence=0.9, wrong=NEXT, drop=()):
    """Scripted chat labeller: right on `correct` ids, wrong (via `wrong`) elsewhere, silent on `drop`."""

    def handler(text: str) -> ex.BatchLabels:
        ids = [i for i in re.findall(r"H-\d{4}", text) if i not in drop]
        return ex.BatchLabels(
            items=[ex.LabelItem(headline_id=i, label=truth[i] if i in correct else wrong[truth[i]], confidence=confidence) for i in ids]
        )

    set_structured_handler(ex.BatchLabels, handler)


def pin_decisions(truth, correct, *, confidence=0.8, wrong=PREV):
    for i, t in truth.items():
        set_decision_override(ex.question_name(i), t if i in correct else wrong[t], confidence)


# --------------------------------------------------------------------------- contract
def test_meets_the_example_contract():
    check_example(ex)


# --------------------------------------------------------------------------- metrics against hand-computed examples
Y_TRUE = ["positive", "positive", "negative", "neutral", "neutral", "negative"]
Y_PRED = ["positive", "negative", "negative", "neutral", "positive", None]  # the last one was not answered


def test_accuracy_counts_a_missing_answer_as_wrong():
    assert ex.accuracy(Y_TRUE, Y_PRED) == 3 / 6
    assert ex.accuracy([], []) is None


def test_confusion_matrix_has_a_none_column_for_unanswered_items():
    m = ex.confusion_matrix(Y_TRUE, Y_PRED)
    assert m == {
        "positive": {"positive": 1, "negative": 1, "neutral": 0, "none": 0},
        "negative": {"positive": 0, "negative": 1, "neutral": 0, "none": 1},
        "neutral": {"positive": 1, "negative": 0, "neutral": 1, "none": 0},
    }


def test_macro_f1_matches_a_hand_calculation():
    # positive: P=1/2 R=1/2 F1=.5 | negative: P=1/2 R=1/2 F1=.5 | neutral: P=1/1 R=1/2 F1=2/3
    s = ex.per_class_scores(Y_TRUE, Y_PRED)
    assert s["positive"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert s["neutral"]["precision"] == 1.0 and s["neutral"]["recall"] == 0.5 and s["neutral"]["f1"] == pytest.approx(2 / 3)
    assert ex.macro_f1(Y_TRUE, Y_PRED) == pytest.approx((0.5 + 0.5 + 2 / 3) / 3)


def test_macro_f1_edges_perfect_and_a_class_never_predicted():
    assert ex.macro_f1(["positive", "negative", "neutral"], ["positive", "negative", "neutral"]) == 1.0
    # everything predicted "positive": positive F1 = 2*(1/3)*1/(4/3) = .5, the other two classes contribute 0
    assert ex.macro_f1(["positive", "negative", "neutral"], ["positive"] * 3) == pytest.approx(0.5 / 3)


def test_brier_top_label_hand_example():
    # ((0.9-1)^2 + (0.8-0)^2 + (0.6-1)^2) / 3 = (0.01 + 0.64 + 0.16) / 3
    assert ex.brier_top([0.9, 0.8, 0.6], [True, False, True]) == pytest.approx(0.27)
    assert ex.brier_top([1.0, 0.0], [True, False]) == 0.0, "perfectly confident and right"
    assert ex.brier_top([1.0], [False]) == 1.0, "perfectly confident and wrong"
    assert ex.brier_top([], []) is None


def test_ece_hand_example_with_five_bins():
    conf, ok = [0.9, 0.95, 0.55, 0.6], [True, False, True, False]
    # bin 0.8-1.0: n=2 acc .5 conf .925 -> |.425| * 2/4 = .2125 ; bin 0.4-0.6: n=1 acc 1 conf .55 -> .45 * 1/4 = .1125
    # bin 0.6-0.8: n=1 acc 0 conf .6 -> .6 * 1/4 = .15   => ECE = .475
    r = ex.expected_calibration_error(conf, ok)
    assert r["ece"] == pytest.approx(0.475)
    assert [(b["bin"], b["n"]) for b in r["bins"]] == [("0.0-0.2", 0), ("0.2-0.4", 0), ("0.4-0.6", 1), ("0.6-0.8", 1), ("0.8-1.0", 2)]


def test_ece_edges_confidence_one_goes_in_the_top_bin_and_a_calibrated_method_scores_zero():
    assert ex.expected_calibration_error([1.0, 1.0], [True, True])["ece"] == 0.0
    assert ex.expected_calibration_error([0.5, 0.5], [True, False])["ece"] == 0.0
    assert ex.expected_calibration_error([0.9] * 10, [True] * 9 + [False])["ece"] == pytest.approx(0.0)
    assert ex.expected_calibration_error([], []) is None


def test_pairwise_agreement_ignores_items_either_method_did_not_answer():
    preds = {
        "a": {"1": "positive", "2": "negative", "3": "neutral", "4": None},
        "b": {"1": "positive", "2": "neutral", "3": "neutral", "4": "negative"},
    }
    assert ex.pairwise_agreement(preds) == {"a vs b": pytest.approx(2 / 3)}


# --------------------------------------------------------------------------- the hard subset
def test_hard_headlines_are_exactly_the_mixed_message_templates():
    for text in [
        "Acme Robotics beats revenue estimates but slashes full-year outlook",
        "Cobalt Semiconductor shares fall despite record bookings as one-off charge weighs",
        "Delphi Health misses on EPS yet announces $40M buyback and strong order backlog",
        "Granite Bank lifts guidance only after accounting restatement",
        "Fathom Cloud cuts jobs as part of efficiency plan; margins to improve",
        "Delphi Health faces probe, but analysts say exposure is immaterial",
    ]:
        assert ex.is_hard(text), text
    for text in ["Acme Robotics misses earnings estimates and cuts guidance", "Analysts upgrade Cobalt Semiconductor citing margin expansion",
                 "Helio Solar schedules earnings call for the 14th"]:
        assert not ex.is_hard(text), text
    df = read_csv("news_headlines.csv")
    flagged = df[df.headline.map(ex.is_hard)]
    assert len(flagged) == 32 and 0.15 <= len(flagged) / len(df) <= 0.20, "about 15% of the file, as documented"
    assert flagged.label.value_counts().to_dict() == {"neutral": 18, "negative": 11, "positive": 3}


def test_the_sample_records_which_headlines_are_hard_and_hides_labels_from_the_headline_list():
    s = drawn(60)
    hard = {h["headline_id"] for h in s["headlines"] if ex.is_hard(h["headline"])}
    assert set(s["hard_ids"]) == hard and s["sample"]["n_hard"] == len(hard)
    assert all(set(h) == {"headline_id", "headline"} for h in s["headlines"])


# --------------------------------------------------------------------------- sampling
def test_sample_size_is_capped_and_floored():
    assert drawn(10_000)["sample"]["n"] == ex.MAX_SAMPLE == 90
    assert drawn(0)["sample"]["n"] == ex.MIN_SAMPLE
    assert ex.sample({})["sample"]["n"] == 30, "default"


def test_sample_is_deterministic_for_a_seed_and_differs_across_seeds():
    ids = lambda s: [h["headline_id"] for h in s["headlines"]]  # noqa: E731
    assert ids(drawn(30, 5)) == ids(drawn(30, 5))
    assert ids(drawn(30, 5)) != ids(drawn(30, 6))
    assert ids(drawn(30, 5)) == sorted(ids(drawn(30, 5))) and len(set(ids(drawn(90)))) == 90


@pytest.mark.parametrize("n", [6, 30, 31, 90])
def test_sample_is_stratified_by_label_and_hardness(n):
    s = drawn(n)
    assert s["sample"]["n"] == n
    for lab, total in {"positive": 59, "negative": 66, "neutral": 55}.items():
        assert abs(s["sample"]["label_counts"][lab] - n * total / 180) < 1.5
    assert abs(s["sample"]["n_hard"] - n * 32 / 180) < 1.5


# --------------------------------------------------------------------------- the rules baseline
def test_rules_baseline_is_perfect_on_plain_headlines_and_blind_on_mixed_ones():
    assert ex.rules_label("Acme Robotics misses earnings estimates and cuts guidance") == "negative"
    assert ex.rules_label("Analysts upgrade Cobalt Semiconductor citing margin expansion") == "positive"
    assert ex.rules_label("Helio Solar schedules earnings call for the 14th") == "neutral"
    assert ex.rules_label("Acme Robotics beats revenue estimates but slashes full-year outlook") == "neutral", "true label: negative"
    df = read_csv("news_headlines.csv")
    hard = df.headline.map(ex.is_hard)
    ok = df.headline.map(ex.rules_label) == df.label
    assert ok[~hard].mean() == 1.0 and ok[hard].mean() == 0.0 and ok.mean() == pytest.approx(148 / 180)


# --------------------------------------------------------------------------- the whole pipeline with known answers
def test_scripted_classifiers_produce_exactly_the_expected_scores():
    s = drawn(30)
    ids, truth, hard = [h["headline_id"] for h in s["headlines"]], s["truth"], set(s["hard_ids"])
    chat_ok, dec_ok = set(ids[:24]), set(ids[:21])  # chat right on 24/30, Decisions right on 21/30
    pin_chat(truth, chat_ok, confidence=0.9)
    pin_decisions(truth, dec_ok, confidence=0.8)

    out = ex.build_graph().invoke({"sample_size": 30, "seed": 0})
    m = out["metrics"]

    assert m["chat_structured"]["accuracy"] == pytest.approx(24 / 30)
    assert m["decisions_api"]["accuracy"] == pytest.approx(21 / 30)
    n_easy_ok = len(set(ids) - hard)
    assert m["rules_baseline"]["accuracy"] == pytest.approx(n_easy_ok / 30), "rules: every plain headline right, every mixed one wrong"
    assert m["rules_baseline"]["hard"]["accuracy"] == 0.0 and m["rules_baseline"]["easy"]["accuracy"] == 1.0

    # hard-subset accuracy of each LLM, computed from which ids were scripted right
    assert m["chat_structured"]["hard"] == {"n": len(hard), "accuracy": pytest.approx(len(hard & chat_ok) / len(hard))}
    assert m["decisions_api"]["hard"]["accuracy"] == pytest.approx(len(hard & dec_ok) / len(hard))

    # calibration: constant confidence c with accuracy a gives Brier = a(1-c)^2 + (1-a)c^2 and ECE = |a - c|
    a, c = 24 / 30, 0.9
    assert m["chat_structured"]["calibration"]["brier_top"] == pytest.approx(a * (1 - c) ** 2 + (1 - a) * c**2) == pytest.approx(0.17)
    assert m["chat_structured"]["calibration"]["ece"] == pytest.approx(0.1)
    a, c = 21 / 30, 0.8
    assert m["decisions_api"]["calibration"]["brier_top"] == pytest.approx(0.22)
    assert m["decisions_api"]["calibration"]["ece"] == pytest.approx(0.1)
    assert m["rules_baseline"]["calibration"] is None, "a lexicon has no confidence, so no calibration"

    # chat and Decisions are both right on the first 21 ids, disagree on the next 3, and use different wrong labels after that
    assert m["agreement"]["pairwise"]["chat_structured vs decisions_api"] == pytest.approx(21 / 30)

    # batching and request shape
    assert m["chat_structured"]["calls"] == 3 and m["decisions_api"]["calls"] == 1 and m["rules_baseline"]["calls"] == 0
    assert all(m[n]["latency_s"] >= 0 for n in ex.CLASSIFIERS)
    assert out["mode"] == "offline" and "meaningless" in out["caveat"]


def test_the_table_shows_every_classifier_and_the_numbers_the_scores_computed():
    s = drawn(30)
    ids, truth = [h["headline_id"] for h in s["headlines"]], s["truth"]
    pin_chat(truth, set(ids[:24]))
    pin_decisions(truth, set(ids[:21]))
    out = ex.build_graph().invoke({"sample_size": 30})
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in out["table"].splitlines() if r.startswith("| ") and "classifier" not in r and "true" not in r}
    assert rows["chat_structured"][0] == "0.800" and rows["decisions_api"][0] == "0.700"
    assert rows["rules_baseline"][4] == "n/a" and rows["rules_baseline"][5] == "n/a", "no Brier / ECE for the lexicon"
    assert "Confusion matrix, decisions_api" in out["table"] and "Pairwise agreement" in out["table"]


# --------------------------------------------------------------------------- request shapes (batching, one Decisions request)
def test_chat_calls_are_batched_ten_headlines_at_a_time():
    seen: list[list[str]] = []

    def handler(text: str) -> ex.BatchLabels:
        ids = re.findall(r"H-\d{4}", text)
        seen.append(ids)
        return ex.BatchLabels(items=[ex.LabelItem(headline_id=i, label="neutral", confidence=0.5) for i in ids])

    set_structured_handler(ex.BatchLabels, handler)
    ex.build_graph().invoke({"sample_size": 90})
    assert sorted(len(b) for b in seen) == [10] * 9
    assert len({i for b in seen for i in b}) == 90, "every headline appears in exactly one batch"


def test_decisions_api_gets_one_request_with_one_choice_question_per_headline(monkeypatch):
    calls = []
    real = ex.decide

    def spy(document, questions, **kw):
        calls.append((document, questions))
        return real(document, questions, **kw)

    monkeypatch.setattr(ex, "decide", spy)
    out = ex.build_graph().invoke({"sample_size": 90})
    assert len(calls) == 1, "ONE request however many headlines"
    document, questions = calls[0]
    assert len(questions) == 90 <= 400
    assert all(q["type"] == "choice" and [c["value"] for c in q["choices"]] == ["positive", "negative", "neutral"] for q in questions)
    assert len({q["name"] for q in questions}) == 90
    assert all(h in document for h in re.findall(r"H-\d{4}", questions[0]["instructions"]))
    assert out["metrics"]["decisions_api"]["calls"] == 1


def test_confidence_for_the_decisions_api_is_the_probability_of_the_chosen_option():
    s = drawn(30)
    ids = [h["headline_id"] for h in s["headlines"]]
    pin_decisions(s["truth"], set(ids), confidence=0.7)
    res = ex.decisions_api({"headlines": s["headlines"]})["results"][0]
    assert {p["confidence"] for p in res["predictions"]} == {0.7}
    assert all(p["label"] == s["truth"][p["headline_id"]] for p in res["predictions"])
    assert set(res["predictions"][0]["probabilities"]) == {"positive", "negative", "neutral"}


def test_token_usage_from_the_provider_is_summed_across_batches(monkeypatch):
    class Stub:
        def with_structured_output(self, schema, include_raw=False):
            def invoke(messages):
                ids = re.findall(r"H-\d{4}", messages[-1].content)
                parsed = ex.BatchLabels(items=[ex.LabelItem(headline_id=i, label="neutral", confidence=0.5) for i in ids])
                raw = AIMessage(content="", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

            return type("R", (), {"invoke": staticmethod(invoke)})()

    monkeypatch.setattr(ex, "get_llm", lambda *a, **k: Stub())
    res = ex.chat_structured({"headlines": drawn(30)["headlines"]})["results"][0]
    assert res["calls"] == 3 and res["usage"] == {"input_tokens": 300, "output_tokens": 60, "total_tokens": 360}


# --------------------------------------------------------------------------- no leakage, parallelism, isolation
def test_classifiers_only_ever_see_headline_texts(monkeypatch):
    seen = {}

    def spy(state):
        seen["keys"], seen["headline_keys"] = set(state), {k for h in state["headlines"] for k in h}
        return {"results": [{"name": "rules_baseline", "predictions": [], "latency_s": 0, "calls": 0, "usage": {}, "notes": []}]}

    monkeypatch.setattr(ex, "rules_baseline", spy)
    ex.build_graph().invoke({"sample_size": 6})
    assert seen["keys"] == {"headlines"}, "per-node input_schema: no truth, no hard flags, no seed"
    assert seen["headline_keys"] == {"headline_id", "headline"}


def test_the_three_branches_and_all_chat_batches_run_at_the_same_time(monkeypatch):
    # 3 chat batches + 1 Decisions request must all be in flight together or the barrier breaks and results go missing.
    barrier = threading.Barrier(4)
    real = ex.decide

    def handler(text: str) -> ex.BatchLabels:
        barrier.wait(timeout=5)
        return ex.BatchLabels(items=[ex.LabelItem(headline_id=i, label="neutral", confidence=0.5) for i in re.findall(r"H-\d{4}", text)])

    def waiting_decide(document, questions, **kw):
        barrier.wait(timeout=5)
        return real(document, questions, **kw)

    set_structured_handler(ex.BatchLabels, handler)
    monkeypatch.setattr(ex, "decide", waiting_decide)
    out = ex.build_graph().invoke({"sample_size": 30})
    assert out["metrics"]["chat_structured"]["answered"] == 30 and out["metrics"]["decisions_api"]["status"] == "ok"


def test_one_failing_branch_is_reported_as_failed_and_the_others_still_score(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("decisions endpoint unavailable")

    monkeypatch.setattr(ex, "decide", broken)
    out = ex.build_graph().invoke({"sample_size": 30})
    assert out["metrics"]["decisions_api"] == {"status": "failed", "error": "RuntimeError: decisions endpoint unavailable"}
    assert out["metrics"]["rules_baseline"]["status"] == "ok" and out["metrics"]["chat_structured"]["status"] == "ok"
    assert "| decisions_api | failed: RuntimeError" in out["table"]
    assert "decisions_api" not in out["verdict"] or out["verdict_source"] == "template"
    assert set(out["metrics"]["agreement"]["pairwise"]) == {"chat_structured vs rules_baseline"}


def test_a_reused_thread_does_not_accumulate_branch_results():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:6]}"}}
    g.invoke({"sample_size": 6}, cfg)
    g.invoke({"sample_size": 6}, cfg)
    assert sorted(r["name"] for r in g.get_state(cfg).values["results"]) == sorted(ex.CLASSIFIERS)


# --------------------------------------------------------------------------- imperfect model output
def test_headlines_the_chat_model_skips_count_as_wrong_and_are_reported():
    s = drawn(30)
    ids, truth = [h["headline_id"] for h in s["headlines"]], s["truth"]
    dropped = set(ids[:3])  # the model silently skips 3 headlines (all in batch 1)
    pin_chat(truth, set(ids), drop=dropped)
    res = ex.chat_structured({"headlines": s["headlines"]})["results"][0]
    assert sum(p["label"] is None for p in res["predictions"]) == 3
    assert any("3 of 10 headlines missing" in n for n in res["notes"])
    out = ex.build_graph().invoke({"sample_size": 30})  # same scripted handler
    m = out["metrics"]["chat_structured"]
    assert m["accuracy"] == pytest.approx(27 / 30) and m["answered"] == 27
    assert m["calibration"]["n_scored"] == 27, "unanswered items have no confidence to score"
    assert sum(r["none"] for r in m["confusion"].values()) == 3


def test_unknown_and_duplicate_ids_in_the_model_output_are_ignored():
    s = drawn(6)
    first = s["headlines"][0]["headline_id"]

    def handler(text):
        items = [ex.LabelItem(headline_id=h["headline_id"], label="neutral", confidence=0.5) for h in s["headlines"]]
        return ex.BatchLabels(items=[ex.LabelItem(headline_id=first, label="positive", confidence=0.9), *items, ex.LabelItem(headline_id="H-9999", label="negative", confidence=1.0)])

    set_structured_handler(ex.BatchLabels, handler)
    preds = {p["headline_id"]: p for p in ex.chat_structured({"headlines": s["headlines"]})["results"][0]["predictions"]}
    assert len(preds) == 6 and "H-9999" not in preds
    assert preds[first]["label"] == "positive", "the first answer for an id wins"


def test_a_failing_chat_batch_loses_only_its_own_headlines():
    s = drawn(30)
    calls = {"n": 0}
    lock = threading.Lock()

    def handler(text):
        with lock:
            calls["n"] += 1
            this = calls["n"]
        if "H-" in text and this == 2:
            raise RuntimeError("rate limited")
        return ex.BatchLabels(items=[ex.LabelItem(headline_id=i, label="neutral", confidence=0.5) for i in re.findall(r"H-\d{4}", text)])

    set_structured_handler(ex.BatchLabels, handler)
    res = ex.chat_structured({"headlines": s["headlines"]})["results"][0]
    assert sum(p["label"] is None for p in res["predictions"]) == 10
    assert any("failed: RuntimeError: rate limited" in n for n in res["notes"])


def test_offline_standins_fill_in_when_the_stub_returns_placeholder_ids_and_can_be_switched_off(monkeypatch):
    hs = drawn(30)["headlines"]
    filled = ex.chat_structured({"headlines": hs})["results"][0]
    assert all(p["label"] in ex.LABELS for p in filled["predictions"]) and any("OFFLINE stand-in" in n for n in filled["notes"])
    monkeypatch.setattr(ex, "OFFLINE_STANDINS", False)
    empty = ex.chat_structured({"headlines": hs})["results"][0]
    assert all(p["label"] is None for p in empty["predictions"])
    out = ex.build_graph().invoke({"sample_size": 30})
    m = out["metrics"]["chat_structured"]
    assert m["accuracy"] == 0.0 and m["calibration"] is None and m["answered"] == 0
    assert "| chat_structured | 0.000 |" in out["table"] and "n/a" in out["table"]


# --------------------------------------------------------------------------- the verdict
def test_verdict_that_quotes_the_table_is_kept():
    def writer(messages, tools):
        table = str(messages[-1].content)
        nums = sorted(ex.table_numbers(table))[:4]
        return AIMessage(content=f"On this sample the best accuracy is shown in the table; the figures {', '.join(nums)} support that.")

    set_chat_handler(writer)
    out = ex.build_graph().invoke({"sample_size": 30})
    assert out["verdict_source"] == "model" and ex.quotes_table(out["verdict"], out["table"])


def test_verdict_that_does_not_quote_the_table_is_replaced_by_a_computed_one():
    s = drawn(30)
    ids, truth = [h["headline_id"] for h in s["headlines"]], s["truth"]
    pin_chat(truth, set(ids))  # chat perfect: accuracy 1.000
    pin_decisions(truth, set(ids[:15]), confidence=0.8)
    set_chat_handler(lambda messages, tools: AIMessage(content="Decisions API is clearly the best, trust me."))
    out = ex.build_graph().invoke({"sample_size": 30})
    assert out["verdict_source"] == "template"
    assert "best accuracy chat_structured (1.000)" in out["verdict"]
    assert not out["verdict"].startswith("Decisions API is clearly")


def test_computed_verdict_names_ties_and_warns_about_small_hard_subsets():
    metrics = {n: {"status": "ok", "accuracy": 0.9, "hard": {"n": 4, "accuracy": 0.5 if n != "rules_baseline" else 0.0},
                   "calibration": None if n == "rules_baseline" else {"ece": 0.05 if n == "chat_structured" else 0.2}} for n in ex.CLASSIFIERS}
    text = ex.deterministic_verdict(metrics, {"n": 30, "n_hard": 4}, "live")
    assert "best accuracy rules_baseline / chat_structured / decisions_api (0.900)" in text
    assert "best on hard headlines chat_structured / decisions_api (0.500)" in text
    assert "lowest calibration error chat_structured (0.050)" in text
    assert "Only 4 hard headlines" in text and "meaningless" not in text


def test_offline_default_verdict_is_computed_and_flagged_meaningless():
    out = ex.build_graph().invoke(ex.EXAMPLE_INPUT)
    assert out["verdict_source"] == "template" and "meaningless" in out["verdict"]

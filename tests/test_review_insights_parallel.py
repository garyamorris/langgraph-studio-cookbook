from __future__ import annotations

import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import scripted, set_structured_handler
from deepdive.common.testing import check_example
from deepdive.examples import review_insights_parallel as ex

PLUG = "Lumen Plug 4-pack"
BURNING_PLUG = "R-0315"


def no_risks():
    return ex.RiskScan(findings=[])


def themes(*pairs):
    return ex.ThemeReport(
        themes=[ex.Theme(name=n, sentiment="positive", summary=f"{n} summary", evidence_review_ids=list(ids)) for n, ids in pairs]
    )


def newest(product: str, n: int) -> pd.DataFrame:
    df = read_csv("product_reviews.csv")
    return df[df["product"] == product].sort_values(["date", "review_id"], ascending=[False, False]).head(n)


def test_contract_and_offline_run():
    out = check_example(ex).output
    assert out["mode"] == "offline" and out["reviews_analyzed"] == 53


# --------------------------------------------------------------------------- the guarantee
def test_burning_plug_is_flagged_even_when_the_model_flags_nothing():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    out = ex.graph.invoke({"product": PLUG})
    flagged = {r["review_id"]: r for r in out["risks"]}
    assert BURNING_PLUG in flagged, "the keyword prefilter must catch the burning plug without the model"
    assert flagged[BURNING_PLUG]["detected_by"] == ["keyword"]
    assert flagged[BURNING_PLUG]["severity"] == "unrated" and "fire_or_overheating" in flagged[BURNING_PLUG]["categories"]
    assert {"hot", "burning", "scorch"} <= set(flagged[BURNING_PLUG]["matched_terms"])


def test_risks_section_comes_first_when_anything_is_flagged():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    set_structured_handler(ex.ThemeReport, scripted([themes(("Easy setup", ["R-0202", "R-0168"]))]))
    digest = ex.graph.invoke({"product": PLUG})["digest"]
    sections = [line for line in digest.splitlines() if line.startswith("## ")]
    assert sections[0].startswith("## Risks needing attention (1)"), sections
    assert sections.index("## Ratings") < sections.index("## Themes")
    assert BURNING_PLUG in digest.split("## Ratings")[0], "the risk is listed before the ratings"


def test_model_only_findings_are_included_and_double_detections_are_merged():
    llm = ex.RiskScan(
        findings=[
            ex.RiskFinding(review_id=BURNING_PLUG, category="fire_or_overheating", severity="critical", reason="Scorch mark on socket."),
            ex.RiskFinding(review_id="R-0202", category="other_safety", severity="medium", reason="Loose fit in socket."),
        ]
    )
    set_structured_handler(ex.RiskScan, scripted([llm]))
    out = ex.graph.invoke({"product": PLUG})
    by_id = {r["review_id"]: r for r in out["risks"]}
    assert by_id[BURNING_PLUG]["detected_by"] == ["keyword", "llm"], "one entry per review, both detectors recorded"
    assert by_id[BURNING_PLUG]["severity"] == "critical" and by_id[BURNING_PLUG]["reason"] == "Scorch mark on socket."
    assert by_id["R-0202"]["detected_by"] == ["llm"], "a risk only the model sees still reaches the digest (union, not intersection)"
    assert [r["review_id"] for r in out["risks"]][0] == BURNING_PLUG, "worst severity first"
    assert sum(r["review_id"] == BURNING_PLUG for r in out["risks"]) == 1


def test_invented_review_ids_from_the_model_are_discarded():
    llm = ex.RiskScan(findings=[ex.RiskFinding(review_id="R-9999", category="injury", severity="critical", reason="made up")])
    set_structured_handler(ex.RiskScan, scripted([llm]))
    set_structured_handler(ex.ThemeReport, scripted([themes(("Ghost theme", ["R-9999"]), ("Real theme", ["R-9999", "R-0202"]))]))
    out = ex.graph.invoke({"product": PLUG})
    assert "R-9999" not in {r["review_id"] for r in out["risks"]}
    assert [t["name"] for t in out["themes"]] == ["Real theme"], "a theme with no real evidence is dropped"
    assert out["themes"][0]["evidence_review_ids"] == ["R-0202"], "invented ids are removed, real ones kept"


def test_product_without_risk_language_has_no_risks_section_but_says_the_scan_ran():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    out = ex.graph.invoke({"product": "Lumen Bell"})
    assert out["risks"] == []
    sections = [line for line in out["digest"].splitlines() if line.startswith("## ")]
    assert sections == ["## Ratings", "## Themes", "## Risk check"]
    assert "Nothing flagged" in out["digest"]


@pytest.mark.parametrize(
    "product,review_id",
    [("Lumen Therm", "R-0316"), ("Lumen Cam Pro", "R-0317"), ("Lumen Bulb 6-pack", "R-0318"), ("Lumen Hub", "R-0320"), ("Lumen Lock", "R-0319")],
)
def test_every_seeded_risk_review_is_caught_by_the_keyword_scan_alone(product, review_id):
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    out = ex.graph.invoke({"product": product})
    assert review_id in {r["review_id"] for r in out["risks"]}


def test_keyword_scan_uses_word_boundaries_and_names_the_matches():
    assert ex.keyword_hits("I took a photo and a shot of the room, no problems") == {}
    assert ex.keyword_hits("Works with Fire TV and Fire Stick, great") == {}
    assert ex.keyword_hits("The auto-lock fired while I was outside")  == {}
    assert ex.keyword_hits("Smelled like burning and then it sparked") == {"fire_or_overheating": ["smelled", "burning", "sparked"]}
    assert set(ex.keyword_hits("I am talking to a lawyer; it is illegal. Another household's feed.")) == {"legal_threat", "privacy"}


# --------------------------------------------------------------------------- the three branches
def test_all_three_branch_outputs_are_present_and_consistent():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    set_structured_handler(ex.ThemeReport, scripted([themes(("Easy setup", ["R-0202", "R-0168"]), ("Reliable", ["R-0202"]))]))
    out = ex.graph.invoke({"product": PLUG})
    # (a) statistics equal an independent pandas computation
    df = newest(PLUG, 60)
    assert out["stats"]["n"] == len(df) == out["reviews_analyzed"]
    assert out["stats"]["mean_rating"] == pytest.approx(df["rating"].mean(), abs=0.005)
    assert out["stats"]["histogram"] == {str(s): int((df["rating"] == s).sum()) for s in range(1, 6)}
    assert sum(out["stats"]["histogram"].values()) == out["stats"]["n"]
    assert out["stats"]["verified_share"] == pytest.approx(df["verified_purchase"].mean(), abs=0.001)
    # (b) themes carry their evidence
    assert [t["name"] for t in out["themes"]] == ["Easy setup", "Reliable"]
    assert out["themes"][0]["evidence_review_ids"] == ["R-0202", "R-0168"]
    # (c) risks
    assert out["risks"], "risk branch contributed"
    for section in ("Ratings", "Themes", "Risks needing attention"):
        assert section in out["digest"], f"{section} missing from the digest"
    assert "Easy setup (positive)" in out["digest"] and "R-0202, R-0168" in out["digest"]


def test_max_reviews_is_respected_and_the_newest_are_used():
    seen = {}

    def spy(prompt: str):
        seen["themes_prompt"] = prompt
        return themes(("Anything", ["R-0315"]))

    set_structured_handler(ex.ThemeReport, spy)
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    out = ex.graph.invoke({"product": PLUG, "max_reviews": 10})
    expected = newest(PLUG, 10)
    assert out["reviews_analyzed"] == out["stats"]["n"] == 10
    assert out["stats"]["date_from"] == expected["date"].min() and out["stats"]["date_to"] == expected["date"].max()
    assert seen["themes_prompt"].count("[R-0") == 10, "the model must only be shown the newest 10 reviews"
    assert all(rid in seen["themes_prompt"] for rid in expected["review_id"])
    older = set(read_csv("product_reviews.csv").query("product == @PLUG").review_id) - set(expected["review_id"])
    assert not any(f"[{rid}]" in seen["themes_prompt"] for rid in older)


def test_reviews_older_than_the_window_are_not_reported():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    window = ex.graph.invoke({"product": "Lumen Cam Pro", "max_reviews": 5})
    assert window["risks"] == [], "R-0317 is the 8th newest Cam Pro review, outside a window of 5"
    wider = ex.graph.invoke({"product": "Lumen Cam Pro", "max_reviews": 8})
    assert [r["review_id"] for r in wider["risks"]] == ["R-0317"]


def test_digest_is_written_exactly_once_after_all_branches():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    events = list(ex.graph.stream({"product": PLUG}, stream_mode="updates"))
    names = [name for event in events for name in event]
    assert names.count("write_digest") == 1
    assert names.index("write_digest") > max(names.index(n) for n in ("rating_stats", "find_themes", "keyword_risk_scan", "llm_risk_scan"))
    assert names[-1] == "write_digest"


def test_findings_do_not_pile_up_when_the_same_thread_runs_twice():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "studio-thread"}}
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    first = g.invoke({"product": PLUG}, cfg)
    second = g.invoke({"product": PLUG}, cfg)
    assert len(first["risks"]) == len(second["risks"]) == 1
    assert len(g.get_state(cfg).values["risk_findings"]) == 1, "the reducer list is reset at the start of each run"
    other = g.invoke({"product": "Lumen Bell"}, cfg)
    assert other["risks"] == [], "findings from the previous product must not leak into this one"


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="Unknown product 'Lumen Toaster'"):
        ex.graph.invoke({"product": "Lumen Toaster"})
    with pytest.raises(ValueError, match="Unknown product ''"):
        ex.graph.invoke({"max_reviews": 5})  # a window with no product is still a bad value, not a blank form
    with pytest.raises(ValueError, match="Unknown ready-made case"):
        ex.graph.invoke({"scenario": "not a case"})
    with pytest.raises(ValueError, match="max_reviews must be"):
        ex.graph.invoke({"product": PLUG, "max_reviews": 0})
    with pytest.raises(ValueError, match="max_reviews must be"):
        ex.graph.invoke({"product": PLUG, "max_reviews": 5000})
    assert ex.graph.invoke({"product": "lumen plug 4-PACK"})["product"] == PLUG, "product match ignores case"


# --------------------------------------------------------------------------- ready-made cases (Studio's dropdown)
def test_a_blank_form_runs_the_first_ready_made_case():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    out = ex.graph.invoke({})  # Studio's Submit on an empty form: no dead end
    assert out["product"] == PLUG and out["reviews_analyzed"] == 53 and BURNING_PLUG in {r["review_id"] for r in out["risks"]}


# label start -> (reviews of the product, reviews analysed, review ids the keyword scan must flag in that window)
CASE_FACTS = {
    "Lumen Plug 4-pack, all 53": (53, 53, {"R-0315"}),
    "Lumen Lock, all 22": (22, 22, {"R-0319", "R-0167", "R-0262"}),
    "Lumen Therm, newest 20 of 40": (40, 20, {"R-0316"}),
    "Lumen Cam Pro, newest 8 of 57": (57, 8, {"R-0317"}),
    "Lumen Cam Pro, newest 5 of 57": (57, 5, set()),
    "Lumen Hub, all 44": (44, 44, {"R-0320"}),
    "Lumen Bell, all 33": (33, 33, set()),
}


def test_every_ready_made_case_agrees_with_its_label_and_the_keyword_scan_alone_proves_it():
    """The labels claim which reviews get flagged. Only the keyword scan can promise that (the model may add more)."""
    assert [k for k in ex.CASES if not any(k.startswith(start) for start in CASE_FACTS)] == [], "a case has no facts row"
    df = read_csv("product_reviews.csv")
    for label, fields in ex.CASES.items():
        start = next(s for s in CASE_FACTS if label.startswith(s))
        total, analysed, flagged = CASE_FACTS[start]
        product = fields["product"]
        assert label.startswith(product), f"{label}: the label must lead with its product"
        assert (df["product"] == product).sum() == total, f"{label}: the product has {(df['product'] == product).sum()} reviews"
        set_structured_handler(ex.RiskScan, scripted([no_risks()]))
        out = ex.graph.invoke(dict(fields))
        assert out["reviews_analyzed"] == analysed, label
        assert {r["review_id"] for r in out["risks"]} == flagged, f"{label}: the keyword scan disagrees with the label"
        assert all(i in label for i in flagged), f"{label}: name the flagged reviews"
        assert all(r["detected_by"] == ["keyword"] for r in out["risks"]), "with a silent model only the keywords flag"


def test_a_case_with_no_flags_ends_with_the_risk_check_and_a_case_with_flags_opens_with_risks():
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    quiet = next(k for k in ex.CASES if k.startswith("Lumen Bell"))
    loud = next(k for k in ex.CASES if k.startswith("Lumen Lock"))
    assert "## Risk check" in ex.graph.invoke({"scenario": quiet})["digest"]
    sections = [line for line in ex.graph.invoke({"scenario": loud})["digest"].splitlines() if line.startswith("## ")]
    assert sections[0] == "## Risks needing attention (3)"


def test_picking_a_case_after_another_starts_from_a_clean_slate_on_the_same_thread():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "pick-twice"}}
    set_structured_handler(ex.RiskScan, scripted([no_risks()]))
    lock = next(k for k in ex.CASES if k.startswith("Lumen Lock"))
    cam5 = next(k for k in ex.CASES if k.startswith("Lumen Cam Pro, newest 5"))
    cam8 = next(k for k in ex.CASES if k.startswith("Lumen Cam Pro, newest 8"))
    assert len(g.invoke({"scenario": lock}, cfg)["risks"]) == 3
    assert g.invoke({"scenario": cam5}, cfg)["risks"] == [], "the Lock's findings must not leak into the next pick"
    assert g.invoke({"scenario": cam8}, cfg)["reviews_analyzed"] == 8
    assert g.invoke({"scenario": cam5}, cfg)["reviews_analyzed"] == 5, "a case with a window replaces the previous window"
    assert g.invoke({"scenario": lock}, cfg)["reviews_analyzed"] == 22, "None means the default window, not the previous 5"


def test_several_model_findings_for_one_review_keep_the_worst_severity():
    llm = ex.RiskScan(
        findings=[
            ex.RiskFinding(review_id="R-0202", category="other_safety", severity="medium", reason="Loose fit."),
            ex.RiskFinding(review_id="R-0202", category="injury", severity="high", reason="Pinched a finger."),
            ex.RiskFinding(review_id="R-0168", category="other_safety", severity="medium", reason="Plastic smell."),
        ]
    )
    set_structured_handler(ex.RiskScan, scripted([llm]))
    by_id = {r["review_id"]: r for r in ex.graph.invoke({"product": PLUG})["risks"]}
    assert by_id["R-0202"]["severity"] == "high" and by_id["R-0202"]["categories"] == ["other_safety", "injury"]
    assert by_id["R-0202"]["reason"] == "Loose fit. Pinched a finger."
    assert by_id["R-0168"]["severity"] == "medium", "a model rating replaces 'unrated'"


def test_review_ids_written_with_brackets_or_prefixes_are_still_matched():
    assert [ex.normalize_review_id(x) for x in ("R-0315", "[R-0315]", " r-0315 ", "Review R-0315:", "nonsense")] == ["R-0315"] * 4 + ["nonsense"]
    llm = ex.RiskScan(findings=[ex.RiskFinding(review_id="[R-0202]", category="injury", severity="high", reason="x")])
    set_structured_handler(ex.RiskScan, scripted([llm]))
    set_structured_handler(ex.ThemeReport, scripted([themes(("Setup", ["[R-0202]", "R-0168"]))]))
    out = ex.graph.invoke({"product": PLUG})
    assert "R-0202" in {r["review_id"] for r in out["risks"]}
    assert out["themes"][0]["evidence_review_ids"] == ["R-0202", "R-0168"]

"""The sales query layer is checked against plain pandas computed independently from the raw CSV."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from deepdive.common.data import read_csv
from deepdive.examples import _sales_tools as st


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    df = read_csv("sales.csv")
    df["d"] = pd.to_datetime(df["order_date"])
    return df


def completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] != "cancelled"]


# --------------------------------------------------------------------------- metrics against pandas
def test_total_metrics_match_pandas(raw):
    out = st.query_sales(metrics=st.METRICS)
    d = completed(raw)
    assert out["total"]["revenue"] == pytest.approx(d["revenue"].sum(), abs=0.01)
    assert out["total"]["units"] == d["quantity"].sum()
    assert out["total"]["orders"] == d["order_id"].nunique()
    assert out["total"]["margin"] == pytest.approx(d["margin"].sum(), abs=0.01)
    assert out["total"]["return_rate"] == pytest.approx((d["status"] == "returned").mean() * 100, abs=0.05)
    assert out["total"]["avg_discount"] == pytest.approx(d["discount_pct"].mean(), abs=0.05)
    assert out["lines"] == len(d)


@pytest.mark.parametrize("group_by,column", [("product", "product"), ("region", "region"), ("channel", "channel"), ("category", "category"), ("segment", "segment")])
def test_group_by_matches_pandas_and_is_ranked(raw, group_by, column):
    out = st.query_sales(metrics=["revenue", "units"], group_by=group_by)
    expected = completed(raw).groupby(column).agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
    got = {r["group"]: r for r in out["rows"]}
    assert set(got) == set(expected.index)
    for key, row in expected.iterrows():
        assert got[key]["revenue"] == pytest.approx(row["revenue"], abs=0.01)
        assert got[key]["units"] == row["units"]
    revenues = [r["revenue"] for r in out["rows"]]
    assert revenues == sorted(revenues, reverse=True), "non-time groups are ranked by the first metric, highest first"


def test_time_groups_are_chronological_and_quarter_labels_are_right(raw):
    months = st.query_sales(metrics=["revenue"], group_by="month")["rows"]
    assert [r["group"] for r in months] == sorted(r["group"] for r in months)
    assert months[0]["group"] == "2025-11" and months[-1]["group"] == "2026-10"
    quarters = st.query_sales(metrics=["revenue", "units"], group_by="quarter")["rows"]
    assert [r["group"] for r in quarters] == ["2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4"]
    c = completed(raw)
    q2 = c[(c["d"] >= "2026-04-01") & (c["d"] <= "2026-06-30")]
    assert quarters[2]["revenue"] == pytest.approx(q2["revenue"].sum(), abs=0.01)
    assert sum(r["units"] for r in quarters) == completed(raw)["quantity"].sum()


def test_status_semantics_completed_all_delivered_returned(raw):
    revenue = lambda **kw: st.query_sales(metrics=["revenue"], **kw)["total"]["revenue"]  # noqa: E731
    by_status = raw.groupby("status")["revenue"].sum()
    assert revenue(status="all") == pytest.approx(raw["revenue"].sum(), abs=0.01)
    assert revenue(status="cancelled") == pytest.approx(by_status["cancelled"], abs=0.01)
    assert revenue(status="returned") == pytest.approx(by_status["returned"], abs=0.01)
    assert revenue(status="delivered") == pytest.approx(by_status["delivered"], abs=0.01)
    assert revenue() == pytest.approx(by_status["delivered"] + by_status["returned"], abs=0.01)
    assert st.query_sales(metrics=["return_rate"], status="delivered")["total"]["return_rate"] == 0.0
    grouped = st.query_sales(metrics=["orders"], group_by="status", status="all")["rows"]
    assert {r["group"] for r in grouped} == {"delivered", "returned", "cancelled"}


def test_return_rate_story_cam_pro_recall_window(raw):
    cam = completed(raw)[completed(raw)["product"] == "Lumen Cam Pro"]
    q2 = cam[(cam["d"] >= "2026-04-01") & (cam["d"] <= "2026-06-30")]
    out = st.query_sales(metrics=["return_rate"], product="Lumen Cam Pro", period="2026Q2")
    assert out["total"]["return_rate"] == pytest.approx((q2["status"] == "returned").mean() * 100, abs=0.05)
    assert out["total"]["return_rate"] > 15, "the recall window is the built-in story"
    by_region = st.query_sales(metrics=["return_rate"], group_by="region", product="Lumen Cam Pro", period="2026Q2")["rows"]
    expected = q2.groupby("region").apply(lambda g: (g["status"] == "returned").mean() * 100, include_groups=False)
    assert by_region[0]["group"] == expected.idxmax(), "rows are ranked by the first metric"


def test_top_n_sort_by_and_ascending():
    top = st.query_sales(metrics=["revenue", "units"], group_by="product", top_n=3)
    assert len(top["rows"]) == 3 and top["rows_matched"] == 8
    assert top["total"]["revenue"] == st.query_sales(metrics=["revenue"])["total"]["revenue"], "total ignores top_n"
    by_units = st.query_sales(metrics=["revenue", "units"], group_by="product", sort_by="units", top_n=2)["rows"]
    assert by_units[0]["units"] >= by_units[1]["units"]
    worst = st.query_sales(metrics=["revenue"], group_by="product", ascending=True, top_n=1)["rows"][0]
    best = st.query_sales(metrics=["revenue"], group_by="product", top_n=1)["rows"][0]
    assert worst["revenue"] < best["revenue"]
    best_months = st.query_sales(metrics=["revenue"], group_by="month", top_n=2)["rows"]
    assert best_months[0]["revenue"] >= best_months[1]["revenue"], "top_n ranks time groups by value, not by date"


# --------------------------------------------------------------------------- filters and windows
def test_filters_are_canonicalised_and_applied(raw):
    out = st.query_sales(metrics=["units"], product="cam pro", region="north america", channel="WEB", segment="consumer")
    assert out["filters"] == {"product": "Lumen Cam Pro", "region": "North America", "channel": "web", "segment": "consumer", "status": "completed"}
    d = completed(raw)
    d = d[(d["product"] == "Lumen Cam Pro") & (d["region"] == "North America") & (d["channel"] == "web") & (d["segment"] == "consumer")]
    assert out["total"]["units"] == d["quantity"].sum()
    assert st.query_sales(metrics=["units"], product="Therm")["filters"]["product"] == "Lumen Therm"
    assert st.query_sales(metrics=["units"], product="plug")["filters"]["product"] == "Lumen Plug 4-pack"


def test_date_windows_period_and_explicit_dates_agree(raw):
    by_period = st.query_sales(metrics=["revenue"], period="2026-05")
    by_dates = st.query_sales(metrics=["revenue"], start_date="2026-05-01", end_date="2026-05-31")
    c = completed(raw)
    d = c[(c["d"] >= "2026-05-01") & (c["d"] <= "2026-05-31")]
    assert by_period["total"] == by_dates["total"]
    assert by_period["total"]["revenue"] == pytest.approx(d["revenue"].sum(), abs=0.01)
    assert by_period["window"] == {"start": "2026-05-01", "end": "2026-05-31"}
    one_day = st.query_sales(metrics=["orders"], period="2026-05-04")
    assert one_day["window"]["start"] == one_day["window"]["end"] == "2026-05-04"


def test_window_beyond_the_data_is_noted_but_fully_outside_is_an_error():
    clipped = st.query_sales(metrics=["revenue"], period="2026Q4")
    assert any("extends beyond" in n for n in clipped["notes"])
    with pytest.raises(ValueError, match="outside the data"):
        st.query_sales(metrics=["revenue"], period="2024Q1")
    empty = st.query_sales(metrics=["revenue"], product="Lumen Lock", region="APAC", period="2026-05-01..2026-05-02")
    assert empty["lines"] == 0 and any("No order lines" in n for n in empty["notes"])


# --------------------------------------------------------------------------- bad input
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"group_by": "colour"}, "group_by must be one of"),
        ({"metrics": ["profit"]}, "Unsupported metric 'profit'"),
        ({"product": "Lumen Thermm"}, "Did you mean 'Lumen Therm'"),
        ({"product": "__import__('os').system('ls')"}, "Unknown product"),
        ({"region": "Mars"}, "Valid region values"),
        ({"status": "pending"}, "status must be one of"),
        ({"top_n": 0, "group_by": "product"}, "top_n must be"),
        ({"top_n": 500, "group_by": "product"}, "top_n must be"),
        ({"top_n": 3}, "needs a group_by"),
        ({"sort_by": "units"}, "must be one of the requested metrics"),
        ({"start_date": "2026-13-45"}, "not a real calendar date"),
        ({"start_date": "last tuesday"}, "calendar date like 2026-05-01"),
        ({"start_date": "2026-06-01", "end_date": "2026-05-01"}, "after end_date"),
        ({"period": "2026Q2", "start_date": "2026-05-01"}, "either period or"),
        ({"period": "Q5 2026"}, "Could not read period"),
        ({"period": "2026-13"}, "must be 01-12"),
    ],
)
def test_invalid_arguments_raise_clear_value_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        st.query_sales(**kwargs)


def test_unknown_values_list_what_is_valid():
    with pytest.raises(ValueError) as e:
        st.query_sales(product="Lumen Toaster")
    assert "Lumen Cam Pro" in str(e.value) and "Lumen Therm" in str(e.value)


# --------------------------------------------------------------------------- quarters and periods
@pytest.mark.parametrize("text", ["2026Q2", "2026q2", "2026-Q2", "Q2 2026", " 2026 Q2 "])
def test_parse_quarter_accepts_common_spellings(text):
    q = st.parse_quarter(text)
    assert (q.label, q.start, q.end) == ("2026Q2", date(2026, 4, 1), date(2026, 6, 30))


@pytest.mark.parametrize("text", ["2026Q5", "Q0 2026", "2026", "next quarter", "", "2026Q"])
def test_parse_quarter_rejects_nonsense(text):
    with pytest.raises(ValueError, match="Could not read quarter"):
        st.parse_quarter(text)


def test_quarter_arithmetic_and_last_full_quarter():
    assert st.parse_quarter("2026Q1").previous().label == "2025Q4"
    assert st.parse_quarter("2026Q4").end == date(2026, 12, 31)
    assert st.quarter_of(date(2026, 10, 31)).label == "2026Q4"
    assert st.last_full_quarter().label == "2026Q3", "AS_OF is inside the (partial) 2026Q4"
    assert st.last_full_quarter(date(2026, 9, 30)).label == "2026Q3"
    assert st.last_full_quarter(date(2026, 1, 15)).label == "2025Q4"
    assert st.AS_OF == date(2026, 10, 31)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026Q3", (date(2026, 7, 1), date(2026, 9, 30))),
        ("2026-02", (date(2026, 2, 1), date(2026, 2, 28))),
        ("2026", (date(2026, 1, 1), date(2026, 12, 31))),
        ("2026-05-01", (date(2026, 5, 1), date(2026, 5, 1))),
        ("2026-02..2026-04", (date(2026, 2, 1), date(2026, 4, 30))),
        ("2026-04-15..2026Q3", (date(2026, 4, 15), date(2026, 9, 30))),
    ],
)
def test_parse_period(text, expected):
    assert st.parse_period(text) == expected


def test_quarter_windows_are_like_for_like():
    q4 = st.quarter_windows(st.parse_quarter("2026Q4"))
    assert q4["partial"] and (q4["start"], q4["end"]) == (date(2026, 10, 1), date(2026, 10, 31))
    assert (q4["prior_start"], q4["prior_end"]) == (date(2026, 7, 1), date(2026, 7, 31)) and q4["prior_comparable"]
    q2 = st.quarter_windows(st.parse_quarter("2026Q2"))
    assert not q2["partial"] and (q2["prior_start"], q2["prior_end"]) == (date(2026, 1, 1), date(2026, 3, 31))
    assert not st.quarter_windows(st.parse_quarter("2025Q4"))["prior_comparable"], "no data before 2025-11-01"
    assert not st.quarter_windows(st.parse_quarter("2026Q1"))["prior_comparable"], "prior quarter is only half covered"
    with pytest.raises(ValueError, match="No data for 2024Q1"):
        st.quarter_windows(st.parse_quarter("2024Q1"))


# --------------------------------------------------------------------------- compare_periods
def test_compare_periods_matches_pandas(raw):
    out = st.compare_periods(metric="units", period_a="2026-02..2026-04", period_b="2026-05..2026-07", product="Lumen Therm")
    d = completed(raw)
    d = d[d["product"] == "Lumen Therm"]
    a = d[(d["d"] >= "2026-02-01") & (d["d"] <= "2026-04-30")]["quantity"].sum()
    b = d[(d["d"] >= "2026-05-01") & (d["d"] <= "2026-07-31")]["quantity"].sum()
    assert (out["a"], out["b"], out["change"]) == (a, b, b - a)
    assert out["pct_change"] == pytest.approx((b - a) / a * 100, abs=0.05)
    assert any("different lengths" in n for n in out["notes"]), "89 vs 92 days must be flagged"


def test_compare_periods_by_group_and_percent_metrics(raw):
    out = st.compare_periods(metric="return_rate", period_a="2026Q1", period_b="2026Q2", group_by="product")
    d = completed(raw)
    rate = lambda df: (df["status"] == "returned").mean() * 100  # noqa: E731
    a = d[(d["d"] >= "2026-01-01") & (d["d"] <= "2026-03-31")]
    b = d[(d["d"] >= "2026-04-01") & (d["d"] <= "2026-06-30")]
    assert out["a"] == pytest.approx(rate(a), abs=0.05) and out["b"] == pytest.approx(rate(b), abs=0.05)
    assert out["change"] == pytest.approx(out["b"] - out["a"], abs=0.1) and out["unit"] == "%"
    assert out["groups"][0]["group"] == "Lumen Cam Pro", "largest increase first: the recall"
    assert any("different lengths" in n for n in out["notes"]), "Q1 has 90 days, Q2 has 91"


def test_compare_periods_validates_inputs():
    with pytest.raises(ValueError, match="exactly one metric"):
        st.compare_periods(metric="revenue,units", period_a="2026Q1", period_b="2026Q2")
    with pytest.raises(ValueError, match="period_b is required"):
        st.compare_periods(metric="revenue", period_a="2026Q1", period_b="")
    with pytest.raises(ValueError, match="Unknown product"):
        st.compare_periods(metric="revenue", period_a="2026Q1", period_b="2026Q2", product="Nope")
    zero = st.compare_periods(metric="revenue", period_a="2026-05-01", period_b="2026-05-02", product="Lumen Lock", region="APAC")
    assert zero["pct_change"] is None or isinstance(zero["pct_change"], float)


# --------------------------------------------------------------------------- self-description
def test_describe_dataset_serves_models_the_vocabularies_and_relative_periods(raw):
    d = st.describe_dataset()
    assert set(d["products"]) == set(raw["product"]) and set(d["regions"]) == set(raw["region"])
    assert d["last_full_quarter"]["label"] == "2026Q3" and d["today"] == "2026-10-31"
    assert d["covers"] == {"start": "2025-11-01", "end": "2026-10-31"}
    assert set(d["metrics"]) == set(st.METRICS) and d["order_lines"] == len(raw)
    brief = st.dataset_brief()
    assert "2026Q3" in brief and "Lumen Cam Pro" in brief


def test_empty_status_means_the_default():
    assert st.query_sales(metrics=["orders"], status=None)["total"] == st.query_sales(metrics=["orders"])["total"]
    assert st.query_sales(metrics=["orders"], status="")["filters"]["status"] == "completed"


@pytest.mark.parametrize("word", ["all", "ALL", "none", "null", "n/a", "", "  "])
def test_words_that_mean_no_filter_are_not_errors(word):
    out = st.query_sales(metrics=["orders"], product=word, region=word)
    assert out["total"] == st.query_sales(metrics=["orders"])["total"] and "product" not in out["filters"]

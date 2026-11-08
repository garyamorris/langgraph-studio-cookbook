"""Unit tests for the deterministic math behind portfolio_desk_lite (analytics, sizing, limits, slicing, note audit).

Expected values are either computed by hand (the arithmetic is written out in the comments) or recomputed with an
independent formulation (plain pandas / a pure-Python Wilder loop on the raw CSV rows).
"""

from __future__ import annotations

import csv
import math
import random

import numpy as np
import pandas as pd
import pytest

from deepdive.common.config import data_dir
from deepdive.common.data import read_csv
from deepdive.examples import _desk_math as dm

NAV = 10_000_000.0


# --------------------------------------------------------------------------- analytics
def test_rsi_hand_computed():
    # closes 10, 12, 11, 13 -> changes +2, -1, +2; period 2.
    # seed: avg_gain = (2 + 0) / 2 = 1.0, avg_loss = (0 + 1) / 2 = 0.5
    # next change +2: avg_gain = (1.0 * 1 + 2) / 2 = 1.5, avg_loss = (0.5 * 1 + 0) / 2 = 0.25
    # RS = 6, RSI = 100 - 100 / 7 = 85.7142857...
    assert dm.rsi_wilder(np.array([10, 12, 11, 13.0]), period=2) == pytest.approx(100 - 100 / 7)


def test_rsi_edge_cases():
    assert dm.rsi_wilder(np.arange(1.0, 30.0)) == 100.0  # only gains
    assert dm.rsi_wilder(np.full(30, 5.0)) == 50.0  # no movement at all
    assert dm.rsi_wilder(np.arange(30.0, 1.0, -1.0)) == pytest.approx(0.0)  # only losses
    with pytest.raises(ValueError, match="at least 15"):
        dm.rsi_wilder(np.arange(10.0))


def _line(n=70):
    """close = 100, 101, ..., 169 and volume = 1000, 1010, ..., 1690 (n = 70 rows)."""
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-01-01", periods=n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [1000 + 10 * i for i in range(n)],
        }
    )


def test_analytics_on_a_straight_line_by_hand():
    a = dm.compute_analytics(_line())
    # last close 169; close[-6] = 164, close[-21] = 149
    assert a["last_close"] == 169.0
    assert a["ret_5d"] == pytest.approx(5 / 164, abs=1e-6)
    assert a["ret_20d"] == pytest.approx(20 / 149, abs=1e-6)
    assert a["rsi_14"] == 100.0  # no down day
    assert a["drawdown_60d"] == 0.0  # the last close is the 60d high
    # z-score: last 50 closes are 120..169, mean 144.5, sample std = sqrt(50 * 51 / 12) = sqrt(212.5)
    assert a["zscore_50d"] == pytest.approx((169 - 144.5) / math.sqrt(212.5), abs=1e-4)
    # volume of the last 20 rows is 1500, 1510, ..., 1690 -> mean 1595
    assert a["adv_20d_shares"] == 1595.0 and a["adv_20d_usd"] == round(1595 * 169)
    # the 20 daily returns are 1/149, 1/150, ... 1/168 (a step of 1 on a base of 149..168); annualised = daily * sqrt(252)
    daily = pd.Series([100.0 + i for i in range(70)]).pct_change().tail(20)
    assert a["vol_20d"] == pytest.approx(daily.std(), abs=1e-6)
    assert a["vol_20d_ann"] == pytest.approx(daily.std() * math.sqrt(252), abs=1e-5)


def test_analytics_sorts_by_date_and_needs_enough_history():
    shuffled = _line().sample(frac=1.0, random_state=1)
    assert dm.compute_analytics(shuffled) == dm.compute_analytics(_line())
    with pytest.raises(ValueError, match="at least 60"):
        dm.compute_analytics(_line(40))


def _raw_rows(ticker):
    with (data_dir() / "prices.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["ticker"] == ticker]
    rows.sort(key=lambda r: r["date"])
    return rows


@pytest.mark.parametrize("ticker", ["ACMR", "BORE", "CBLT", "DLPH", "EMBR", "FTHM", "GRNT", "HELO"])
def test_analytics_match_an_independent_pandas_and_pure_python_calculation(ticker):
    raw = _raw_rows(ticker)
    close = [float(r["close"]) for r in raw]
    vol = [float(r["volume"]) for r in raw]
    c = pd.Series(close)
    # Wilder RSI written as a plain loop over python floats
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    ag, al = sum(gains[:14]) / 14, sum(losses[:14]) / 14
    for g, l in zip(gains[14:], losses[14:]):
        ag, al = (ag * 13 + g) / 14, (al * 13 + l) / 14
    rsi = 100 - 100 / (1 + ag / al)

    prices = read_csv("prices.csv")
    a = dm.compute_analytics(prices[prices.ticker == ticker])
    assert a["asof"] == raw[-1]["date"] == "2026-10-30"
    assert a["last_close"] == pytest.approx(close[-1])
    assert a["ret_5d"] == pytest.approx(c.iloc[-1] / c.iloc[-6] - 1, abs=1e-6)
    assert a["ret_20d"] == pytest.approx(c.iloc[-1] / c.iloc[-21] - 1, abs=1e-6)
    assert a["vol_20d"] == pytest.approx(c.pct_change().tail(20).std(), abs=1e-6)  # pandas std is ddof=1 too
    assert a["rsi_14"] == pytest.approx(rsi, abs=1e-3)
    assert a["drawdown_60d"] == pytest.approx(c.iloc[-1] / c.tail(60).max() - 1, abs=1e-6)
    assert a["zscore_50d"] == pytest.approx((c.iloc[-1] - c.tail(50).mean()) / c.tail(50).std(), abs=1e-4)
    assert a["adv_20d_shares"] == pytest.approx(sum(vol[-20:]) / 20, abs=0.1)


def test_acmr_analytics_are_pinned():
    """Literal values (cross-checked against the independent formulas above) so a silent change of definition is caught."""
    prices = read_csv("prices.csv")
    a = dm.compute_analytics(prices[prices.ticker == "ACMR"])
    assert (a["ret_5d"], a["ret_20d"], a["vol_20d"]) == pytest.approx((0.095223, 0.112322, 0.026149), abs=1e-6)
    assert (a["rsi_14"], a["drawdown_60d"], a["zscore_50d"]) == pytest.approx((65.416, -0.04184, 1.655), abs=1e-3)
    assert a["adv_20d_shares"] == pytest.approx(1_789_230.1, abs=0.1)


def test_recent_headlines_window_order_and_no_labels():
    news = pd.DataFrame(
        {
            "headline_id": ["H-1", "H-2", "H-3", "H-4", "H-5"],
            "date": ["2026-09-01", "2026-10-05", "2026-10-30", "2026-10-30", "2026-11-02"],
            "ticker": ["X"] * 5,
            "headline": ["too old", "inside window", "newest B", "newest A", "after the as-of date"],
            "label": ["negative"] * 5,
        }
    )
    got = dm.recent_headlines(news, "2026-10-30", days=30, k=3)
    assert [h["headline"] for h in got] == ["newest B", "newest A", "inside window"]  # newest first, ties by id
    assert all(set(h) == {"date", "headline"} for h in got)  # the dataset's sentiment label is never passed on
    assert dm.recent_headlines(news, "2026-10-30", days=30, k=1)[0]["headline"] == "newest B"


def test_fact_sheet_has_the_numbers_and_no_labels():
    prices, news = read_csv("prices.csv"), read_csv("news_headlines.csv")
    a = dm.compute_analytics(prices[prices.ticker == "ACMR"])
    heads = dm.recent_headlines(news[news.ticker == "ACMR"], a["asof"])
    meta = {"ticker": "ACMR", "company": "Acme Robotics", "sector": "industrial tech", "group": "technology", "beta": 1.25}
    sheet = dm.fact_sheet(meta, a, heads)
    assert "TICKER: ACMR" in sheet and "LAST CLOSE: 205.19" in sheet and "RETURN 5D: +9.5%" in sheet
    assert "AVERAGE DAILY VOLUME 20D: 1,789,230 shares" in sheet
    assert heads and all(h["headline"] in sheet for h in heads)


# --------------------------------------------------------------------------- groups, CIO validation, fallback
def test_sector_groups_cover_every_company_and_unknown_sectors_stand_alone():
    companies = read_csv("companies.csv")
    groups = {t: dm.sector_group(s) for t, s in zip(companies.ticker, companies.sector)}
    assert groups == {
        "ACMR": "technology", "CBLT": "technology", "FTHM": "technology",
        "BORE": "cyclicals", "HELO": "cyclicals", "GRNT": "cyclicals",
        "DLPH": "defensives", "EMBR": "defensives",
    }  # fmt: skip
    assert dm.sector_group("  Biotech ") == "biotech"
    # the reason groups exist: every sector in the file is unique, so a per-sector cap could never bind
    assert companies.sector.nunique() == len(companies)


def test_validate_focus_normalises_dedupes_and_caps():
    universe = {"ACMR", "BORE", "CBLT", "DLPH", "EMBR", "FTHM"}
    valid, dropped = dm.validate_focus([" acmr", "BORE", "bore", "XXXX", 7, "CBLT", "DLPH", "EMBR", "FTHM"], universe)
    assert valid == ["ACMR", "BORE", "CBLT", "DLPH", "EMBR"]  # normalised, de-duplicated, capped at 5
    assert dropped == ["bore", "XXXX", "7"]
    assert dm.validate_focus(None, universe) == ([], [])


@pytest.mark.parametrize(
    "raw, expected",
    [(0.45, 0.45), (45, 0.45), (0.1, 0.25), (0.95, 0.7), (1.5, 0.7), (150, 0.7), (-3, 0.25), (None, 0.45), ("abc", 0.45), (float("nan"), 0.45), ("0.5", 0.5)],
)
def test_clamp_budget(raw, expected):
    assert dm.clamp_budget(raw) == expected


UNIVERSE = [
    {"ticker": "AAAA", "company": "Alpha Corp", "sector": "software", "group": "technology", "ret_20d": 0.01},
    {"ticker": "BBBB", "company": "Beta Mining", "sector": "energy", "group": "cyclicals", "ret_20d": -0.20},
    {"ticker": "CCCC", "company": "Gamma Health", "sector": "healthcare", "group": "defensives", "ret_20d": 0.12},
    {"ticker": "DDDD", "company": "Delta Foods", "sector": "consumer staples", "group": "defensives", "ret_20d": 0.03},
    {"ticker": "EEEE", "company": "Eps Silicon", "sector": "semiconductors", "group": "technology", "ret_20d": -0.05},
]


def test_fallback_focus_hint_first_then_biggest_movers_and_is_deterministic():
    assert dm.fallback_focus("", UNIVERSE) == ["BBBB", "CCCC", "EEEE", "DDDD"]  # |20d| = .20, .12, .05, .03
    # the hint mentions a sector group (technology) and a ticker symbol (cccc): hinted names first, movers order inside
    got = dm.fallback_focus("technology plus CCCC please", UNIVERSE)
    assert got[:3] == ["CCCC", "EEEE", "AAAA"] and len(got) == 4
    assert dm.fallback_focus("technology plus CCCC please", UNIVERSE) == got
    assert dm.fallback_focus("alpha", UNIVERSE)[0] == "AAAA"  # first word of the company name
    assert dm.fallback_focus("healthcare", UNIVERSE)[0] == "CCCC"  # sector phrase
    assert len(dm.fallback_focus("defensives technology cyclicals", UNIVERSE)) == 5  # hint can fill the maximum


# --------------------------------------------------------------------------- decision interpretation
@pytest.mark.parametrize(
    "pick, p, expected",
    [
        ("LONG", 0.9, ("LONG", False)),
        ("SHORT", 0.40, ("SHORT", False)),  # exactly at the gate passes
        ("SHORT", 0.399, ("AVOID", True)),
        ("LONG", 0.0, ("AVOID", True)),
        ("AVOID", 0.0, ("AVOID", False)),  # AVOID is never "gated": it already is the safe answer
        (None, 0.99, ("AVOID", False)),  # a refused pick fails safe
        ("BUY", 0.99, ("AVOID", False)),
    ],
)
def test_resolve_direction(pick, p, expected):
    assert dm.resolve_direction(pick, p) == expected


def test_expected_level_conviction_and_tilt_helpers():
    levels = ["low", "medium", "high"]
    assert dm.expected_level({"low": 0.2, "medium": 0.5, "high": 0.3}, levels) == pytest.approx(0.5 + 0.6)  # 0*.2 + 1*.5 + 2*.3
    assert dm.expected_level({"low": 2, "medium": 2, "high": 4}, levels) == pytest.approx(1.25)  # normalised: .25/.25/.5
    assert dm.expected_level({}, levels, fallback=0.7) == 0.7
    assert [dm.conviction_mult(x) for x in (-1, 0, 1, 2, 5)] == [0.5, 0.5, 1.0, 1.5, 1.5]
    assert dm.apply_tail_override("NET_LONG", 0.70) == ("NEUTRAL", True)  # >= trigger
    assert dm.apply_tail_override("NET_LONG", 0.699) == ("NET_LONG", False)
    assert dm.apply_tail_override("NEUTRAL", 0.95) == ("DEFENSIVE", True)
    assert dm.apply_tail_override("DEFENSIVE", 0.95) == ("DEFENSIVE", False)  # can only get MORE defensive
    assert [dm.debate_winner(a, b) for a, b in [(1.5, 1.0), (1.0, 1.5), (1.0, 1.2), (1.0, 1.25)]] == ["bull", "bear", "draw", "draw"]


# --------------------------------------------------------------------------- sizing
def _cand(ticker, direction, conviction_score, vol, group="technology", price=100.0):
    return {
        "ticker": ticker, "direction": direction, "conviction_score": conviction_score, "group": group,
        "analytics": {"vol_20d": vol, "last_close": price},
    }  # fmt: skip


def test_size_book_weights_by_hand():
    # NEUTRAL tilt: long_mult = short_mult = 1. gross 900,000.
    #   A: conviction 1.0 -> mult 1.0, vol 0.02 -> raw 50      B: conviction 0 -> mult 0.5, vol 0.02 -> raw 25
    #   C (SHORT): conviction 2.0 -> mult 1.5, vol 0.03 -> raw 50      total 125
    #   A = 900,000 * 50/125 = 360,000;  B = 180,000;  C = 360,000
    legs = dm.size_book([_cand("AAA", "LONG", 1.0, 0.02), _cand("BBB", "LONG", 0.0, 0.02), _cand("CCC", "SHORT", 2.0, 0.03)], "NEUTRAL", 900_000, NAV)
    assert {l["ticker"]: l["notional_usd"] for l in legs} == {"AAA": 360_000, "BBB": 180_000, "CCC": 360_000}
    assert [l["side"] for l in legs] == ["LONG", "LONG", "SHORT"]
    assert legs[0]["pct_nav"] == 0.036 and legs[0]["history"] == [{"step": "desk", "scale": 1.0, "notional_usd": 360_000}]
    assert legs[0]["basis"]["share_of_gross"] == pytest.approx(0.4)


def test_size_book_tilt_reweights_sides_and_floors_to_lots():
    cands = [_cand("AAA", "LONG", 1.0, 0.02), _cand("BBB", "SHORT", 1.0, 0.02)]
    even = {l["ticker"]: l["notional_usd"] for l in dm.size_book(cands, "NEUTRAL", 1_000_000, NAV)}
    assert even == {"AAA": 500_000, "BBB": 500_000}
    # NET_LONG: short_mult 0.6 -> weights 1 : 0.6 -> 625,000 / 375,000.  DEFENSIVE: long_mult 0.6 -> 375,000 / 625,000
    nl = {l["ticker"]: l["notional_usd"] for l in dm.size_book(cands, "NET_LONG", 1_000_000, NAV)}
    df = {l["ticker"]: l["notional_usd"] for l in dm.size_book(cands, "DEFENSIVE", 1_000_000, NAV)}
    assert nl == {"AAA": 625_000, "BBB": 375_000} and df == {"AAA": 375_000, "BBB": 625_000}
    # three equal legs of 1,000,000: 333,333.33 each, rounded DOWN to a lot, so the sum never exceeds the gross
    thirds = dm.size_book([_cand(t, "LONG", 1.0, 0.02) for t in ("A", "B", "C")], "NEUTRAL", 1_000_000, NAV)
    assert [l["notional_usd"] for l in thirds] == [333_300] * 3


def test_size_book_inverse_volatility_with_a_floor():
    # a near-zero vol is floored at 0.005, so the calm name is not given an absurd weight: 0.005 vs 0.01 -> 2 : 1
    legs = dm.size_book([_cand("CALM", "LONG", 1.0, 0.0001), _cand("WILD", "LONG", 1.0, 0.01)], "NEUTRAL", 900_000, NAV)
    assert {l["ticker"]: l["notional_usd"] for l in legs} == {"CALM": 600_000, "WILD": 300_000}


def test_floor_lot_never_rounds_up():
    assert [dm.floor_lot(x) for x in (0, 99.99, 100, 1_499_999.99, 1_500_000.0)] == [0, 0, 100, 1_499_900, 1_500_000]
    assert dm.floor_lot(14999.999999999998 * 100) == 1_500_000  # float noise below a multiple of 100 does not cost a lot


# --------------------------------------------------------------------------- risk engine
def leg(ticker, side, notional, group):
    return {
        "ticker": ticker, "side": side, "notional_usd": float(notional), "group": group, "pct_nav": notional / NAV,
        "last_price": 100.0, "status": "proposed", "basis": {}, "history": [{"step": "desk", "scale": 1.0, "notional_usd": float(notional)}],
    }  # fmt: skip


def codes(breaches):
    return sorted((b["code"], b["scope"]) for b in breaches)


def test_each_limit_fires_on_its_own_and_boundaries_are_inclusive():
    budget = 6_000_000.0
    ok = [leg("A", "LONG", 1_500_000, "g1"), leg("B", "LONG", 1_000_000, "g1"), leg("C", "SHORT", 1_000_000, "g2")]
    assert dm.check_limits(ok, nav=NAV, gross_budget_usd=budget) == []  # exactly 15% is allowed; g1 = 25%

    one_name = [leg("A", "LONG", 1_500_100, "g1")]  # $100 over 15%
    assert codes(dm.check_limits(one_name, nav=NAV, gross_budget_usd=budget)) == [("SINGLE_NAME", "A")]

    group = [leg(t, "LONG", 1_200_000, "tech") for t in "ABC"]  # 12% each, 36% together
    b = dm.check_limits(group, nav=NAV, gross_budget_usd=budget)
    assert codes(b) == [("SECTOR_GROUP", "tech")] and b[0]["tickers"] == ["A", "B", "C"]
    assert b[0]["actual_usd"] == 3_600_000 and b[0]["limit_usd"] == 3_500_000
    assert dm.check_limits([leg(t, "LONG", 1_166_600, "tech") for t in "ABC"], nav=NAV, gross_budget_usd=budget) == []  # 34.998%

    spread = [leg(t, "LONG", 1_000_000, f"g{i}") for i, t in enumerate("ABCDEFG")]  # 7M gross, no name or group is large
    b = dm.check_limits(spread, nav=NAV, gross_budget_usd=budget)
    assert codes(b) == [("GROSS", "book")] and len(b[0]["tickers"]) == 7

    forbidden = [leg("A", "SHORT", 500_000, "g1"), leg("B", "LONG", 500_000, "g1")]
    assert codes(dm.check_limits(forbidden, nav=NAV, gross_budget_usd=budget, avoid={"A", "B"})) == [("AVOID_SHORT", "A")]  # a LONG in an AVOID name is not this rule's business


def test_zero_notional_legs_are_ignored_and_details_are_readable():
    legs = [leg("A", "LONG", 2_000_000, "g1"), leg("B", "LONG", 0, "g1")]
    (b,) = dm.check_limits(legs, nav=NAV, gross_budget_usd=9e9)
    assert b["detail"] == "A LONG 2,000,000 is 20.0% of NAV (limit 15%)"
    assert dm.check_limits([leg("A", "SHORT", 0, "g")], nav=NAV, gross_budget_usd=0, avoid={"A"}) == []


def test_cure_scales_by_hand_take_the_minimum_over_breaches():
    # A is 2,000,000 (20%) -> single-name cure 1.5M / 2.0M = 0.75.   A, B, C together are 2.0 + 1.4 + 1.4 = 4.8M in 'tech',
    # limit 3.5M -> group cure 3.5 / 4.8 = 0.72916... -> 0.7291 (rounded DOWN to 4 decimals).  A takes min(0.75, 0.7291).
    legs = [leg("A", "LONG", 2_000_000, "tech"), leg("B", "LONG", 1_400_000, "tech"), leg("C", "LONG", 1_400_000, "tech"), leg("D", "LONG", 900_000, "def")]
    b = dm.check_limits(legs, nav=NAV, gross_budget_usd=9e9)
    assert codes(b) == [("SECTOR_GROUP", "tech"), ("SINGLE_NAME", "A")]
    assert dm.cure_scales(b) == {"A": 0.7291, "B": 0.7291, "C": 0.7291}  # D is in neither breach: untouched
    assert dm.breaching_names(b) == ["A", "B", "C"]
    # an AVOID short is cured only by dropping it, even if it also breaches a size limit
    legs2 = [leg("S", "SHORT", 2_000_000, "x")]
    b2 = dm.check_limits(legs2, nav=NAV, gross_budget_usd=9e9, avoid={"S"})
    assert dm.cure_scales(b2) == {"S": 0.0}


@pytest.mark.parametrize(
    "proposed, required, expected",
    [
        (0.6, 0.75, 0.6),  # deeper than required but within the 2x bound (required cut 25%, this cut 40%): accepted
        (0.5, 0.75, 0.5),  # exactly twice the required cut: accepted
        (0.1, 0.75, 0.5),  # deeper than twice the required cut: bounded at min_scale = 2 * 0.75 - 1
        (0.3, 0.9533, 0.9066),  # a 4.67% trim may become 9.34%, not 70%
        (0.0, 0.4, 0.0),  # a position that needs a >50% cut anyway may be dropped
        (0.9, 0.75, 0.9),  # insufficient: accepted as is (the loop goes round again)
        (1.7, 0.75, 1.0),  # never above 1.0
        (-2, 0.75, 0.5),
        (float("nan"), 0.75, 1.0),  # nonsense means "no cut"
        (None, 0.75, 1.0),
        ("abc", 0.75, 1.0),
        (0.9, 0.0, 0.0),  # a forbidden position is dropped whatever the model says
        (1.0, 0.0, 0.0),
    ],
)
def test_applied_scale_bounds_the_model(proposed, required, expected):
    assert dm.applied_scale(proposed, required) == expected


def test_min_scale_is_twice_the_required_cut():
    assert [dm.min_scale(x) for x in (0.9533, 0.75, 0.5, 0.4, 0.0)] == [0.9066, 0.5, 0.0, 0.0, 0.0]


def test_apply_scale_rounds_down_logs_and_marks_dropped():
    base = leg("A", "LONG", 2_000_000, "g")
    cut = dm.apply_scale(base, 0.7291, nav=NAV, step="resize round 1", status="resized")
    assert cut["notional_usd"] == 1_458_200 and cut["status"] == "resized" and cut["pct_nav"] == 0.14582
    assert cut["history"][-1] == {"step": "resize round 1", "scale": 0.7291, "notional_usd": 1_458_200}
    assert base["notional_usd"] == 2_000_000 and len(base["history"]) == 1  # the input is not mutated
    gone = dm.apply_scale(base, 0.0, nav=NAV, step="x", status="resized")
    assert gone["status"] == "dropped" and gone["notional_usd"] == 0


def test_enforce_limits_trims_only_what_breaches():
    legs = [leg("A", "LONG", 2_000_000, "tech"), leg("B", "LONG", 1_000_000, "x"), leg("S", "SHORT", 400_000, "y")]
    out, trims = dm.enforce_limits(legs, nav=NAV, gross_budget_usd=9e9, avoid={"S"})
    by = {l["ticker"]: l for l in out}
    assert by["A"]["notional_usd"] == 1_500_000 and by["A"]["status"] == "trimmed"
    assert by["S"]["notional_usd"] == 0 and by["S"]["status"] == "dropped"
    assert by["B"] is legs[1]  # untouched legs are the very same objects
    assert [t["ticker"] for t in trims] == ["A", "S"]
    assert trims[0] == {"ticker": "A", "scale": 0.75, "before_usd": 2_000_000, "after_usd": 1_500_000}
    assert dm.check_limits(out, nav=NAV, gross_budget_usd=9e9, avoid={"S"}) == []
    same, none = dm.enforce_limits(out, nav=NAV, gross_budget_usd=9e9, avoid={"S"})
    assert none == [] and same == out


def test_one_trim_always_yields_a_clean_book_fuzz():
    """The termination argument in the example docstring: scaling every breaching name by its cure scale ONCE cures
    everything, never increases a position, and never touches a name that is in no breach."""
    rng = random.Random(20261108)
    groups = ["tech", "cyc", "def"]
    for _ in range(600):
        nav = rng.choice([1e6, 7.3e6, 1e7, 2.5e8, 4e9])
        n = rng.randint(1, 8)
        legs = [
            leg(f"T{i}", rng.choice(["LONG", "SHORT"]), dm.floor_lot(rng.uniform(0, 0.4) * nav), rng.choice(groups))
            for i in range(n)
        ]
        for l in legs:  # the helper divides by the module NAV; recompute for this nav
            l["pct_nav"] = l["notional_usd"] / nav
        avoid = {l["ticker"] for l in legs if rng.random() < 0.25}
        budget = dm.floor_lot(rng.uniform(0.2, 0.9) * nav)
        before = dm.check_limits(legs, nav=nav, gross_budget_usd=budget, avoid=avoid)
        out, trims = dm.enforce_limits(legs, nav=nav, gross_budget_usd=budget, avoid=avoid)
        assert dm.check_limits(out, nav=nav, gross_budget_usd=budget, avoid=avoid) == []
        by_before = {l["ticker"]: l for l in legs}
        for l in out:
            assert l["notional_usd"] <= by_before[l["ticker"]]["notional_usd"]
            if l["ticker"] not in dm.breaching_names(before):
                assert l is by_before[l["ticker"]]
        assert {t["ticker"] for t in trims} == set(dm.breaching_names(before))


def test_group_exposure_counts_only_live_legs():
    legs = [leg("A", "LONG", 1_000_000, "tech"), leg("B", "SHORT", 500_000, "tech"), leg("C", "LONG", 0, "def")]
    assert dm.group_exposure(legs, NAV) == {"tech": 0.15}


# --------------------------------------------------------------------------- execution
def test_plan_execution_by_hand():
    # adv 1.3M shares, 5% participation over 13 slots: max child = floor(0.05 * 1,300,000 / 13) = 5000 shares
    # 20,000 shares -> 4 slices of 5,000, placed at slots (i * 13) // 4 = 0, 3, 6, 9 -> 09:30, 11:00, 12:30, 14:00
    # vol 2%: half spread = 1 + 50 * 0.02 = 2.0 bps; impact = 10000 * 0.5 * 0.02 * sqrt(20000 / 1.3M) = 100 * 0.124035 = 12.4035 bps
    p = dm.plan_execution(20_000, 100.0, 1_300_000, 0.02)
    assert p["max_child_shares"] == 5000 and p["n_slices"] == 4 and p["sessions"] == 1
    assert [(o["start"], o["shares"]) for o in p["child_orders"]] == [("09:30", 5000), ("11:00", 5000), ("12:30", 5000), ("14:00", 5000)]
    assert p["participation_of_adv"] == pytest.approx(20_000 / 1_300_000, abs=1e-6)
    assert p["half_spread_bps"] == 2.0 and p["impact_bps"] == pytest.approx(12.4035, abs=1e-3)
    assert p["est_cost_bps"] == pytest.approx(14.4035, abs=2e-3)
    assert p["est_cost_usd"] == pytest.approx(20_000 * 100 * 14.4035 / 1e4, abs=0.5)


def test_plan_execution_remainders_and_sessions():
    # adv 780 -> max child floor(0.05 * 780 / 13) = 3.  10 shares -> 4 slices, remainder spread: 3, 3, 2, 2
    p = dm.plan_execution(10, 50.0, 780, 0.01)
    assert [o["shares"] for o in p["child_orders"]] == [3, 3, 2, 2] and sum(o["shares"] for o in p["child_orders"]) == 10
    # 100 shares -> 34 slices: more than the 13 slots of a session, so it spills into 3 sessions
    big = dm.plan_execution(100, 50.0, 780, 0.01)
    assert big["n_slices"] == 34 and big["sessions"] == 3
    assert sum(o["shares"] for o in big["child_orders"]) == 100 and max(o["shares"] for o in big["child_orders"]) <= 3
    assert [o["session"] for o in big["child_orders"]][:14] == [1] * 13 + [2]
    # tiny order: one slice, never zero
    one = dm.plan_execution(7, 10.0, 1_000_000, 0.02)
    assert one["n_slices"] == 1 and one["child_orders"][0] == {"slice": 1, "session": 1, "start": "09:30", "shares": 7}
    nothing = dm.plan_execution(0, 10.0, 1_000_000, 0.02)
    assert nothing["child_orders"] == [] and nothing["est_cost_bps"] == 0.0


def test_cost_grows_with_size_and_volatility():
    small = dm.plan_execution(1_000, 100.0, 1_300_000, 0.02)["est_cost_bps"]
    large = dm.plan_execution(100_000, 100.0, 1_300_000, 0.02)["est_cost_bps"]
    wild = dm.plan_execution(100_000, 100.0, 1_300_000, 0.04)["est_cost_bps"]
    assert small < large < wild


# --------------------------------------------------------------------------- note audit
def test_numbers_in_parses_formats():
    assert dm.numbers_in("Gross 1,234,500 USD (12.5% of NAV), 7 positions, 3.0 bps; -4") == {1234500.0, 12.5, 7.0, 3.0, 4.0}
    assert dm.numbers_in("ends with a comma 1,500, and 0.50") == {1500.0, 0.5}
    assert dm.numbers_in("no digits here") == set()


def test_unsupported_numbers():
    facts = "GROSS: 1,234,500 USD (12.5% of NAV)\nPOSITIONS: 3\nestimated cost 8.80 bps"
    assert dm.unsupported_numbers("Gross is 1,234,500 USD, 12.5% of NAV, across 3 positions at 8.8 bps.", facts) == []
    assert dm.unsupported_numbers("Gross is $1.2M.", facts) == ["1.2"]
    assert dm.unsupported_numbers("Gross is 1234500 and 13% of NAV with 4 positions.", facts) == ["13", "4"]  # 1234500 == 1,234,500
    assert dm.unsupported_numbers("", facts) == []

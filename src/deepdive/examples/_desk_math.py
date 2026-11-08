"""Deterministic math behind ``portfolio_desk_lite`` (private helper, not an example).

Everything here is a pure function of its arguments: no I/O, no model calls, no LangGraph imports. That is the point.
The example lets models *judge* (which names to look at, bull and bear arguments, how deep to cut a position) and
lets this module *decide and verify* (analytics, position sizes, limit checks, child-order slicing, whether a model
written note quotes only numbers that really are in the blotter). Each function is unit-tested against hand-computed
values in ``tests/test__desk_math.py``.

The market is FICTIONAL, the data is synthetic, and nothing here is investment advice.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- policy constants
MAX_NAME_PCT = 0.15  # one position may not exceed 15% of NAV
MAX_GROUP_PCT = 0.35  # one sector group may not exceed 35% of NAV (gross)
SETUP_GATE = 0.40  # P(setup_supportive) below this turns LONG/SHORT into AVOID
TAIL_TRIGGER = 0.70  # P(tail_risk_elevated) at/above this makes the tilt one notch more defensive
RIGOR_MARGIN = 0.25  # expected-rigor gap needed to call a debate winner
VOL_FLOOR = 0.005  # daily vol floor so a calm name cannot get an absurd inverse-vol weight
LOT = 100.0  # notionals are rounded DOWN to whole hundreds of dollars
EPS = 0.005  # half a cent: tolerance when comparing dollar amounts to limits
MIN_HISTORY = 60  # closes needed for every indicator window
BUCKETS = 13  # half-hour execution slots in a 09:30-16:00 session
PARTICIPATION_CAP = 0.05  # at most 5% of the expected volume of a slot per child order
IMPACT_COEF = 0.5  # square-root impact coefficient (order-of-magnitude, NOT calibrated)
DEFAULT_BUDGET = 0.45
BUDGET_RANGE = (0.25, 0.70)

DIRECTIONS = ("LONG", "SHORT", "AVOID")
TILT_ORDER = ("NET_LONG", "NEUTRAL", "DEFENSIVE")  # most to least risk-seeking
TILTS: dict[str, dict[str, float]] = {
    # gross_scale multiplies the CIO's gross budget; long_mult / short_mult re-weight the two sides before the
    # weights are normalised to that gross.
    "NET_LONG": {"gross_scale": 1.0, "long_mult": 1.0, "short_mult": 0.6},
    "NEUTRAL": {"gross_scale": 0.8, "long_mult": 1.0, "short_mult": 1.0},
    "DEFENSIVE": {"gross_scale": 0.6, "long_mult": 0.6, "short_mult": 1.0},
}

# companies.csv has one company per sector, so a literal "one sector" cap could never bind. The cap is applied to
# these coarser sector groups instead (an unknown sector becomes its own group).
SECTOR_GROUPS = {
    "industrial tech": "technology",
    "semiconductors": "technology",
    "software": "technology",
    "energy": "cyclicals",
    "renewables": "cyclicals",
    "financials": "cyclicals",
    "healthcare": "defensives",
    "consumer staples": "defensives",
}


def sector_group(sector: str) -> str:
    s = str(sector).strip().lower()
    return SECTOR_GROUPS.get(s, s)


def floor_lot(x: float) -> float:
    """Round a dollar amount DOWN to a whole LOT so rounding can never push a book over a limit."""
    return float(math.floor(x / LOT + 1e-9) * LOT)


def _num(x: Any, nd: int = 6) -> float:
    return round(float(x), nd)


# --------------------------------------------------------------------------- analytics
def rsi_wilder(close: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI: seed with the simple average of the first ``period`` changes, then smooth recursively."""
    if len(close) < period + 1:
        raise ValueError(f"need at least {period + 1} closes for RSI({period})")
    delta = np.diff(close.astype(float))
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_gain, avg_loss = gain[:period].mean(), loss[:period].mean()
    for g, l in zip(gain[period:], loss[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def compute_analytics(prices: pd.DataFrame) -> dict[str, Any]:
    """Indicators for ONE ticker from its OHLCV rows. Returns plain floats (fractions, not percent).

    ret_5d / ret_20d   close[-1] / close[-6 or -21] - 1
    vol_20d            sample standard deviation (ddof=1) of the last 20 daily simple returns (a daily figure)
    rsi_14             Wilder RSI over the whole history
    drawdown_60d       close[-1] / max(close[-60:]) - 1  (zero or negative; closes, not intraday highs)
    zscore_50d         (close[-1] - mean(close[-50:])) / std(close[-50:], ddof=1)
    adv_20d_shares     mean(volume[-20:])
    """
    df = prices.sort_values("date")
    if len(df) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} trading days of prices, got {len(df)}")
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    daily = close[1:] / close[:-1] - 1.0
    vol = float(np.std(daily[-20:], ddof=1))
    last = float(close[-1])
    adv = float(volume[-20:].mean())
    window50 = close[-50:]
    return {
        "asof": str(df["date"].iloc[-1]),
        "last_close": _num(last, 4),
        "ret_5d": _num(last / close[-6] - 1.0),
        "ret_20d": _num(last / close[-21] - 1.0),
        "vol_20d": _num(vol),
        "vol_20d_ann": _num(vol * math.sqrt(252)),
        "rsi_14": _num(rsi_wilder(close), 3),
        "drawdown_60d": _num(last / float(close[-60:].max()) - 1.0),
        "zscore_50d": _num((last - window50.mean()) / np.std(window50, ddof=1), 4),
        "adv_20d_shares": _num(adv, 1),
        "adv_20d_usd": _num(adv * last, 0),
    }


def recent_headlines(news: pd.DataFrame, asof: str, *, days: int = 30, k: int = 6) -> list[dict[str, str]]:
    """The ``k`` newest headlines within ``days`` calendar days up to and including ``asof`` (labels are NOT passed on)."""
    end = pd.Timestamp(asof)
    d = pd.to_datetime(news["date"])
    rows = news[(d <= end) & (d >= end - pd.Timedelta(days=days))].copy()
    rows["_d"] = pd.to_datetime(rows["date"])
    rows = rows.sort_values(["_d", "headline_id"], ascending=[False, True]).head(k)
    return [{"date": str(r.date), "headline": str(r.headline)} for r in rows.itertuples()]


def pct(x: float, signed: bool = True) -> str:
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def fact_sheet(meta: dict[str, Any], an: dict[str, Any], headlines: list[dict[str, str]]) -> str:
    """Plain-text evidence for one ticker (what the Decisions API and the debaters read)."""
    heads = "\n".join(f"- {h['date']}  {h['headline']}" for h in headlines) or "- (none in the last 30 days)"
    return (
        f"TICKER: {meta['ticker']}\n"
        f"COMPANY: {meta['company']} (sector {meta['sector']}, group {meta['group']}, beta {meta['beta']})\n"
        f"AS OF: {an['asof']}  (fictional market, synthetic data)\n"
        f"LAST CLOSE: {an['last_close']:.2f}\n"
        f"RETURN 5D: {pct(an['ret_5d'])}   RETURN 20D: {pct(an['ret_20d'])}\n"
        f"REALISED VOL 20D: {pct(an['vol_20d'], False)} per day ({pct(an['vol_20d_ann'], False)} annualised)\n"
        f"RSI(14): {an['rsi_14']:.1f}  (above 70 is overbought, below 30 is oversold)\n"
        f"DRAWDOWN FROM 60D HIGH: {pct(an['drawdown_60d'])}\n"
        f"Z-SCORE OF PRICE VS 50D MEAN: {an['zscore_50d']:+.2f}\n"
        f"AVERAGE DAILY VOLUME 20D: {an['adv_20d_shares']:,.0f} shares\n"
        f"HEADLINES (last 30 days, newest first):\n{heads}"
    )


# --------------------------------------------------------------------------- CIO validation and fallback
def validate_focus(raw: list[Any], universe: set[str], *, maximum: int = 5) -> tuple[list[str], list[str]]:
    """Normalise model-proposed tickers: upper-case, de-duplicate, keep only known ones. Returns (valid, dropped)."""
    valid: list[str] = []
    dropped: list[str] = []
    for item in raw or []:
        t = str(item).strip().upper()
        if t in universe and t not in valid:
            valid.append(t)
        else:
            dropped.append(str(item))
    return valid[:maximum], dropped


def clamp_budget(x: Any) -> float:
    """Gross budget as a fraction of NAV in BUDGET_RANGE. 45 is read as 45%; junk becomes DEFAULT_BUDGET."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return DEFAULT_BUDGET
    if not math.isfinite(v):
        return DEFAULT_BUDGET
    if v > 1.5:
        v /= 100.0
    return round(min(max(v, BUDGET_RANGE[0]), BUDGET_RANGE[1]), 4)


def fallback_focus(hint: str, universe: list[dict[str, Any]], *, minimum: int = 4, maximum: int = 5) -> list[str]:
    """Deterministic focus list: names the hint mentions (ticker, company, sector or group), biggest 20-day movers first,
    then topped up with the biggest 20-day movers until there are ``minimum`` names."""
    text = (hint or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    movers = sorted(universe, key=lambda u: (-abs(u["ret_20d"]), u["ticker"]))

    def mentioned(u: dict[str, Any]) -> bool:
        return bool(text) and (
            u["ticker"].lower() in tokens
            or u["company"].split()[0].lower() in tokens
            or u["company"].lower() in text
            or u["sector"].lower() in text
            or u["group"].lower() in text
        )

    chosen = [u["ticker"] for u in movers if mentioned(u)][:maximum]
    for u in movers:
        if len(chosen) >= minimum:
            break
        if u["ticker"] not in chosen:
            chosen.append(u["ticker"])
    return chosen


# --------------------------------------------------------------------------- Decisions interpretation
def resolve_direction(pick: str | None, p_setup: float) -> tuple[str, bool]:
    """Final direction and whether the setup gate overrode the model's pick.

    A refused/unknown pick fails safe to AVOID; a LONG/SHORT pick needs P(setup_supportive) >= SETUP_GATE."""
    if pick not in DIRECTIONS:
        return "AVOID", False
    if pick != "AVOID" and p_setup < SETUP_GATE:
        return "AVOID", True
    return pick, False


def expected_level(probabilities: dict[str, float], levels: list[str], fallback: float = 0.0) -> float:
    """Expected index of an ordered score (0 = lowest level) computed from the label probabilities, so it does not
    depend on how the API scales its own ``score`` field."""
    total = sum(max(probabilities.get(lv, 0.0), 0.0) for lv in levels)
    if total <= 0:
        return fallback
    return sum(i * max(probabilities.get(lv, 0.0), 0.0) for i, lv in enumerate(levels)) / total


def conviction_mult(expected: float) -> float:
    """Expected conviction in [0, 2] -> sizing multiplier in [0.5, 1.5]."""
    return 0.5 + 0.5 * min(max(expected, 0.0), 2.0)


def apply_tail_override(tilt: str, p_tail: float) -> tuple[str, bool]:
    """A high tail-risk probability moves the tilt ONE notch toward DEFENSIVE (never the other way)."""
    if p_tail >= TAIL_TRIGGER and tilt in TILT_ORDER and tilt != TILT_ORDER[-1]:
        return TILT_ORDER[TILT_ORDER.index(tilt) + 1], True
    return tilt, False


def debate_winner(bull_score: float, bear_score: float) -> str:
    gap = bull_score - bear_score
    return "bull" if gap > RIGOR_MARGIN else "bear" if gap < -RIGOR_MARGIN else "draw"


# --------------------------------------------------------------------------- sizing
def size_book(candidates: list[dict[str, Any]], tilt: str, gross_usd: float, nav: float) -> list[dict[str, Any]]:
    """Proposed legs from actionable dossiers.

    weight = conviction_mult * side_mult(tilt) / max(vol_20d, VOL_FLOOR)   (inverse-volatility, conviction-weighted)
    notional = floor_lot(gross_usd * weight / sum(weights))

    Deliberately NOT limit-aware: the desk sizes to the gross budget and the risk committee is the independent check.
    """
    t = TILTS[tilt]
    raw: dict[str, float] = {}
    basis: dict[str, dict[str, float]] = {}
    for c in candidates:
        side_mult = t["long_mult"] if c["direction"] == "LONG" else t["short_mult"]
        cm = conviction_mult(c["conviction_score"])
        vol = max(c["analytics"]["vol_20d"], VOL_FLOOR)
        raw[c["ticker"]] = cm * side_mult / vol
        basis[c["ticker"]] = {"conviction_mult": round(cm, 4), "side_mult": side_mult, "vol_20d": round(vol, 6)}
    total = sum(raw.values())
    legs = []
    for c in sorted(candidates, key=lambda c: c["ticker"]):
        share = raw[c["ticker"]] / total
        notional = floor_lot(gross_usd * share)
        legs.append(
            {
                "ticker": c["ticker"],
                "side": c["direction"],
                "notional_usd": notional,
                "pct_nav": round(notional / nav, 6),
                "group": c["group"],
                "last_price": c["analytics"]["last_close"],
                "status": "proposed",
                "basis": {**basis[c["ticker"]], "share_of_gross": round(share, 6)},
                "history": [{"step": "desk", "scale": 1.0, "notional_usd": notional}],
            }
        )
    return legs


def apply_scale(leg: dict[str, Any], scale: float, *, nav: float, step: str, status: str) -> dict[str, Any]:
    """A copy of ``leg`` with its notional multiplied by ``scale`` (rounded DOWN to a lot) and the change logged."""
    new = floor_lot(leg["notional_usd"] * scale)
    return {
        **leg,
        "notional_usd": new,
        "pct_nav": round(new / nav, 6),
        "status": "dropped" if new <= 0 else status,
        "history": [*leg["history"], {"step": step, "scale": round(scale, 4), "notional_usd": new}],
    }


# --------------------------------------------------------------------------- risk engine
def group_exposure(legs: list[dict[str, Any]], nav: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for leg in legs:
        if leg["notional_usd"] > 0:
            out[leg["group"]] = out.get(leg["group"], 0.0) + leg["notional_usd"]
    return {g: round(v / nav, 6) for g, v in sorted(out.items())}


def check_limits(
    legs: list[dict[str, Any]], *, nav: float, gross_budget_usd: float, avoid: set[str] | frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """Every limit breach in the book, as records ``{code, scope, tickers, actual_usd, limit_usd, detail}``.

    SINGLE_NAME   one position above MAX_NAME_PCT of NAV
    SECTOR_GROUP  gross of one sector group above MAX_GROUP_PCT of NAV
    GROSS         total gross above the (tilt-scaled) budget
    AVOID_SHORT   any SHORT in a name whose dossier says AVOID
    """
    live = [leg for leg in legs if leg["notional_usd"] > 0]
    breaches: list[dict[str, Any]] = []
    name_cap = MAX_NAME_PCT * nav
    for leg in sorted(live, key=lambda leg: leg["ticker"]):
        if leg["notional_usd"] > name_cap + EPS:
            breaches.append(
                {
                    "code": "SINGLE_NAME",
                    "scope": leg["ticker"],
                    "tickers": [leg["ticker"]],
                    "actual_usd": leg["notional_usd"],
                    "limit_usd": round(name_cap, 2),
                    "detail": f"{leg['ticker']} {leg['side']} {leg['notional_usd']:,.0f} is "
                    f"{leg['notional_usd'] / nav:.1%} of NAV (limit {MAX_NAME_PCT:.0%})",
                }
            )
    group_cap = MAX_GROUP_PCT * nav
    totals: dict[str, float] = {}
    for leg in live:
        totals[leg["group"]] = totals.get(leg["group"], 0.0) + leg["notional_usd"]
    for group, total in sorted(totals.items()):
        if total > group_cap + EPS:
            members = sorted(leg["ticker"] for leg in live if leg["group"] == group)
            breaches.append(
                {
                    "code": "SECTOR_GROUP",
                    "scope": group,
                    "tickers": members,
                    "actual_usd": total,
                    "limit_usd": round(group_cap, 2),
                    "detail": f"group '{group}' ({', '.join(members)}) gross {total:,.0f} is "
                    f"{total / nav:.1%} of NAV (limit {MAX_GROUP_PCT:.0%})",
                }
            )
    gross = sum(leg["notional_usd"] for leg in live)
    if gross > gross_budget_usd + EPS:
        breaches.append(
            {
                "code": "GROSS",
                "scope": "book",
                "tickers": sorted(leg["ticker"] for leg in live),
                "actual_usd": gross,
                "limit_usd": round(gross_budget_usd, 2),
                "detail": f"book gross {gross:,.0f} exceeds the budget {gross_budget_usd:,.0f}",
            }
        )
    for leg in sorted(live, key=lambda leg: leg["ticker"]):
        if leg["side"] == "SHORT" and leg["ticker"] in avoid:
            breaches.append(
                {
                    "code": "AVOID_SHORT",
                    "scope": leg["ticker"],
                    "tickers": [leg["ticker"]],
                    "actual_usd": leg["notional_usd"],
                    "limit_usd": 0.0,
                    "detail": f"{leg['ticker']} is SHORT {leg['notional_usd']:,.0f} but research marked it AVOID",
                }
            )
    return breaches


def breaching_names(breaches: list[dict[str, Any]]) -> list[str]:
    return sorted({t for b in breaches for t in b["tickers"]})


def _floor4(x: float) -> float:
    return math.floor(x * 1e4 + 1e-9) / 1e4


def cure_scales(breaches: list[dict[str, Any]]) -> dict[str, float]:
    """For each breaching ticker, the LARGEST scale (<= 1, 4 decimals, rounded down) that cures every breach it is in.

    A group or gross breach implicates every live leg in it, so each is scaled by limit/actual; an AVOID short is
    cured only by dropping it (0.0). Because each leg takes the minimum over its breaches, applying these scales once
    always yields a clean book (proof sketch in the example docstring; fuzz-tested)."""
    out: dict[str, float] = {}
    for b in breaches:
        s = 0.0 if b["code"] == "AVOID_SHORT" else _floor4(min(1.0, b["limit_usd"] / b["actual_usd"]))
        for t in b["tickers"]:
            out[t] = min(out.get(t, 1.0), s)
    return dict(sorted(out.items()))


def min_scale(required: float) -> float:
    """The deepest cut the desk accepts for a position whose cure scale is ``required``: at most TWICE the cut the
    limit demands, i.e. ``1 - scale <= 2 * (1 - required)``. A 5% trim may become 10%, never 50%."""
    return round(max(0.0, 2.0 * required - 1.0), 4)


def applied_scale(proposed: Any, required: float) -> float:
    """The scale actually applied to a leg: the model's proposal, bounded by code.

    * required == 0 (a forbidden position): always 0, whatever the model says
    * the proposal may be insufficient (the loop will go round again) but never above 1.0
    * the proposal may cut deeper than required, but not deeper than ``min_scale(required)`` (a confused model must
      not liquidate a position that only needs a trim)
    * NaN / missing proposals mean "no cut" (1.0)"""
    if required <= 0:
        return 0.0
    try:
        p = float(proposed)
    except (TypeError, ValueError):
        p = 1.0
    if not math.isfinite(p):
        p = 1.0
    return round(min(1.0, max(p, min_scale(required))), 4)


def enforce_limits(
    legs: list[dict[str, Any]], *, nav: float, gross_budget_usd: float, avoid: set[str] | frozenset[str] = frozenset()
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic trim used when the revision cap is reached: scale every breaching leg by its cure scale.

    Returns (all legs after the trim, one record per leg that changed)."""
    breaches = check_limits(legs, nav=nav, gross_budget_usd=gross_budget_usd, avoid=avoid)
    cures = cure_scales(breaches)
    out: list[dict[str, Any]] = []
    trims: list[dict[str, Any]] = []
    for leg in legs:
        s = cures.get(leg["ticker"])
        if s is None:
            out.append(leg)
            continue
        new = apply_scale(leg, s, nav=nav, step="cap trim", status="trimmed")
        out.append(new)
        trims.append(
            {"ticker": leg["ticker"], "scale": round(s, 4), "before_usd": leg["notional_usd"], "after_usd": new["notional_usd"]}
        )
    return out, trims


# --------------------------------------------------------------------------- execution
def plan_execution(shares: int, price: float, adv: float, vol_daily: float) -> dict[str, Any]:
    """Child-order schedule and a cost estimate for ``shares`` of one name.

    max_child_shares  floor(PARTICIPATION_CAP * adv / BUCKETS): the most one child order may be, so it stays under
                      PARTICIPATION_CAP of the expected volume of its half-hour slot
    n_slices          ceil(shares / max_child_shares), at least 1; more than BUCKETS slices spill into later sessions
    sizes             an even split (the first ``shares % n`` slices carry one extra share)
    cost (bps)        half spread (1 + 50 * vol) + square-root impact (10000 * IMPACT_COEF * vol * sqrt(shares / adv))
    """
    if shares <= 0:
        return {"shares": 0, "n_slices": 0, "sessions": 0, "child_orders": [], "est_cost_bps": 0.0, "est_cost_usd": 0.0}
    max_child = max(1, int(math.floor(PARTICIPATION_CAP * adv / BUCKETS)))
    n = max(1, math.ceil(shares / max_child))
    base, extra = divmod(shares, n)
    orders = []
    for i in range(n):
        slot = (i * BUCKETS) // n if n <= BUCKETS else i % BUCKETS
        minutes = 9 * 60 + 30 + 30 * slot
        orders.append(
            {
                "slice": i + 1,
                "session": i // BUCKETS + 1,
                "start": f"{minutes // 60:02d}:{minutes % 60:02d}",
                "shares": base + (1 if i < extra else 0),
            }
        )
    participation = shares / adv
    half_spread = 1.0 + 50.0 * vol_daily
    impact = 10_000.0 * IMPACT_COEF * vol_daily * math.sqrt(participation)
    bps = half_spread + impact
    return {
        "shares": shares,
        "participation_of_adv": round(participation, 6),
        "max_child_shares": max_child,
        "n_slices": n,
        "sessions": math.ceil(n / BUCKETS),
        "child_orders": orders,
        "half_spread_bps": round(half_spread, 3),
        "impact_bps": round(impact, 3),
        "est_cost_bps": round(bps, 3),
        "est_cost_usd": round(shares * price * bps / 10_000.0, 2),
    }


# --------------------------------------------------------------------------- note audit
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every numeric token in ``text`` as a float ('1,234.50' -> 1234.5; signs and units are ignored)."""
    out: set[float] = set()
    for tok in _NUMBER.findall(text or ""):
        tok = tok.strip(",")
        if tok:
            out.add(round(float(tok.replace(",", "")), 6))
    return out


def unsupported_numbers(note: str, facts: str) -> list[str]:
    """Numbers quoted in ``note`` that do not appear in ``facts`` (empty list = the note only quotes the blotter)."""
    allowed = numbers_in(facts)
    bad = []
    for tok in _NUMBER.findall(note or ""):
        tok = tok.strip(",")
        if tok and round(float(tok.replace(",", "")), 6) not in allowed:
            bad.append(tok)
    return bad

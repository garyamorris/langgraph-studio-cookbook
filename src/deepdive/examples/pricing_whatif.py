"""Pricing what-if with time travel: fork a saved analysis at its assumptions and recompute only what depends on them.

PROBLEM
    A pricing manager reads the analysis for a product and asks "what if we had targeted a 35% margin instead of
    28%?". Then "and if demand is less price-sensitive than you estimated?". Re-running the whole analysis for each
    question repeats the slow, costly part - loading and aggregating sales, fitting a demand model with a bootstrap,
    and an LLM review of the fit - to change one number that only the last two steps use. We want to FORK the saved
    run at the point where the assumptions were fixed, edit them, and recompute only the steps downstream, keeping
    the original run intact for comparison.

DESIGN
    The graph is cut so that the question "what depends on what" is visible in the topology:

      * ``load_history`` (data) and ``estimate_elasticity`` (numpy) are deterministic and upstream of everything.
        Elasticity is the slope of a log-log regression of weekly units on the weekly average selling price, with
        a seeded bootstrap (500 resamples of weeks, 5th-95th percentile) for an interval, so the same data always
        gives the same numbers. ``review_estimate`` (LLM) turns the fit into three plain sentences about how far to
        trust it. This is the expensive, reusable part.
      * ``set_assumptions`` is deliberately its own node and is THE fork point (a node cannot share a name with a
        state key, so the node is ``set_assumptions`` and the dict it writes is ``assumptions``). It holds
        every number the rest of the graph is allowed to use: unit cost from the data, variable cost rate, margin
        target, elasticity (and its interval), baseline price and weekly units, horizon, candidate prices.
        Everything downstream reads ONLY this dict (``recommend`` takes the margin target from it, not from the
        input), so editing it is a complete, honest what-if. Edit ``margin_target``, ``elasticity`` or
        ``variable_cost_rate``; do not edit values the data produced unless you mean to override them.
      * ``simulate`` (code) projects a 13-week horizon for each candidate price with a constant-elasticity demand
        curve: expected units, revenue, contribution margin, margin %, a units interval from the elasticity
        interval, and flags for "meets the margin target" and "inside the price range the data supports".
      * ``recommend`` picks the winner in CODE: the candidate that maximises total contribution margin among those
        that meet the margin target and are within 10% of the weekly prices actually observed. The model only
        writes the paragraph, and a check requires the paragraph to quote the chosen price, margin and contribution
        exactly as the table has them; if it does not, a fixed template is used instead, so no number in the output
        comes from a language model.

    Margin here is CONTRIBUTION margin: (price x (1 - variable_cost_rate) - unit_cost) / price. The dataset has unit
    cost but not payment fees, shipping or returns, so ``variable_cost_rate`` (default 22%) is an assumption that
    finance should own; it is why the default 28% target roughly equals today's margin for Lumen Therm and 35% is a
    real change. Because demand here is more elastic than -1 and the margin target is a FLOOR, profit keeps rising as
    the price falls until the floor binds, so the answer is "the lowest candidate price that still meets the floor";
    the interesting output is how far that price moves and what it does to units.

    WHAT TIME TRAVEL GIVES YOU HERE. Every super-step is a checkpoint. ``get_state_history`` lists them; each has a
    ``next`` field. The checkpoint whose ``next`` is ``("simulate",)`` is the state right after ``set_assumptions``.
    ``graph.update_state(that_checkpoint, {"assumptions": edited})`` writes a NEW checkpoint (a child of that one)
    and returns its config; ``graph.invoke(None, fork_config)`` then runs only the nodes after it. Nothing is
    overwritten: checkpoints are immutable, the original final checkpoint stays reachable by its
    ``checkpoint_id``, and the thread's "latest" now follows the fork. ``scripts/time_travel_demo.py`` does exactly
    this and prints both outcomes side by side.

GRAPH
    START -> load_history -> estimate_elasticity -> review_estimate -> set_assumptions -> simulate -> recommend -> END
             (data)          (numpy, seeded)         (LLM, 1 call)       (code)             (code)     (code + 1 LLM call)
                                                                     writes `assumptions`
                                                                     = THE FORK POINT
                                                                     (checkpoint with next=("simulate",))

LANGGRAPH FEATURES
    * checkpoints and ``get_state_history``: the run is a list of states you can inspect
    * ``update_state`` on a past checkpoint = forking; ``invoke(None, fork_config)`` = continuing from it
    * replay semantics: nodes before the chosen checkpoint are NOT re-run (the tests count executions)
    * branching histories in one thread, original branch preserved
    * ``input_schema`` / ``output_schema`` so Studio shows a clean form and result

MODELS AND DATA
    OpenAI chat twice (``review_estimate`` and the paragraph in ``recommend``); everything else is numpy/pandas on
    ``data/sales.csv`` (cancelled orders excluded). Honest limits: Lumen Therm has 44 weeks with sales and 1 to 11
    units per week, so the elasticity is noisy (about -4.8 with a 90% interval from roughly -8 to -2, R-squared about
    0.17). Weeks with no sales are left out of the fit, a log-log fit on small counts is crude (a Poisson model would be
    better), prices move together with seasonality, and a constant elasticity extrapolated 20% from observed prices
    is a guess. The recommendation says so when it leaves the observed range. This is a demonstration of forking,
    not a pricing engine.

TRY IT IN STUDIO
    1. Pick a case from the "Ready-made case" dropdown and press Submit (on a new thread a blank form runs the first
       one). Each case is a product from ``sales.csv`` and, for some, a margin floor; the recommendation is chosen in
       code from seeded numbers, so these results repeat exactly:
         * Lumen Therm, default 28% floor: about $145 (its list price is now $149).
         * Lumen Therm, 35% floor: $170 with an extrapolation warning, because no week ever averaged above $159.
         * Lumen Therm, 45% floor: "no feasible price".
         * Lumen Cam Pro, default 28% floor: $80.
         * Lumen Sensor Kit, default floor: price barely moves its demand (elasticity -0.24, R-squared 0.00), so the
           analysis pushes the price up to $60, an extrapolation: a fit with no signal gives a weak recommendation.
       Read ``assumptions`` and the ``recommendation`` of the run.
    2. Open the thread's history, select the ``set_assumptions`` step (the last checkpoint before ``simulate``),
       edit the ``assumptions`` value in the state panel - set ``"margin_target": 0.35`` - and fork / re-run from that
       step. (Button names differ a little between Studio versions; what matters is that you edit state at that
       checkpoint and continue from it.) Only ``simulate`` and ``recommend`` run again; for Lumen Therm the new
       recommendation is about $170 and carries an extrapolation warning, as in the second case above.
    3. Select the original run's final step: it is unchanged. Try ``"elasticity": -3`` for a less price-sensitive
       market, or ``"margin_target": 0.45`` to see the "no feasible price" answer.
    4. By hand: ``{"product": "Lumen Therm", "margin_target": 0.28}`` (both fields are optional), or any of the eight
       products. Same thing in code: ``.venv/bin/python scripts/time_travel_demo.py`` (add ``--margin 0.30`` for
       another what-if).

EXTEND IT
    Fork twice from the same checkpoint (two scenarios) and diff them. Add a second fork point after
    ``estimate_elasticity`` to let the manager override the demand model. Replace the single-product log-log fit with
    a Decisions-API-graded set of price-sensitivity cues from reviews, kept upstream of ``assumptions``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import read_csv
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="pricing_whatif",
    title="Pricing what-if: fork a saved run with time travel",
    category="Human-in-the-loop & persistence",
    problem="Let a pricing manager change one assumption in a finished analysis and recompute only the dependent "
    "steps by forking the saved run at its assumptions checkpoint, without losing the original.",
    features=[
        "checkpoints",
        "get_state_history",
        "update_state forking",
        "replay without re-running earlier nodes",
        "branching histories",
    ],
    models=["OpenAI chat"],
    datasets=["sales.csv"],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"product": "Lumen Therm"}

# Ready-made cases for Studio's dropdown (see deepdive.common.scenarios). Each is a product of data/sales.csv and, optionally, a
# margin floor (``None`` = the default, 28%). The first (EXAMPLE_INPUT) runs on a blank form. The recommended price, the "no
# feasible price" answer and the extrapolation warning are computed in code from a seeded fit, so the labels may promise them
# (tests/test_pricing_whatif.py re-checks every number in a label); only the paragraph that explains them is written by a model.
CASES: dict[str, dict] = {
    "Lumen Therm, default 28% margin floor -> recommends $145 (its list price is now $149)": {
        "product": "Lumen Therm", "margin_target": None},
    "Lumen Therm, 35% margin floor -> recommends $170, an extrapolation (no week ever averaged above $159)": {
        "product": "Lumen Therm", "margin_target": 0.35},
    "Lumen Therm, 45% margin floor -> no feasible price: nothing in the supported range meets it": {
        "product": "Lumen Therm", "margin_target": 0.45},
    "Lumen Cam Pro (unit cost $38, list price $89), default 28% margin floor -> recommends $80": {
        "product": "Lumen Cam Pro", "margin_target": None},
    "Lumen Sensor Kit, price barely moves demand (elasticity -0.24, R-squared 0.00) -> $60, an extrapolation": {
        "product": "Lumen Sensor Kit", "margin_target": None},
}
SCENARIO = scenario_field(CASES)

# --------------------------------------------------------------------------- constants
DEFAULT_PRODUCT = "Lumen Therm"
DEFAULT_MARGIN_TARGET = 0.28
DEFAULT_VARIABLE_COST_RATE = 0.22  # ASSUMPTION, not in the dataset: payment fees + shipping + returns, as a share of price
MIN_WEEKS = 12  # fewer weeks with sales than this and we refuse to fit a demand curve
BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 7
BOOTSTRAP_INTERVAL = (5, 95)
BASELINE_WEEKS = 26  # baseline demand = average weekly units over the last 26 calendar weeks of data
HORIZON_WEEKS = 13
PRICE_BAND = (0.85, 1.25)  # candidate prices span 85% to 125% of today's average selling price ...
PRICE_STEP = 5  # ... in whole-dollar steps of $5
SUPPORT_MARGIN = 0.10  # candidates more than 10% beyond the observed weekly price range are not recommendable
ELASTICITY_CEILING = -0.2  # a fitted elasticity above this (wrong sign / no signal) is clamped to it, with a note


# --------------------------------------------------------------------------- state
class PriceIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made case; replaces the two fields below
    product: str  # default "Lumen Therm"
    margin_target: float  # contribution-margin floor, default 0.28


class PriceOut(TypedDict, total=False):
    product: str
    elasticity: dict
    assumptions: dict
    simulation: dict
    recommendation: dict
    narrative: str
    narrative_source: str
    mode: str


class PriceState(PriceIn, total=False):
    history: list[dict]
    unit_cost: float
    current_list_price: float
    elasticity: dict
    estimate_notes: str
    assumptions: dict
    simulation: dict
    recommendation: dict
    narrative: str
    narrative_source: str
    mode: str


# --------------------------------------------------------------------------- pure helpers (unit-tested)
def fit_elasticity(prices: list[float], units: list[float], *, seed: int = BOOTSTRAP_SEED, n_boot: int = BOOTSTRAP_N) -> dict:
    """Log-log OLS: ln(units) = a + e * ln(price). Returns the point estimate ``e`` and a seeded bootstrap interval."""
    x, y = np.log(np.asarray(prices, dtype=float)), np.log(np.asarray(units, dtype=float))
    if len(x) < MIN_WEEKS:
        raise ValueError(f"Only {len(x)} weeks with sales; need at least {MIN_WEEKS} to estimate elasticity.")
    if np.ptp(x) < 1e-6:
        raise ValueError("The price never changed in this data, so its effect on demand cannot be estimated.")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 1e-6:
            slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.percentile(slopes, BOOTSTRAP_INTERVAL)
    return {
        "point": round(float(slope), 3),
        "lo": round(float(lo), 3),
        "hi": round(float(hi), 3),
        "interval": f"{BOOTSTRAP_INTERVAL[1] - BOOTSTRAP_INTERVAL[0]}% bootstrap ({len(slopes)} resamples, seed {seed})",
        "r2": round(r2, 3),
        "n_weeks": int(len(x)),
        "price_min": round(float(np.exp(x.min())), 2),
        "price_max": round(float(np.exp(x.max())), 2),
        "method": "OLS of ln(weekly units) on ln(weekly average selling price); weeks without sales excluded",
    }


def candidate_grid(baseline_price: float) -> list[int]:
    lo = math.ceil(PRICE_BAND[0] * baseline_price / PRICE_STEP) * PRICE_STEP
    hi = math.floor(PRICE_BAND[1] * baseline_price / PRICE_STEP) * PRICE_STEP
    return list(range(lo, hi + 1, PRICE_STEP))


def simulate_price(a: dict, price: float) -> dict:
    """Project the horizon at one average selling price from the assumptions dict (constant-elasticity demand)."""
    ratio = price / a["baseline_price"]
    per_week = [a["baseline_units_per_week"] * ratio ** e for e in (a["elasticity"], a["elasticity_low"], a["elasticity_high"])]
    units_wk, lo_wk, hi_wk = per_week[0], min(per_week), max(per_week)
    unit_margin = price * (1 - a["variable_cost_rate"]) - a["unit_cost"]
    units = units_wk * a["horizon_weeks"]
    margin_pct = unit_margin / price
    return {
        "price": round(float(price), 2),
        "units_per_week": round(units_wk, 2),
        "units_per_week_low": round(lo_wk, 2),
        "units_per_week_high": round(hi_wk, 2),
        "units": round(units, 1),
        "revenue": round(units * price, 2),
        "contribution": round(units * unit_margin, 2),
        "margin_pct": round(margin_pct, 4),
        "meets_margin_target": bool(round(margin_pct, 4) >= a["margin_target"]),
        "within_support": bool(a["support_min"] <= price <= a["support_max"]),
    }


def choose_price(rows: list[dict]) -> dict | None:
    """Deterministic argmax: highest total contribution among rows that meet the target and are within support."""
    feasible = [r for r in rows if r["meets_margin_target"] and r["within_support"]]
    return max(feasible, key=lambda r: (r["contribution"], -r["price"])) if feasible else None


def _money(x: float) -> str:
    return f"${x:,.0f}"


def quotes_for(rec: dict, a: dict) -> dict:
    """The exact strings the narrative must contain, formatted once so the check and the template agree."""
    if rec["status"] != "recommended":
        return {"target": f"{a['margin_target']:.0%}"}
    return {"price": _money(rec["price"]), "margin": f"{rec['margin_pct']:.1%}", "contribution": _money(rec["contribution"])}


def template_narrative(rec: dict, a: dict) -> str:
    """Fixed wording used when the model's paragraph omits a required figure."""
    if rec["status"] != "recommended":
        return (
            f"No candidate price between {_money(min(a['candidate_prices']))} and {_money(max(a['candidate_prices']))} meets a "
            f"{a['margin_target']:.0%} contribution margin within the price range the data supports "
            f"({_money(a['support_min'])} to {_money(a['support_max'])}). Lower the target or widen the range."
        )
    q = quotes_for(rec, a)
    text = (
        f"At a {a['margin_target']:.0%} margin floor, the best candidate is {q['price']} (today {_money(a['baseline_price'])}): "
        f"{rec['margin_pct']:.1%} margin, about {rec['units_per_week']:.1f} units a week ({rec['units_vs_baseline_pct']:+.0f}% vs today) and "
        f"{q['contribution']} contribution over {a['horizon_weeks']} weeks ({rec['contribution_vs_baseline_pct']:+.0f}% vs today)."
    )
    return text + (" " + " ".join(rec["warnings"]) if rec["warnings"] else "")


# --------------------------------------------------------------------------- nodes
def load_history(state: PriceState) -> dict:
    """Weekly units and average selling price for one product, plus its unit cost and list price, from sales.csv.

    A picked ready-made case sets BOTH ``product`` and ``margin_target`` (``None`` = the default), so it replaces whatever
    an earlier run on this thread left in them; ``set_assumptions`` reads the margin from the state this node writes.
    """
    picked = expand(state, CASES, has_input=lambda s: bool(s.get("product") or s.get("margin_target")))
    state = {**state, **picked}
    product = state.get("product") or DEFAULT_PRODUCT
    df = read_csv("sales.csv")
    known = sorted(df["product"].unique())
    if product not in known:
        raise ValueError(f"Unknown product {product!r}; try one of: {', '.join(known)}")
    df = df[(df["product"] == product) & (df["status"] != "cancelled")].copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["week"] = df["order_date"].dt.to_period("W").dt.start_time  # Monday of the ISO week
    weekly = df.groupby("week").agg(units=("quantity", "sum"), revenue=("revenue", "sum")).reset_index()
    weekly["avg_price"] = weekly["revenue"] / weekly["units"]
    if len(weekly) < MIN_WEEKS:
        raise ValueError(f"{product} has only {len(weekly)} weeks with sales; need at least {MIN_WEEKS}.")
    latest = df.sort_values(["order_date", "order_id"]).iloc[-1]
    return {
        **picked,
        "product": product,
        "history": [
            {"week": r.week.strftime("%Y-%m-%d"), "units": int(r.units), "revenue": round(float(r.revenue), 2), "avg_price": round(float(r.avg_price), 4)}
            for r in weekly.itertuples()
        ],
        "unit_cost": round(float(df["cost"].sum() / df["quantity"].sum()), 2),
        "current_list_price": float(latest["unit_price"]),
        "mode": mode_label(),
    }


def estimate_elasticity(state: PriceState) -> dict:
    """Deterministic demand model: log-log slope with a seeded bootstrap interval."""
    rows = state["history"]
    return {"elasticity": fit_elasticity([r["avg_price"] for r in rows], [r["units"] for r in rows])}


REVIEW_PROMPT = (
    "You are a pricing analyst. Using ONLY the figures given, write exactly three short sentences for a pricing manager: "
    "(1) what the elasticity means in plain terms (for a 10% price increase, roughly what % change in units, using point "
    "estimate x 10%); (2) how far to trust it, referring to the number of weeks, R-squared and the width of the interval; "
    "(3) the single most important caveat. Do not recommend a price and do not introduce numbers that are not given."
)


def review_estimate(state: PriceState) -> dict:
    """LLM: a plain-English reading of the fit. Upstream of the fork point, so a fork never pays for it again."""
    e = state["elasticity"]
    facts = (
        f"Product: {state['product']}\nElasticity point estimate: {e['point']}\n90% interval: {e['lo']} to {e['hi']}\n"
        f"R-squared: {e['r2']}\nWeeks with sales: {e['n_weeks']}\nObserved weekly average price range: ${e['price_min']} to ${e['price_max']}"
    )
    text = get_llm("fast").invoke([SystemMessage(REVIEW_PROMPT), HumanMessage(facts)])
    return {"estimate_notes": str(text.text).strip()}


def assumptions(state: PriceState) -> dict:
    """THE fork point: one explicit dict with every number the rest of the graph may use. Pure code, no model."""
    rows, e = state["history"], state["elasticity"]
    last = pd.Timestamp(rows[-1]["week"])
    window = [r for r in rows if pd.Timestamp(r["week"]) > last - pd.Timedelta(weeks=BASELINE_WEEKS)]
    units, revenue = sum(r["units"] for r in window), sum(r["revenue"] for r in window)
    baseline_price = revenue / units
    notes = [f"Baseline = last {BASELINE_WEEKS} calendar weeks of sales (weeks without sales count as zero units)."]
    elasticity = e["point"]
    if elasticity > ELASTICITY_CEILING:
        notes.append(f"Fitted elasticity {elasticity} is not meaningfully negative; clamped to {ELASTICITY_CEILING}.")
        elasticity = ELASTICITY_CEILING
    return {
        "assumptions": {
            "product": state["product"],
            "unit_cost": state["unit_cost"],
            "variable_cost_rate": DEFAULT_VARIABLE_COST_RATE,
            "margin_target": float(state.get("margin_target") or DEFAULT_MARGIN_TARGET),
            "elasticity": elasticity,
            "elasticity_low": min(e["lo"], elasticity),
            "elasticity_high": max(min(e["hi"], -0.05), elasticity),
            "baseline_price": round(baseline_price, 2),
            "baseline_units_per_week": round(units / BASELINE_WEEKS, 3),
            "horizon_weeks": HORIZON_WEEKS,
            "candidate_prices": candidate_grid(baseline_price),
            "support_min": round(e["price_min"] * (1 - SUPPORT_MARGIN), 2),
            "support_max": round(e["price_max"] * (1 + SUPPORT_MARGIN), 2),
            "notes": notes,
        }
    }


def simulate(state: PriceState) -> dict:
    """Project each candidate price (and today's price as a baseline row) from the assumptions alone."""
    a = state["assumptions"]
    return {"simulation": {"baseline": simulate_price(a, a["baseline_price"]), "rows": [simulate_price(a, p) for p in a["candidate_prices"]]}}


NARRATIVE_PROMPT = (
    "You are a pricing analyst writing for a pricing manager. Write ONE paragraph of at most 90 words. State the "
    "recommendation first, then the trade-off against today's price, then the single biggest caveat. You MUST quote "
    "these figures exactly as written: {quotes}. Use no other numbers except ones in the facts. Do not invent."
)


def recommend(state: PriceState) -> dict:
    """Pick the price in code; let the model write the paragraph; verify the paragraph quotes the real numbers."""
    a, sim = state["assumptions"], state["simulation"]
    best = choose_price(sim["rows"])
    base = sim["baseline"]
    if best is None:
        rec: dict[str, Any] = {
            "status": "no_feasible_price", "price": None, "margin_target": a["margin_target"],
            "reason": "No candidate meets the margin target inside the supported price range.",
            "warnings": [],
        }
    else:
        warnings = []
        obs = state["elasticity"]
        if best["price"] > obs["price_max"] or best["price"] < obs["price_min"]:
            warnings.append(f"{_money(best['price'])} is outside the weekly average prices ever observed ({_money(obs['price_min'])} to {_money(obs['price_max'])}), so this is an extrapolation.")
        if obs["r2"] < 0.25:
            warnings.append(f"The demand model explains little of the weekly variation (R-squared {obs['r2']}); treat the units forecast as indicative.")
        if best["units_per_week_high"] > 1.5 * best["units_per_week_low"]:
            warnings.append(f"The units forecast is very uncertain ({best['units_per_week_low']:.1f} to {best['units_per_week_high']:.1f} a week across the elasticity interval).")
        runner_up = sorted((r for r in sim["rows"] if r["meets_margin_target"] and r["within_support"] and r is not best), key=lambda r: -r["contribution"])[:1]
        rec = {
            "status": "recommended",
            **best,
            "margin_target": a["margin_target"],
            "units_vs_baseline_pct": round(100 * (best["units"] / base["units"] - 1), 1),
            "contribution_vs_baseline_pct": round(100 * (best["contribution"] / base["contribution"] - 1), 1),
            "runner_up": {"price": runner_up[0]["price"], "contribution": runner_up[0]["contribution"]} if runner_up else None,
            "warnings": warnings,
        }
    quotes = quotes_for(rec, a)
    facts = {"recommendation": rec, "baseline_today": base, "margin_target": a["margin_target"], "estimate_notes": state.get("estimate_notes", "")}
    draft = str(
        get_llm()
        .invoke([SystemMessage(NARRATIVE_PROMPT.format(quotes=", ".join(quotes.values()))), HumanMessage(str(facts))])
        .text
    ).strip()
    if all(q in draft for q in quotes.values()):
        narrative, source = draft, "model (checked: quotes the simulated figures)"
    else:
        narrative, source = template_narrative(rec, a), "template (model text omitted a required figure)"
    return {"recommendation": rec, "narrative": narrative, "narrative_source": source}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(PriceState, input_schema=PriceIn, output_schema=PriceOut)
    g.add_node("load_history", load_history)
    g.add_node("estimate_elasticity", estimate_elasticity)
    g.add_node("review_estimate", review_estimate)
    g.add_node("set_assumptions", assumptions)
    g.add_node("simulate", simulate)
    g.add_node("recommend", recommend)
    g.add_edge(START, "load_history")
    g.add_edge("load_history", "estimate_elasticity")
    g.add_edge("estimate_elasticity", "review_estimate")
    g.add_edge("review_estimate", "set_assumptions")
    g.add_edge("set_assumptions", "simulate")
    g.add_edge("simulate", "recommend")
    g.add_edge("recommend", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

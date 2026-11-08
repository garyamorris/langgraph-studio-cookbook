"""A typed, safe query layer over ``data/sales.csv`` shared by the analytics examples.

Private helper (the leading underscore keeps it out of the example registry). It exists so that every example which
answers questions about sales does it the same way: the model chooses *arguments* from closed vocabularies, this code
runs pandas, and the model reads back exact numbers. Nothing here evaluates model-written code.

CONVENTIONS (also served to models by ``describe_dataset``)
    * One row of ``sales.csv`` is one order *line*. ``orders`` counts distinct ``order_id``.
    * ``status`` filter defaults to ``"completed"`` = delivered + returned lines. Cancelled lines never happened, so
      they are excluded unless you ask for ``"all"`` or ``"cancelled"``. Revenue therefore includes returned lines
      (gross of returns); pass ``status="delivered"`` for revenue net of returns.
    * ``return_rate`` = share of the included lines whose status is ``returned``, in percent (0-100).
    * ``avg_discount`` = mean ``discount_pct`` over the included lines, in percent.
    * Money is rounded to cents, percentages to one decimal.
    * ``AS_OF`` is the dataset's "today": 2026-10-31. The last *full* quarter is 2026Q3.

Every public function raises ``ValueError`` with an actionable message for bad arguments (unknown product, malformed
date, unsupported group_by ...). Callers hand ``str(error)`` back to a model as an observation so it can correct itself.
"""

from __future__ import annotations

import calendar
import difflib
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Sequence

import pandas as pd

from deepdive.common.data import read_csv

# The dataset's notion of "today". Defined once; everything that says "last quarter" derives from it.
AS_OF = date(2026, 10, 31)

GROUP_BYS = ("product", "category", "region", "segment", "channel", "month", "quarter", "status")
TIME_GROUPS = ("month", "quarter")
METRICS = ("revenue", "units", "orders", "margin", "return_rate", "avg_discount")
STATUS_FILTERS = ("completed", "delivered", "returned", "cancelled", "all")
MAX_TOP_N = 50

METRIC_DOCS = {
    "revenue": "sum of line revenue, USD (after discount; includes returned lines unless status='delivered')",
    "units": "sum of quantity",
    "orders": "number of distinct order_id",
    "margin": "sum of line margin (revenue - cost), USD",
    "return_rate": "percent of included order lines whose status is 'returned' (0-100)",
    "avg_discount": "mean discount_pct over included order lines, percent",
}
METRIC_UNITS = {"revenue": "USD", "units": "units", "orders": "orders", "margin": "USD", "return_rate": "%", "avg_discount": "%"}


# --------------------------------------------------------------------------- data
def load_sales() -> pd.DataFrame:
    """The sales table with three derived columns: ``date`` (datetime), ``month`` ("2026-05"), ``quarter`` ("2026Q2")."""
    df = read_csv("sales.csv")
    df["date"] = pd.to_datetime(df["order_date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["quarter"] = df["date"].dt.year.astype(str) + "Q" + df["date"].dt.quarter.astype(str)
    return df


def coverage() -> tuple[date, date]:
    """First and last order date in the data."""
    df = load_sales()
    return df["date"].min().date(), df["date"].max().date()


def _values(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(str(v) for v in df[col].unique())


# --------------------------------------------------------------------------- dates, quarters, periods
@dataclass(frozen=True)
class Quarter:
    """A calendar quarter such as ``2026Q2``."""

    year: int
    q: int

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.q}"

    @property
    def start(self) -> date:
        return date(self.year, 3 * (self.q - 1) + 1, 1)

    @property
    def end(self) -> date:
        month = 3 * self.q
        return date(self.year, month, calendar.monthrange(self.year, month)[1])

    def previous(self) -> "Quarter":
        return Quarter(self.year - 1, 4) if self.q == 1 else Quarter(self.year, self.q - 1)


_QUARTER_RE = re.compile(r"^\s*(?:(\d{4})\s*-?\s*Q([1-4])|Q([1-4])\s*-?\s*(\d{4}))\s*$", re.IGNORECASE)
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
PERIOD_HELP = (
    "Use a quarter like 2026Q2, a month like 2026-05, a year like 2026, a date like 2026-05-01, "
    "or a range like 2026-04-01..2026-06-30."
)


def parse_quarter(text: str) -> Quarter:
    """Parse ``"2026Q2"`` (also ``"2026-Q2"``, ``"Q2 2026"``, any case) into a ``Quarter``."""
    m = _QUARTER_RE.match(str(text))
    if not m:
        raise ValueError(f"Could not read quarter {text!r}; write it like 2026Q2 (year, then Q1-Q4).")
    year, q = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    return Quarter(int(year), int(q))


def quarter_of(d: date) -> Quarter:
    return Quarter(d.year, (d.month - 1) // 3 + 1)


def last_full_quarter(as_of: date = AS_OF) -> Quarter:
    """The most recent quarter that had completely ended on ``as_of`` ("last quarter" in conversation)."""
    q = quarter_of(as_of)
    return q if as_of >= q.end else q.previous()


def parse_date(text: Any, name: str = "date") -> date:
    """Strict ``YYYY-MM-DD`` (no times, no free text)."""
    s = str(text).strip()
    if not _DAY_RE.match(s):
        raise ValueError(f"{name} must be a calendar date like 2026-05-01, got {text!r}.")
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{name} {text!r} is not a real calendar date; use YYYY-MM-DD.") from None


def _period_edge(text: str, edge: str) -> date:
    t = str(text).strip()
    if _DAY_RE.match(t):
        return parse_date(t)
    if m := _MONTH_RE.match(t):
        year, month = int(m.group(1)), int(m.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"Month {t!r} is not valid; the month must be 01-12.")
        return date(year, month, 1 if edge == "start" else calendar.monthrange(year, month)[1])
    if _QUARTER_RE.match(t):
        q = parse_quarter(t)
        return q.start if edge == "start" else q.end
    if _YEAR_RE.match(t):
        return date(int(t), 1, 1) if edge == "start" else date(int(t), 12, 31)
    raise ValueError(f"Could not read period {text!r}. {PERIOD_HELP}")


def parse_period(text: str) -> tuple[date, date]:
    """Turn a period phrase into inclusive ``(start, end)`` dates.

    Accepts a quarter (``2026Q2``), month (``2026-05``), year (``2026``), single day, or a range ``A..B`` whose ends
    may each be any of those (``2026-02..2026-04`` is Feb 1 to Apr 30).
    """
    t = str(text).strip()
    for sep in ("..", " to "):
        if sep in t:
            a, b = t.split(sep, 1)
            start, end = _period_edge(a, "start"), _period_edge(b, "end")
            if start > end:
                raise ValueError(f"Period {text!r} ends before it starts.")
            return start, end
    return _period_edge(t, "start"), _period_edge(t, "end")


def resolve_window(period: str | None = None, start_date: str | None = None, end_date: str | None = None) -> tuple[date, date]:
    """Inclusive date window from either ``period`` or ``start_date``/``end_date`` (default: the whole dataset).

    Raises if the window does not overlap the data at all, which almost always means the caller resolved a relative
    phrase ("last quarter") wrongly - better to say so than to return an empty table.
    """
    lo, hi = coverage()
    if period:
        if start_date or end_date:
            raise ValueError("Give either period or start_date/end_date, not both.")
        start, end = parse_period(period)
    else:
        start = parse_date(start_date, "start_date") if start_date else lo
        end = parse_date(end_date, "end_date") if end_date else hi
        if start > end:
            raise ValueError(f"start_date {start} is after end_date {end}.")
    if end < lo or start > hi:
        raise ValueError(f"The window {start}..{end} is outside the data, which covers {lo}..{hi}.")
    return start, end


# --------------------------------------------------------------------------- argument validation
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


NO_FILTER = {"", "all", "any", "none", "null", "n/a"}  # what a model writes in a filter field when it means "no filter"


def canonical(value: str | None, valid: Sequence[str], label: str, *, strip_prefix: str = "") -> str | None:
    """Map a user/model spelling to a real value: case/punctuation-insensitive, optional prefix, unique-prefix match.

    ``canonical("cam pro", products, "product", strip_prefix="lumen")`` -> ``"Lumen Cam Pro"``. ``None`` and words that
    mean "no filter" (``"all"``, ``"none"``, ``""``) return ``None``. Unknown values raise
    ``ValueError`` listing the valid values (and the closest one, when there is one).
    """
    if value is None or str(value).strip().lower() in NO_FILTER:
        return None

    def key(s: str) -> str:
        k = _norm(s)
        return k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k

    wanted = key(value)
    keyed = {v: key(v) for v in valid}
    exact = [v for v, k in keyed.items() if k == wanted]
    if exact:
        return exact[0]
    prefix = [v for v, k in keyed.items() if wanted and k.startswith(wanted)]
    if len(prefix) == 1:
        return prefix[0]
    close = difflib.get_close_matches(wanted, list(keyed.values()), n=1, cutoff=0.6)
    hint = f" Did you mean {next(v for v, k in keyed.items() if k == close[0])!r}?" if close else ""
    raise ValueError(f"Unknown {label} {value!r}.{hint} Valid {label} values: {', '.join(valid)}.")


def normalize_metrics(metrics: Iterable[str] | str | None) -> list[str]:
    """A list of metric names (a comma-separated string is accepted). Defaults to ``["revenue"]``."""
    if metrics is None or metrics == "":
        return ["revenue"]
    items = [m.strip() for m in metrics.split(",")] if isinstance(metrics, str) else [str(m).strip() for m in metrics]
    items = [m for m in items if m]
    if not items:
        return ["revenue"]
    bad = [m for m in items if m not in METRICS]
    if bad:
        raise ValueError(f"Unsupported metric {bad[0]!r}; choose from {', '.join(METRICS)}.")
    return list(dict.fromkeys(items))


def check_group_by(group_by: str | None) -> str | None:
    if group_by is None or group_by == "":
        return None
    if group_by not in GROUP_BYS:
        raise ValueError(f"group_by must be one of {', '.join(GROUP_BYS)}; got {group_by!r}.")
    return group_by


# --------------------------------------------------------------------------- the query
def _filter(
    df: pd.DataFrame,
    *,
    product: str | None,
    category: str | None,
    region: str | None,
    segment: str | None,
    channel: str | None,
    status: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply validated filters. Returns the filtered frame and the canonical filters that were applied."""
    applied: dict[str, str] = {}
    for col, raw, prefix in (
        ("product", product, "lumen"),
        ("category", category, ""),
        ("region", region, ""),
        ("segment", segment, ""),
        ("channel", channel, ""),
    ):
        value = canonical(raw, _values(df, col), col, strip_prefix=prefix)
        if value is not None:
            df = df[df[col] == value]
            applied[col] = value
    status = status or "completed"
    if status not in STATUS_FILTERS:
        raise ValueError(f"status must be one of {', '.join(STATUS_FILTERS)}; got {status!r}.")
    if status == "completed":
        df = df[df["status"] != "cancelled"]
    elif status != "all":
        df = df[df["status"] == status]
    applied["status"] = status
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df[mask], applied


def _metric_values(g: pd.DataFrame, metrics: Sequence[str]) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    for m in metrics:
        if m == "revenue":
            out[m] = round(float(g["revenue"].sum()), 2)
        elif m == "units":
            out[m] = int(g["quantity"].sum())
        elif m == "orders":
            out[m] = int(g["order_id"].nunique())
        elif m == "margin":
            out[m] = round(float(g["margin"].sum()), 2)
        elif m == "return_rate":
            out[m] = round(float((g["status"] == "returned").mean()) * 100, 1) if len(g) else None
        elif m == "avg_discount":
            out[m] = round(float(g["discount_pct"].mean()), 1) if len(g) else None
    return out


def query_sales(
    *,
    metrics: Sequence[str] | str | None = None,
    group_by: str | None = None,
    product: str | None = None,
    category: str | None = None,
    region: str | None = None,
    segment: str | None = None,
    channel: str | None = None,
    status: str = "completed",
    period: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
) -> dict[str, Any]:
    """Aggregate sales. All arguments are flat, optional and validated; nothing is evaluated.

    Returns a JSON-serialisable dict: ``total`` (all matched lines), ``rows`` (one per group when ``group_by`` is
    set), the canonical ``filters`` applied, the ``window``, and ``notes``. Time groups (month, quarter) come back in
    chronological order; other groups are ranked by ``sort_by`` (default: the first metric, highest first). ``top_n``
    keeps the first n rows *after* ranking, so with a time group it keeps the n best periods, not the latest.
    """
    metric_list = normalize_metrics(metrics)
    group_by = check_group_by(group_by)
    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= MAX_TOP_N:
            raise ValueError(f"top_n must be a whole number between 1 and {MAX_TOP_N}; got {top_n!r}.")
        if group_by is None:
            raise ValueError("top_n needs a group_by (what should be ranked?).")
    if sort_by is not None and sort_by not in metric_list:
        raise ValueError(f"sort_by {sort_by!r} must be one of the requested metrics: {', '.join(metric_list)}.")
    start, end = resolve_window(period, start_date, end_date)

    df, filters = _filter(
        load_sales(), product=product, category=category, region=region, segment=segment, channel=channel,
        status=status, start=start, end=end,
    )
    notes: list[str] = []
    lo, hi = coverage()
    if start < lo or end > hi:
        notes.append(f"The requested window extends beyond the data, which covers {lo}..{hi}; only that part is counted.")
    if df.empty:
        notes.append("No order lines matched these filters.")

    result: dict[str, Any] = {
        "group_by": group_by,
        "metrics": metric_list,
        "filters": filters,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "lines": int(len(df)),
        "total": _metric_values(df, metric_list),
        "notes": notes,
    }
    if group_by:
        rows = [{"group": str(key), **_metric_values(g, metric_list), "lines": int(len(g))} for key, g in df.groupby(group_by, sort=True)]
        if group_by not in TIME_GROUPS or top_n is not None or sort_by is not None:
            key_metric = sort_by or metric_list[0]
            # None (empty metric) sorts last; ties break alphabetically so output is deterministic.
            rows.sort(key=lambda r: (r[key_metric] is None, (r[key_metric] or 0) * (1 if ascending else -1), r["group"]))
        result["rows_matched"] = len(rows)
        result["rows"] = rows[:top_n] if top_n else rows
    return result


# --------------------------------------------------------------------------- comparing two periods
def _pct_change(a: float | None, b: float | None) -> float | None:
    if a in (None, 0) or b is None:
        return None
    return round((b - a) / a * 100, 1)


def compare_periods(
    *,
    metric: str,
    period_a: str,
    period_b: str,
    group_by: str | None = None,
    product: str | None = None,
    category: str | None = None,
    region: str | None = None,
    segment: str | None = None,
    channel: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """One metric over period A and period B (same filters), with the absolute and percentage change B vs A.

    ``change`` is ``b - a`` (percentage points for the percent metrics ``return_rate`` / ``avg_discount``);
    ``pct_change`` is relative to ``a`` and is ``None`` when ``a`` is zero or missing. With ``group_by`` it also
    compares every group (a group missing from a period counts as zero lines: value ``None`` for the rate metrics).
    Periods of unequal length are compared as given but flagged in ``notes`` - totals are not per-day rates.
    """
    metric_list = normalize_metrics(metric)
    if len(metric_list) != 1:
        raise ValueError("compare_periods compares exactly one metric.")
    metric = metric_list[0]
    group_by = check_group_by(group_by)
    sides: dict[str, dict[str, Any]] = {}
    for tag, text in (("a", period_a), ("b", period_b)):
        if not text:
            raise ValueError(f"period_{tag} is required. {PERIOD_HELP}")
        start, end = parse_period(text)
        sides[tag] = query_sales(
            metrics=[metric], group_by=group_by, product=product, category=category, region=region, segment=segment,
            channel=channel, status=status, start_date=start.isoformat(), end_date=end.isoformat(),
        )
    a, b = sides["a"]["total"][metric], sides["b"]["total"][metric]
    days = {t: (date.fromisoformat(s["window"]["end"]) - date.fromisoformat(s["window"]["start"])).days + 1 for t, s in sides.items()}
    notes = sorted({n for s in sides.values() for n in s["notes"]})
    if days["a"] != days["b"]:
        notes.append(f"The periods have different lengths ({days['a']} vs {days['b']} days); compare per-day rates, not totals.")
    out: dict[str, Any] = {
        "metric": metric,
        "unit": METRIC_UNITS[metric],
        "filters": sides["a"]["filters"],
        "period_a": {**sides["a"]["window"], "days": days["a"], "lines": sides["a"]["lines"]},
        "period_b": {**sides["b"]["window"], "days": days["b"], "lines": sides["b"]["lines"]},
        "a": a,
        "b": b,
        "change": None if a is None or b is None else round(b - a, 2),
        "pct_change": _pct_change(a, b),
        "notes": notes,
    }
    if group_by:
        va = {r["group"]: r[metric] for r in sides["a"]["rows"]}
        vb = {r["group"]: r[metric] for r in sides["b"]["rows"]}
        groups = []
        for g in sorted(set(va) | set(vb)):
            ga, gb = va.get(g), vb.get(g)
            if metric not in ("return_rate", "avg_discount"):
                ga, gb = ga if ga is not None else 0, gb if gb is not None else 0
            groups.append({"group": g, "a": ga, "b": gb, "change": None if ga is None or gb is None else round(gb - ga, 2), "pct_change": _pct_change(ga, gb)})
        groups.sort(key=lambda r: (r["change"] is None, -(r["change"] or 0), r["group"]))
        out["groups"] = groups
    return out


# --------------------------------------------------------------------------- like-for-like quarter windows
def quarter_windows(quarter: Quarter) -> dict[str, Any]:
    """The effective window of a quarter and the like-for-like window of the quarter before it.

    The data does not cover every quarter fully: 2025Q4 starts on Nov 1 and 2026Q4 ends on Oct 31. Comparing a
    one-month quarter with a three-month one would show a fake collapse, so the prior window is shifted by the same
    day offsets (Oct 1-31 is compared with Jul 1-31). ``prior_comparable`` is False unless the dataset covers that whole
    prior window (2025Q4 has no earlier quarter; 2026Q1 would be compared with a two-month 2025Q4).
    """
    lo, hi = coverage()
    start, end = max(quarter.start, lo), min(quarter.end, hi)
    if start > end:
        raise ValueError(f"No data for {quarter.label}; the dataset covers {lo}..{hi}.")
    prev = quarter.previous()
    partial = (start, end) != (quarter.start, quarter.end)
    if partial:  # same day offsets inside the prior quarter
        p_start = prev.start + (start - quarter.start)
        p_end = min(prev.start + (end - quarter.start), prev.end)
    else:
        p_start, p_end = prev.start, prev.end
    prior_comparable = p_start >= lo and p_end <= hi
    return {
        "quarter": quarter.label,
        "start": start,
        "end": end,
        "days": (end - start).days + 1,
        "quarter_days": (quarter.end - quarter.start).days + 1,
        "partial": partial,
        "prior": prev.label,
        "prior_start": p_start,
        "prior_end": p_end,
        "prior_comparable": prior_comparable,
    }


# --------------------------------------------------------------------------- display formatting
def money(x: float | None) -> str:
    """``$1,234.50`` (``n/a`` for None)."""
    return "n/a" if x is None else f"${x:,.2f}"


def pct(x: float | None) -> str:
    """``12.3%`` (``n/a`` for None)."""
    return "n/a" if x is None else f"{x:.1f}%"


def format_value(metric: str, value: float | None) -> str:
    """A metric value the way a person reads it: money, whole numbers, or percent."""
    if value is None:
        return "n/a"
    if metric in ("revenue", "margin"):
        return money(value)
    if metric in ("return_rate", "avg_discount"):
        return pct(value)
    return f"{int(value):,}"


# --------------------------------------------------------------------------- self-description for models
def describe_dataset() -> dict[str, Any]:
    """Everything a model needs to ask good questions: vocabularies, definitions, 'today', and relative periods."""
    df = load_sales()
    lo, hi = coverage()
    last_q = last_full_quarter()
    this_q = quarter_of(AS_OF)
    return {
        "order_lines": int(len(df)),
        "distinct_orders": int(df["order_id"].nunique()),
        "covers": {"start": lo.isoformat(), "end": hi.isoformat()},
        "today": AS_OF.isoformat(),
        "last_full_quarter": {"label": last_q.label, "start": last_q.start.isoformat(), "end": last_q.end.isoformat()},
        "current_quarter": {"label": this_q.label, "start": this_q.start.isoformat(), "end": min(this_q.end, hi).isoformat(), "note": "partial: data ends today"},
        "last_month": (AS_OF.replace(day=1) - timedelta(days=1)).strftime("%Y-%m"),
        "products": _values(df, "product"),
        "categories": _values(df, "category"),
        "regions": _values(df, "region"),
        "segments": _values(df, "segment"),
        "channels": _values(df, "channel"),
        "statuses": {k: int(v) for k, v in df["status"].value_counts().sort_index().items()},
        "group_by_options": list(GROUP_BYS),
        "metrics": dict(METRIC_DOCS),
        "status_filter": "completed (default) = delivered + returned; delivered; returned; cancelled; all",
        "period_formats": PERIOD_HELP,
    }


def dataset_brief() -> str:
    """``describe_dataset`` as a compact paragraph for a system prompt."""
    d = describe_dataset()
    metrics = "; ".join(f"{k}: {v}" for k, v in d["metrics"].items())
    return (
        f"Dataset: {d['order_lines']} order lines of the fictional company Lumen Home, {d['covers']['start']} to "
        f"{d['covers']['end']}. Today is {d['today']}. 'Last quarter' = {d['last_full_quarter']['label']} "
        f"({d['last_full_quarter']['start']}..{d['last_full_quarter']['end']}); 'this quarter' = "
        f"{d['current_quarter']['label']} (October only so far); 'last month' = {d['last_month']}.\n"
        f"Products: {', '.join(d['products'])}.\nRegions: {', '.join(d['regions'])}.\n"
        f"Segments: {', '.join(d['segments'])}. Channels: {', '.join(d['channels'])}. "
        f"Categories: {', '.join(d['categories'])}.\n"
        f"Group by: {', '.join(d['group_by_options'])}.\nMetrics: {metrics}.\n"
        f"Order lines have a status (delivered / returned / cancelled); the default 'completed' filter excludes "
        f"cancelled lines. Periods: {d['period_formats']}"
    )

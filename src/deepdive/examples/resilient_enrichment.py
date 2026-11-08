"""Batch enrichment from a flaky vendor API: retries, cache, error handler, timeout, live progress.

PROBLEM
    A sales-ops team enriches a batch of company records (industry, headcount, headquarters) from a third-party data
    vendor. The vendor is unreliable in the usual ways: requests time out or get HTTP 429 and succeed a moment later,
    for one company the vendor's upstream source is down for the whole run, and a few records are simply bad (the
    vendor answers 404 and always will). The job has to survive all of that: one bad record must not sink the other
    39, transient failures must be retried with backoff (and only transient ones), a company that was already looked
    up should not cost another paid call, and the operator watching the run wants to see progress while it happens.

    Everything the vendor does is SIMULATED in this file (no network): ``vendor_fetch(domain, attempt)`` is a pure
    function whose failures depend only on the domain and the attempt number, so every run, and every test, behaves
    exactly the same way.

DESIGN
    Failures are sorted into two kinds at the point where they are raised, because they need opposite treatment:

      * ``TransientVendorError`` (timeouts, 429, 503): worth retrying. The node's ``RetryPolicy`` is told to retry on
        this class only, up to 4 attempts with exponential backoff and jitter.
      * ``PermanentVendorError`` (404, malformed domain): retrying is wasted money. It is not in ``retry_on``, so it
        goes straight to the node's ``error_handler``.

    When retries are exhausted (or the error is permanent) LangGraph calls the node's ``error_handler``. The handler
    records a ``failed_*`` result for that one domain and routes on to the next stage, so the rest of the batch is
    untouched. Without a handler a single failure would raise out of the whole run.

    Duplicates and the cache. ``plan`` groups records by domain and fans out ONE lookup per unique domain
    (a duplicate within a batch never reaches the vendor). Across runs the cache takes over: ``lookup`` has a
    ``CachePolicy`` keyed on the domain alone, so re-running a batch - for instance after the vendor has recovered -
    serves the companies that succeeded from the cache and calls the vendor only for the ones that failed (failed
    attempts are never cached). Entries expire after ``CACHE_TTL_S`` (10 minutes here; pick what your data tolerates). The key matters: the default key hashes the whole Send payload, which here includes
    the batch id, so every run would miss. Why not rely on the cache for duplicates inside one batch too? Because
    parallel tasks of one step are all scheduled before any of them finishes, so both would call the vendor (verified).

    Graceful degradation. After a successful lookup, ``normalize`` maps vendor labels to our taxonomy through a
    simulated slow service, guarded by a ``TimeoutPolicy``. When it times out, the handler keeps the RAW vendor values
    and marks the record ``enriched_unnormalized``: a slightly worse answer now beats a hung job.

    Progress is a UI concern, so it goes out through ``get_stream_writer()`` as custom events and not through graph
    state: ``plan`` (batch size), ``transient_error`` (what failed and whether it will be retried), ``progress``
    ("looked up 7/13"), ``cache_hit``, ``normalized`` / ``degraded`` and a final ``done``.
    Two LangGraph behaviours shape the code (both verified on 1.2.20):

      * A cache hit replays the node's WRITES but does not run the node, so the node cannot announce its own cache
        hit. Each lookup result is therefore stamped with the id of the batch that really called the vendor; any
        result carrying an older batch id was served from the cache, and ``after_lookup`` reports it.
      * A plain join is enough for ``report``, and ``defer=True`` would change nothing here (verified by running this
        example both ways: one ``report`` run, same summary). Nothing writes to ``report`` until the second-stage
        ``normalize`` tasks, which all run in one step, so it fires once, after them. ``defer`` is NOT a barrier for
        ``Send`` stages in general: if another branch has already written to the join and the only work left is a
        ``Send``, the deferred join runs alongside those ``Send`` tasks and again afterwards (a pending ``Send`` does
        not count as scheduled work; see docs/AUTHORING.md). If you add branches of uneven length, test the join.

GRAPH
    START -> plan -- Send per unique domain --> lookup ----------------------------> after_lookup
                                                 retry (transient errors only)          | finds the cache hits, emits events
                                                 cache keyed by domain                  |
                                                 error_handler -> failed_* result ----->+
                                                                                        +-- Send per success --> normalize --> report --> END
                                                                                        |    (async, TimeoutPolicy; the handler    ^
                                                                                        |     keeps raw values on timeout) --------+
                                                                                        +-- nothing succeeded ---------------------+

LANGGRAPH FEATURES
    * ``RetryPolicy(retry_on=TransientVendorError)``: backoff for transient errors only. ``runtime.execution_info.node_attempt``
      gives the node its attempt number, which is how the simulated vendor knows "this is the 3rd call".
    * ``CachePolicy(key_func=..., ttl=...)`` with ``compile(cache=...)``: paid lookups are reused. ``build_graph``
      supplies an in-memory cache unless you pass one, so Studio re-runs share it for the life of the server process.
    * ``error_handler`` + ``NodeError``: contain one record's failure; the handler runs after the retries are used up.
    * ``TimeoutPolicy(run_timeout=...)`` on an ``async`` node (timeouts only work through asyncio cancellation).
    * ``get_stream_writer()`` custom events for live progress.
    * ``Send`` fan-out, ``Command(goto=[Send, ...])`` for the second fan-out, and ``Overwrite`` to reset the
      accumulating result lists at the start of each run on a thread.

MODELS AND DATA
    No model: this example is about runtime behaviour, and the vendor is a simulation. Data: ``companies.csv`` supplies
    the eight listed companies of the default batch; the rest are made-up private companies defined in this file.
    Special domains in ``SCENARIOS`` misbehave on purpose (see the table in the code).

TRY IT IN STUDIO
    1. Input ``{"count": 14}`` (the default sample). Run it as a normal run; the whole thing takes a few seconds because
       of the real backoff sleeps and the 1 s normalisation timeout. Output ``summary`` should read: 14 records, 13
       unique domains, 11 enriched, 1 enriched_unnormalized (Slowpoke Systems, timed out), 1 failed_permanent
       (Ghostcorp, 404, 1 attempt), 1 failed_retries_exhausted (Meridian Grid, 4 attempts), 8 retries used,
       21 vendor calls, 0 cache hits.
    2. Run the SAME input again (same server process). The 11 companies that succeeded come from the cache
       (``cache_hits: 11``); only the two failing domains call the vendor again (5 calls, 3 retries).
    3. Watch the custom stream events (they are emitted while the run is in progress; if your Studio view does not show
       them, call the API with ``stream_mode="custom"``). Look for ``transient_error`` with ``will_retry: true``, then
       ``progress`` counting up, ``cache_hit`` on the second run and one ``degraded`` for Slowpoke Systems.
    4. Your own records: ``{"records": [{"company": "Foo", "domain": "foo.example"}, {"company": "Bar", "domain": "not a domain"}]}``
       (the malformed domain becomes a ``failed_permanent`` with no retries; ``record_id`` is optional).
    A synchronous ``graph.invoke(...)`` cannot enforce timeouts, so LangGraph refuses to run ``normalize`` that way and
    every record comes back ``enriched_unnormalized`` with that reason. Use ``ainvoke`` / ``astream`` (Studio does).

EXTEND IT
    Respect a vendor rate limit by running the fan-out in waves (a loop that dispatches N domains at a time); add a
    ``RetryPolicy`` for ``NodeTimeoutError`` on ``normalize`` before degrading; or persist failed domains in the Store so
    tomorrow's run starts with them.
"""

import asyncio
import operator
import re
import threading
import uuid
from collections import OrderedDict
from typing import Annotated, Any, Literal

from langgraph.cache.memory import InMemoryCache
from langgraph.config import get_stream_writer
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, Command, Overwrite, RetryPolicy, Send, TimeoutPolicy
from typing_extensions import NotRequired, TypedDict

from deepdive.common.data import read_csv
from deepdive.common.fakes import stable_int
from deepdive.common.meta import ExampleMeta

META = ExampleMeta(
    id="resilient_enrichment",
    title="Resilient batch enrichment (retry, cache, timeout, progress)",
    category="Runtime & resilience",
    problem="Enrich a batch of company records from a flaky vendor API so that transient errors are retried, repeat "
    "lookups hit a cache, one bad record never sinks the batch, and the operator sees live progress.",
    features=["RetryPolicy", "CachePolicy", "error_handler / NodeError", "TimeoutPolicy", "custom streaming", "Send fan-out"],
    models=["none (simulated vendor API)"],
    datasets=["companies.csv"],
    difficulty="advanced",
)

EXAMPLE_INPUT = {"count": 14}

# Knobs (module constants so tests can shrink the waits before building a graph).
MAX_RECORDS = 60
MAX_ATTEMPTS = 4  # 1 call + 3 retries
RETRY_INITIAL_S = 0.2  # backoff: 0.2s, 0.4s, 0.8s (+ jitter)
CACHE_TTL_S = 600
NORMALIZE_TIMEOUT_S = 1.0
SLOW_NORMALIZE_S = 3.0  # how long the simulated normaliser takes for a "slow_normalize" domain


# --------------------------------------------------------------------------- the simulated vendor
class VendorError(Exception):
    """Base class. ``attempt`` is the attempt number on which it was raised (so the handler can report it)."""

    def __init__(self, domain: str, status: int, message: str, attempt: int):
        super().__init__(f"{status} {message} ({domain}, attempt {attempt})")
        self.domain, self.status, self.attempt = domain, status, attempt


class TransientVendorError(VendorError):
    """Timeouts, 429, 503: retrying later can work."""


class PermanentVendorError(VendorError):
    """404, malformed input: retrying cannot work, so it is not in ``retry_on``."""


# domain -> (behaviour, number of failing attempts). Anything not listed is healthy.
SCENARIOS: dict[str, tuple[str, int]] = {
    "pinecrest-labs.example": ("timeout", 2),  # attempts 1-2 time out (504), attempt 3 succeeds
    "harborline-freight.example": ("rate_limit", 1),  # attempt 1 is 429, attempt 2 succeeds
    "cobalt-semiconductor.example": ("rate_limit", 2),  # attempts 1-2 are 429
    "meridian-grid.example": ("down", 0),  # every attempt is 503: retries are exhausted
    "ghostcorp.example": ("gone", 0),  # 404 immediately: permanent
    "slowpoke-systems.example": ("slow_normalize", 0),  # the vendor is fine, the normaliser is too slow
}

_DOMAIN = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+")
_INDUSTRIES = [
    "Computer Software", "IT Svcs & Consulting", "Industrial Automation & Controls", "Renewable Energy & Environment",
    "Hospital & Health Care", "Banking / Financial Svcs", "Food & Beverages", "Semiconductors", "Logistics and Supply Chain",
    "Media Production",
]
_HEADCOUNTS = ["11-50", "51-200", "201-500", "501-1,000", "1,001-5,000"]
_CITIES = ["Austin, US", "Leeds, UK", "Rotterdam, NL", "Pune, IN", "Denver, US", "Lyon, FR", "Toronto, CA", "Porto, PT"]
INDUSTRY_TAXONOMY = {
    "Computer Software": "Software",
    "IT Svcs & Consulting": "Software",
    "Industrial Automation & Controls": "Industrial technology",
    "Renewable Energy & Environment": "Energy",
    "Hospital & Health Care": "Healthcare",
    "Banking / Financial Svcs": "Financial services",
    "Food & Beverages": "Consumer goods",
    "Semiconductors": "Hardware",
    "Logistics and Supply Chain": "Logistics",
    "Media Production": "Media",
}


def vendor_fetch(domain: str, attempt: int) -> dict:
    """The simulated vendor. Pure: the outcome depends only on ``domain`` and ``attempt`` (1-indexed)."""
    if not _DOMAIN.fullmatch(domain):
        raise PermanentVendorError(domain, 400, "malformed domain", attempt)
    behaviour, failing = SCENARIOS.get(domain, ("ok", 0))
    if behaviour == "gone":
        raise PermanentVendorError(domain, 404, "no such company", attempt)
    if behaviour == "down":
        raise TransientVendorError(domain, 503, "service unavailable", attempt)
    if behaviour == "timeout" and attempt <= failing:
        raise TransientVendorError(domain, 504, "gateway timeout", attempt)
    if behaviour == "rate_limit" and attempt <= failing:
        raise TransientVendorError(domain, 429, "too many requests", attempt)
    h = stable_int(domain)
    return {
        "legal_name": domain.split(".")[0].replace("-", " ").title() + " Inc.",
        "industry_raw": _INDUSTRIES[h % len(_INDUSTRIES)],
        "employees_raw": _HEADCOUNTS[(h // 10) % len(_HEADCOUNTS)],
        "hq_raw": _CITIES[(h // 100) % len(_CITIES)],
        "founded": 1985 + (h // 1000) % 35,
    }


def normalize_profile(profile: dict) -> dict:
    """Map messy vendor values onto our fields (pure, fast; the slowness is simulated in the node)."""
    low, _, high = profile["employees_raw"].replace(",", "").partition("-")
    city, _, country = profile["hq_raw"].partition(", ")
    return {
        "legal_name": profile["legal_name"],
        "industry": INDUSTRY_TAXONOMY.get(profile["industry_raw"], "Other"),
        "employees_min": int(low),
        "employees_max": int(high),
        "hq_city": city,
        "hq_country": country,
        "founded": profile["founded"],
    }


# --------------------------------------------------------------------------- progress numbering
class _Progress:
    """Process-local counter used ONLY to number progress messages ("looked up 7/13").

    Deliberately not graph state: progress is a view for the operator. Putting a counter in state would need a reducer
    and a checkpoint per tick, and parallel workers cannot otherwise know their rank. Bounded so it cannot leak.
    """

    def __init__(self, keep: int = 64):
        self._lock = threading.Lock()
        self._n: OrderedDict[str, int] = OrderedDict()
        self._keep = keep

    def tick(self, batch_id: str) -> int:
        with self._lock:
            self._n[batch_id] = self._n.get(batch_id, 0) + 1
            while len(self._n) > self._keep:
                self._n.popitem(last=False)
            return self._n[batch_id]

    def done(self, batch_id: str) -> None:
        with self._lock:
            self._n.pop(batch_id, None)


PROGRESS = _Progress()


def emit(event: str, message: str, **fields: Any) -> None:
    """Send one custom stream event (a no-op outside a run, so helpers stay unit-testable)."""
    try:
        get_stream_writer()({"event": event, "message": message, **fields})
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- state
class EnrichIn(TypedDict):
    count: NotRequired[int]  # size of the built-in sample batch (default 14, max 24)
    records: NotRequired[list[dict]]  # or your own: [{"company": ..., "domain": ..., "record_id": optional}]


class EnrichOut(TypedDict):
    batch_id: str
    summary: dict
    records: list[dict]


class EnrichState(TypedDict, total=False):
    count: int
    records: list[dict]
    batch_id: str
    domains: list[str]
    lookups: Annotated[list[dict], operator.add]  # one per unique domain (success or failed_*)
    normalized: Annotated[list[dict], operator.add]
    summary: dict


# --------------------------------------------------------------------------- sample batch
PROSPECTS = [
    ("Pinecrest Labs", "pinecrest-labs.example"),
    ("Harborline Freight", "harborline-freight.example"),
    ("Meridian Grid", "meridian-grid.example"),
    ("Ghostcorp", "ghostcorp.example"),
    ("Slowpoke Systems", "slowpoke-systems.example"),
    ("Acme Robotics (EU subsidiary)", "acme-robotics.example"),  # same domain as the listed company: a duplicate
    ("Quillstone Media", "quillstone-media.example"),
    ("Tidewater Logistics", "tidewater-logistics.example"),
    ("Orchard Bio", "orchard-bio.example"),
    ("Lantern Payments", "lantern-payments.example"),
    ("Fernhill Foods", "fernhill-foods.example"),
    ("Kestrel Aero", "kestrel-aero.example"),
    ("Juniper Textiles", "juniper-textiles.example"),
    ("Basalt Mining", "basalt-mining.example"),
    ("Willowbrook Health", "willowbrook-health.example"),
    ("Sable Software", "sable-software.example"),
]


def sample_records(count: int) -> list[dict]:
    """The eight listed companies from ``companies.csv`` followed by the made-up prospects."""
    listed = [
        {"company": c, "domain": re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-") + ".example"}
        for c in read_csv("companies.csv")["company"]
    ]
    rows = listed + [{"company": c, "domain": d} for c, d in PROSPECTS]
    return rows[: max(1, min(count, len(rows)))]


# --------------------------------------------------------------------------- nodes
def plan(state: EnrichState) -> dict:
    """Resolve the batch, group it by domain, and reset the per-run result lists."""
    raw = state["records"] if state.get("records") is not None else sample_records(14 if state.get("count") is None else int(state["count"]))
    if not raw:
        raise ValueError("no records to enrich")
    if len(raw) > MAX_RECORDS:
        raise ValueError(f"at most {MAX_RECORDS} records per run (got {len(raw)}); split the batch")
    records = [
        {
            "record_id": str(r.get("record_id") or f"R-{i:03d}"),
            "company": str(r.get("company", "")),
            "domain": str(r.get("domain", "")).strip().lower(),
        }
        for i, r in enumerate(raw, start=1)
    ]
    domains = list(dict.fromkeys(r["domain"] for r in records))
    batch_id = f"b-{uuid.uuid4().hex[:8]}"  # stamps every vendor call of this run (see after_lookup)
    emit("plan", f"{len(records)} records, {len(domains)} unique domains", records=len(records), unique_domains=len(domains))
    # Overwrite: on a reused thread the list channels would otherwise keep accumulating across runs.
    return {
        "records": records,
        "domains": domains,
        "batch_id": batch_id,
        "lookups": Overwrite([]),
        "normalized": Overwrite([]),
    }


def fan_out_lookups(state: EnrichState) -> list[Send]:
    """One lookup per UNIQUE domain; the payload carries what the worker needs, not the whole state."""
    return [Send("lookup", {"domain": d, "batch_id": state["batch_id"], "total": len(state["domains"])}) for d in state["domains"]]


def cache_key(payload: dict) -> str:
    """Cache on the domain only. The default key would hash batch_id too and never hit.

    Tolerates a payload without a domain: LangGraph's graph drawing (``get_graph()``, which Studio calls just to open the
    graph) probes every cache policy with an EMPTY payload. Real lookups always carry a domain, so their keys are unchanged.
    """
    domain = payload.get("domain", "") if isinstance(payload, dict) else ""
    return f"vendor-profile:{domain}"


def lookup(payload: dict, runtime: Runtime) -> dict:
    """One vendor lookup. Raises Transient/Permanent errors; RetryPolicy and the error handler decide what happens next."""
    domain, attempt = payload["domain"], runtime.execution_info.node_attempt
    try:
        profile = vendor_fetch(domain, attempt)
    except TransientVendorError as e:
        emit(
            "transient_error",
            f"{e.status} from {domain} on attempt {attempt}/{MAX_ATTEMPTS}" + (" - will retry" if attempt < MAX_ATTEMPTS else " - giving up"),
            domain=domain, status=e.status, attempt=attempt, will_retry=attempt < MAX_ATTEMPTS,
        )
        raise
    n = PROGRESS.tick(payload["batch_id"])
    emit("progress", f"looked up {n}/{payload['total']}: {domain}" + (f" (after {attempt - 1} retr{'y' if attempt == 2 else 'ies'})" if attempt > 1 else ""),
         done=n, total=payload["total"], domain=domain)
    return {"lookups": [{"domain": domain, "status": "ok", "profile": profile, "attempts": attempt, "batch_id": payload["batch_id"]}]}


def lookup_failed(payload: dict, error: NodeError) -> Command:
    """Runs once the retries are used up (or at once for a permanent error). Records the failure, keeps the batch going."""
    err = error.error
    if isinstance(err, PermanentVendorError):
        status, attempts, why = "failed_permanent", err.attempt, str(err)
    elif isinstance(err, TransientVendorError):
        status, attempts, why = "failed_retries_exhausted", err.attempt, str(err)
    else:  # a bug or an unexpected exception: still contained, but labelled so it is not mistaken for a vendor problem
        status, attempts, why = "failed_unexpected", 1, f"{type(err).__name__}: {err}"
    n = PROGRESS.tick(payload["batch_id"])
    emit("progress", f"looked up {n}/{payload['total']}: {payload['domain']} FAILED ({status})",
         done=n, total=payload["total"], domain=payload["domain"], failed=True)
    entry = {"domain": payload["domain"], "status": status, "error": why, "attempts": attempts, "batch_id": payload["batch_id"]}
    return Command(update={"lookups": [entry]}, goto="after_lookup")


def after_lookup(state: EnrichState) -> Command[Literal["normalize", "report"]]:
    """Join after the lookups: find the cache hits, then fan out normalisation for the successes."""
    batch = state["batch_id"]
    ok = [e for e in state["lookups"] if e["status"] == "ok"]
    for e in ok:
        if e["batch_id"] != batch:  # stamped by an earlier run: the cache replayed it, the vendor was not called
            n = PROGRESS.tick(batch)
            emit("cache_hit", f"looked up {n}/{len(state['domains'])}: {e['domain']} (served from cache)",
                 done=n, total=len(state["domains"]), domain=e["domain"])
    sends = [Send("normalize", {"domain": e["domain"], "profile": e["profile"], "batch_id": batch}) for e in ok]
    return Command(goto=sends or "report")  # `or "report"`: with nothing to normalise, nobody else would trigger the report


async def normalize(payload: dict) -> dict:
    """Normalise one profile through a simulated slow service. ``async`` because only async nodes can time out."""
    domain = payload["domain"]
    if SCENARIOS.get(domain, ("ok", 0))[0] == "slow_normalize":
        await asyncio.sleep(SLOW_NORMALIZE_S)  # cancelled by the TimeoutPolicy long before it finishes
    emit("normalized", f"normalized {domain}", domain=domain)
    return {"normalized": [{"domain": domain, "status": "normalized", "profile": normalize_profile(payload["profile"])}]}


def normalize_degraded(payload: dict, error: NodeError) -> Command:
    """Graceful degradation: keep the raw vendor values and say why."""
    timed_out = isinstance(error.error, NodeTimeoutError)
    reason = f"timed out after {NORMALIZE_TIMEOUT_S:g}s" if timed_out else f"{type(error.error).__name__}: {error.error}"
    emit("degraded", f"normalization unavailable for {payload['domain']} ({reason}); keeping raw vendor values", domain=payload["domain"])
    entry = {"domain": payload["domain"], "status": "degraded", "reason": reason, "profile": payload["profile"]}
    return Command(update={"normalized": [entry]}, goto="report")


def report(state: EnrichState) -> dict:
    """Join records with their per-domain results and summarise. Pure code: counts must be exact."""
    batch = state["batch_id"]
    lookups = {e["domain"]: e for e in state["lookups"]}
    normalized = {e["domain"]: e for e in state["normalized"]}
    first_with_domain: dict[str, str] = {}
    rows: list[dict] = []
    for rec in state["records"]:
        look = lookups.get(rec["domain"], {"status": "failed_unexpected", "error": "no lookup result", "attempts": 0, "batch_id": batch})
        row = {**rec, "attempts": look["attempts"], "from_cache": look["status"] == "ok" and look["batch_id"] != batch}
        if look["status"] == "ok":
            norm = normalized.get(rec["domain"])
            if norm and norm["status"] == "normalized":
                row.update(status="enriched", profile=norm["profile"])
            else:
                row.update(status="enriched_unnormalized", profile=look["profile"], note=(norm or {}).get("reason", "normalization missing"))
        else:
            row.update(status=look["status"], error=look["error"])
        dup_of = first_with_domain.setdefault(rec["domain"], rec["record_id"])
        if dup_of != rec["record_id"]:
            row["shares_lookup_with"] = dup_of
        rows.append(row)

    fresh = [e for e in lookups.values() if e["batch_id"] == batch]
    statuses = ("enriched", "enriched_unnormalized", "failed_permanent", "failed_retries_exhausted", "failed_unexpected")
    summary = {
        "records": len(rows),
        "unique_domains": len(lookups),
        "by_status": {s: sum(r["status"] == s for r in rows) for s in statuses},
        "vendor_calls": sum(e["attempts"] for e in fresh),
        "retries_used": sum(max(e["attempts"] - 1, 0) for e in fresh),
        "cache_hits": sum(e["status"] == "ok" and e["batch_id"] != batch for e in lookups.values()),
        "normalization_degraded": sum(e["status"] == "degraded" for e in normalized.values()),
    }
    emit("done", f"batch finished: {summary['by_status']}, {summary['retries_used']} retries, {summary['cache_hits']} cache hits")
    PROGRESS.done(batch)
    return {"summary": summary, "records": rows}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    """Pass ``checkpointer`` / ``store`` / ``cache`` as usual. Without a ``cache`` the graph owns an in-memory one."""
    compile_kwargs.setdefault("cache", InMemoryCache())
    g = StateGraph(EnrichState, input_schema=EnrichIn, output_schema=EnrichOut)
    g.add_node("plan", plan)
    g.add_node(
        "lookup",
        lookup,
        retry_policy=RetryPolicy(
            max_attempts=MAX_ATTEMPTS, initial_interval=RETRY_INITIAL_S, backoff_factor=2.0, retry_on=TransientVendorError
        ),
        cache_policy=CachePolicy(key_func=cache_key, ttl=CACHE_TTL_S),
        error_handler=lookup_failed,
        destinations=("after_lookup",),
    )
    g.add_node("after_lookup", after_lookup, destinations=("normalize", "report"))
    g.add_node(
        "normalize",
        normalize,
        timeout=TimeoutPolicy(run_timeout=NORMALIZE_TIMEOUT_S),
        error_handler=normalize_degraded,
        destinations=("report",),
    )
    g.add_node("report", report)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out_lookups, ["lookup"])
    g.add_edge("lookup", "after_lookup")
    g.add_edge("normalize", "report")
    g.add_edge("report", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

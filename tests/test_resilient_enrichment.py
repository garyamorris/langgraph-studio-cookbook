from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.testing import check_example
from deepdive.examples import resilient_enrichment as ex

FLAKY = "pinecrest-labs.example"  # attempts 1-2 time out, attempt 3 succeeds
RATE_LIMITED = "harborline-freight.example"  # attempt 1 is a 429
DOWN = "meridian-grid.example"  # every attempt 503
GONE = "ghostcorp.example"  # 404
SLOW = "slowpoke-systems.example"  # lookup fine, normaliser too slow


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    """Shrink the real waits (backoff, timeout) before any graph is built."""
    monkeypatch.setattr(ex, "RETRY_INITIAL_S", 0.001)
    monkeypatch.setattr(ex, "NORMALIZE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ex, "SLOW_NORMALIZE_S", 5.0)


@pytest.fixture
def vendor_calls(monkeypatch):
    """Count every call the simulated vendor receives, as (domain, attempt)."""
    calls: list[tuple[str, int]] = []
    real = ex.vendor_fetch

    def counting(domain, attempt):
        calls.append((domain, attempt))
        return real(domain, attempt)

    monkeypatch.setattr(ex, "vendor_fetch", counting)
    return calls


def recs(*domains):
    return [{"company": d.split(".")[0], "domain": d} for d in domains]


def run(graph, payload, config=None):
    """Async run: timeouts only exist on the async path."""
    return asyncio.run(graph.ainvoke(payload, config))


def by_domain(out):
    return {r["domain"]: r for r in out["records"]}


def test_studio_can_open_the_graph_without_running_it():
    """Studio calls get_graph / the schema endpoints just to display a graph. LangGraph's drawing code probes each node's
    cache policy with an EMPTY payload, so the cache key function must not require a domain (regression: KeyError)."""
    g = ex.build_graph()
    assert ex.cache_key({}) == "vendor-profile:" and ex.cache_key({"domain": "acme.example", "batch_id": "x"}) == "vendor-profile:acme.example"
    mermaid = g.get_graph(xray=1).draw_mermaid()
    assert "lookup" in mermaid and "normalize" in mermaid and "report" in mermaid
    g.get_graph()  # the non-xray view too
    assert "records" in g.get_input_jsonschema()["properties"]
    assert {"summary", "records", "batch_id"} <= set(g.get_output_jsonschema()["properties"])


# --------------------------------------------------------------------------- contract and the simulated vendor
def test_meets_the_example_contract():
    check_example(ex)


def test_vendor_schedule_is_a_pure_function_of_domain_and_attempt():
    assert ex.vendor_fetch("acme.example", 1) == ex.vendor_fetch("acme.example", 7)
    assert [_outcome(FLAKY, a) for a in (1, 2, 3, 4)] == ["504", "504", "ok", "ok"]
    assert [_outcome(RATE_LIMITED, a) for a in (1, 2)] == ["429", "ok"]
    assert [_outcome(DOWN, a) for a in (1, 4, 9)] == ["503", "503", "503"]
    assert _outcome(GONE, 1) == "404" and _outcome("not a domain", 1) == "400"
    assert issubclass(ex.TransientVendorError, ex.VendorError) and not issubclass(ex.PermanentVendorError, ex.TransientVendorError)


def _outcome(domain, attempt):
    try:
        ex.vendor_fetch(domain, attempt)
        return "ok"
    except ex.VendorError as e:
        return str(e.status)


def test_normalize_profile_maps_vendor_labels_to_our_taxonomy():
    out = ex.normalize_profile(
        {"legal_name": "X Inc.", "industry_raw": "IT Svcs & Consulting", "employees_raw": "1,001-5,000", "hq_raw": "Leeds, UK", "founded": 2001}
    )
    assert out["industry"] == "Software" and (out["employees_min"], out["employees_max"]) == (1001, 5000)
    assert (out["hq_city"], out["hq_country"]) == ("Leeds", "UK")


# --------------------------------------------------------------------------- retry
def test_transient_failures_are_retried_with_the_attempt_number_until_success(vendor_calls):
    out = run(ex.build_graph(), {"records": recs(FLAKY)})
    row = out["records"][0]
    assert row["status"] == "enriched" and row["attempts"] == 3
    assert vendor_calls == [(FLAKY, 1), (FLAKY, 2), (FLAKY, 3)], "node_attempt reached the vendor: 1, 2, 3"
    assert out["summary"]["retries_used"] == 2 and out["summary"]["vendor_calls"] == 3


def test_exhausted_retries_stop_at_max_attempts_then_hit_the_error_handler(vendor_calls):
    out = run(ex.build_graph(), {"records": recs(DOWN)})
    row = out["records"][0]
    assert row["status"] == "failed_retries_exhausted" and row["attempts"] == ex.MAX_ATTEMPTS == 4
    assert len(vendor_calls) == 4 and "503" in row["error"]


def test_permanent_errors_are_not_retried(vendor_calls):
    out = run(ex.build_graph(), {"records": recs(GONE, "not a domain")})
    rows = by_domain(out)
    assert rows[GONE]["status"] == "failed_permanent" and rows[GONE]["attempts"] == 1
    assert rows["not a domain"]["status"] == "failed_permanent" and "400" in rows["not a domain"]["error"]
    assert sorted(vendor_calls) == sorted([(GONE, 1), ("not a domain", 1)]), "one call each, no retries"
    assert out["summary"]["retries_used"] == 0


def test_one_bad_record_does_not_sink_the_batch():
    batch = recs("fine-one.example", GONE, DOWN, FLAKY, "fine-two.example", "not a domain")
    out = run(ex.build_graph(), {"records": batch})
    rows = by_domain(out)
    assert len(out["records"]) == 6, "every record gets a result"
    assert out["summary"]["by_status"] == {
        "enriched": 3, "enriched_unnormalized": 0, "failed_permanent": 2, "failed_retries_exhausted": 1, "failed_unexpected": 0,
    }
    assert rows["fine-one.example"]["profile"]["industry"] and rows[FLAKY]["status"] == "enriched"


def test_an_unexpected_exception_is_contained_and_labelled_not_retried(monkeypatch):
    calls = []

    def buggy(domain, attempt):
        calls.append(domain)
        raise RuntimeError("parser bug")

    monkeypatch.setattr(ex, "vendor_fetch", buggy)
    out = run(ex.build_graph(), {"records": recs("a.example", "b.example")})
    assert {r["status"] for r in out["records"]} == {"failed_unexpected"}
    assert "RuntimeError: parser bug" in out["records"][0]["error"]
    assert sorted(calls) == ["a.example", "b.example"], "RetryPolicy(retry_on=Transient) did not retry a RuntimeError"


# --------------------------------------------------------------------------- duplicates and the cache
def test_duplicate_domains_in_one_batch_call_the_vendor_once(vendor_calls):
    batch = [
        {"record_id": "A", "company": "Acme", "domain": "acme.example"},
        {"record_id": "B", "company": "Acme EU", "domain": "ACME.example "},  # same domain after normalisation
    ]
    out = run(ex.build_graph(), {"records": batch})
    assert vendor_calls == [("acme.example", 1)]
    a, b = out["records"]
    assert a["profile"] == b["profile"] and b["shares_lookup_with"] == "A" and "shares_lookup_with" not in a
    assert out["summary"]["records"] == 2 and out["summary"]["unique_domains"] == 1


def test_repeat_lookups_in_a_later_run_are_served_from_the_cache(vendor_calls):
    g = ex.build_graph()  # owns one cache for both runs
    first = run(g, {"records": recs("acme.example", FLAKY)})
    assert len(vendor_calls) == 1 + 3 and first["summary"]["cache_hits"] == 0
    vendor_calls.clear()

    second = run(g, {"records": [{"record_id": "NEW-ID", "company": "Acme again", "domain": "acme.example"}, *recs(FLAKY)]})
    assert vendor_calls == [], "both domains came from the cache: same domain, different record id, different batch id"
    assert second["summary"]["cache_hits"] == 2 and second["summary"]["vendor_calls"] == 0 and second["summary"]["retries_used"] == 0
    assert all(r["from_cache"] for r in second["records"])
    assert second["records"][0]["profile"] == first["records"][0]["profile"]


def test_failures_are_never_cached_so_a_rerun_retries_only_them(vendor_calls):
    g = ex.build_graph()
    run(g, {"records": recs("fine.example", DOWN, GONE)})
    vendor_calls.clear()
    out = run(g, {"records": recs("fine.example", DOWN, GONE)})
    assert sorted(set(d for d, _ in vendor_calls)) == sorted([DOWN, GONE]), "only the failures went back to the vendor"
    assert len([c for c in vendor_calls if c[0] == DOWN]) == 4
    assert out["summary"]["cache_hits"] == 1


def test_each_graph_without_an_explicit_cache_has_its_own_and_an_explicit_one_is_honoured(vendor_calls):
    run(ex.build_graph(), {"records": recs("acme.example")})
    run(ex.build_graph(), {"records": recs("acme.example")})
    assert len(vendor_calls) == 2, "separate graphs, separate in-memory caches"
    vendor_calls.clear()
    shared = InMemoryCache()
    run(ex.build_graph(cache=shared), {"records": recs("acme.example")})
    run(ex.build_graph(cache=shared), {"records": recs("acme.example")})
    assert len(vendor_calls) == 1, "an explicitly passed cache is shared"


def test_cache_entries_expire_after_the_ttl(vendor_calls, monkeypatch):
    monkeypatch.setattr(ex, "CACHE_TTL_S", 1)
    g = ex.build_graph()
    run(g, {"records": recs("acme.example")})
    time.sleep(1.2)
    run(g, {"records": recs("acme.example")})
    assert len(vendor_calls) == 2


# --------------------------------------------------------------------------- timeout and graceful degradation
def test_slow_normalisation_times_out_and_the_record_degrades_to_raw_values():
    started = time.perf_counter()
    out = run(ex.build_graph(), {"records": recs(SLOW, "fine.example")})
    elapsed = time.perf_counter() - started
    rows = by_domain(out)
    assert elapsed < 2.0, "the 5 s simulated call was cancelled by the 0.05 s TimeoutPolicy"
    assert rows[SLOW]["status"] == "enriched_unnormalized"
    assert "timed out" in rows[SLOW]["note"] and "industry_raw" in rows[SLOW]["profile"], "raw vendor values were kept"
    assert rows["fine.example"]["status"] == "enriched" and "industry" in rows["fine.example"]["profile"]
    assert out["summary"]["normalization_degraded"] == 1


def test_a_sync_invoke_cannot_enforce_timeouts_and_degrades_with_that_reason():
    out = ex.build_graph().invoke({"records": recs("fine.example")})
    row = out["records"][0]
    assert row["status"] == "enriched_unnormalized" and "async" in row["note"]


def test_when_every_lookup_fails_the_report_still_runs():
    out = run(ex.build_graph(), {"records": recs(GONE, DOWN)})
    assert out["summary"]["by_status"]["failed_permanent"] == 1 and out["summary"]["by_status"]["failed_retries_exhausted"] == 1
    assert len(out["records"]) == 2


# --------------------------------------------------------------------------- streaming
def collect_custom(graph, payload, config=None):
    async def go():
        return [ev async for ev in graph.astream(payload, config, stream_mode="custom")]

    return asyncio.run(go())


def test_custom_stream_reports_plan_errors_progress_degradation_and_the_final_summary():
    events = collect_custom(ex.build_graph(), {"records": recs("fine.example", FLAKY, GONE, SLOW)})
    kinds = [e["event"] for e in events]
    assert kinds[0] == "plan" and kinds[-1] == "done"
    assert events[0]["unique_domains"] == 4

    transient = [e for e in events if e["event"] == "transient_error"]
    assert [(e["attempt"], e["will_retry"]) for e in transient] == [(1, True), (2, True)] and {e["status"] for e in transient} == {504}

    progress = [e for e in events if e["event"] == "progress"]
    assert sorted(e["done"] for e in progress) == [1, 2, 3, 4] and {e["total"] for e in progress} == {4}
    assert sum(1 for e in progress if e.get("failed")) == 1
    assert any("looked up 4/4" in e["message"] for e in progress)

    assert [e["domain"] for e in events if e["event"] == "degraded"] == [SLOW]
    assert {e["domain"] for e in events if e["event"] == "normalized"} == {"fine.example", FLAKY}
    assert "1 retries" not in events[-1]["message"] and "2 retries" in events[-1]["message"]


def test_synchronous_stream_also_yields_custom_progress_events():
    # graph.stream() works for the lookup stage (normalisation needs the async runtime, see the sync-invoke test above)
    events = list(ex.build_graph().stream({"records": recs("fine.example", FLAKY)}, stream_mode="custom"))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "plan" and kinds.count("transient_error") == 2 and kinds.count("progress") == 2


def test_second_run_streams_cache_hit_events_and_numbers_them_into_the_same_progress_count():
    g = ex.build_graph()
    run(g, {"records": recs("fine.example", "other.example")})
    events = collect_custom(g, {"records": recs("fine.example", "other.example", "new.example")})
    hits = [e for e in events if e["event"] == "cache_hit"]
    assert {e["domain"] for e in hits} == {"fine.example", "other.example"}
    numbered = sorted(e["done"] for e in events if e["event"] in ("progress", "cache_hit"))
    assert numbered == [1, 2, 3], "the live progress reaches 3/3 although two of the three never called the vendor"


def test_progress_counters_are_cleaned_up_after_a_run():
    before = len(ex.PROGRESS._n)
    run(ex.build_graph(), {"records": recs("fine.example", GONE)})
    assert len(ex.PROGRESS._n) == before


# --------------------------------------------------------------------------- state hygiene and input handling
def test_a_reused_thread_does_not_accumulate_results_across_runs():
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:6]}"}}
    first = run(g, {"records": recs("a.example", "b.example", "c.example")}, cfg)
    second = run(g, {"records": recs("d.example")}, cfg)
    assert first["summary"]["unique_domains"] == 3
    assert second["summary"]["unique_domains"] == 1 and len(second["records"]) == 1, "Overwrite reset the list channels in plan"
    assert second["batch_id"] != first["batch_id"]


def test_input_validation():
    g = ex.build_graph()
    with pytest.raises(ValueError, match="at most 60"):
        run(g, {"records": recs(*[f"d{i}.example" for i in range(61)])})
    with pytest.raises(ValueError, match="no records"):
        run(g, {"records": []})


def test_default_sample_batch_matches_the_documented_numbers(vendor_calls):
    g = ex.build_graph()
    out = run(g, {"count": 14})
    s = out["summary"]
    assert (s["records"], s["unique_domains"], s["vendor_calls"], s["retries_used"], s["cache_hits"]) == (14, 13, 21, 8, 0)
    assert s["by_status"] == {
        "enriched": 11, "enriched_unnormalized": 1, "failed_permanent": 1, "failed_retries_exhausted": 1, "failed_unexpected": 0,
    }
    rerun = run(g, {"count": 14})["summary"]
    assert (rerun["vendor_calls"], rerun["retries_used"], rerun["cache_hits"]) == (5, 3, 11)
    assert 0 < len(ex.sample_records(999)) <= 24 and len(ex.sample_records(0)) == 1

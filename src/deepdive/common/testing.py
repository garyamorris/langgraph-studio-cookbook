"""Helpers for driving an example end to end in tests and scripts.

``run_example(module)`` builds the graph with an in-memory checkpointer/store/cache, runs the module's
``EXAMPLE_INPUT``, and answers every ``interrupt()`` with the next value from ``EXAMPLE_RESUMES``.
"""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

from langchain_core.messages import BaseMessage
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command


def lint_openai_strict(schema: type[Any]) -> list[str]:
    """Return problems that would make OpenAI's *strict* structured-output mode reject ``schema``.

    Offline runs use a fake model, so they cannot notice these; this lint closes that gap. Checked: free-form
    ``dict`` fields (they become ``additionalProperties: true``), nesting deeper than 8 levels, and > 200 properties.
    """
    from openai.lib._pydantic import to_strict_json_schema

    problems: list[str] = []
    js = to_strict_json_schema(schema)
    defs = js.get("$defs", {})
    n_props = 0

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal n_props
        if isinstance(node, dict):
            if depth > 8:
                problems.append(f"{path}: nesting deeper than 8 levels")
                return
            if node.get("type") == "object" or "properties" in node:
                n_props += len(node.get("properties", {}))
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: free-form object (dict field?) - strict mode needs fixed properties")
            if "$ref" in node:
                walk(defs.get(node["$ref"].split("/")[-1], {}), f"{path}->{node['$ref'].split('/')[-1]}", depth + 1)
            for k, v in node.items():
                if k in ("properties", "$defs"):
                    for pk, pv in v.items():
                        walk(pv, f"{path}.{pk}", depth + 1)
                elif k in ("items", "anyOf", "oneOf", "allOf"):
                    for i, item in enumerate(v if isinstance(v, list) else [v]):
                        walk(item, f"{path}[{k}{i}]", depth + 1)
        return

    walk({k: v for k, v in js.items() if k != "$defs"}, schema.__name__, 0)
    if n_props > 200:
        problems.append(f"{schema.__name__}: {n_props} properties (> 200)")
    return problems


@dataclass
class RunResult:
    output: dict[str, Any]
    interrupts: list[Any] = field(default_factory=list)  # payloads of every interrupt that fired, in order
    graph: Any = None
    config: dict[str, Any] = field(default_factory=dict)


def _split(out: Any) -> tuple[dict[str, Any], list[Any]]:
    """Normalise invoke() output (dict with __interrupt__, or a GraphOutput) -> (values, interrupts)."""
    if isinstance(out, dict):
        pending = out.get("__interrupt__") or []
        return {k: v for k, v in out.items() if k != "__interrupt__"}, list(pending)
    value = getattr(out, "value", out)
    return (value if isinstance(value, dict) else {"value": value}), list(getattr(out, "interrupts", []) or [])


def run_graph(
    graph: Any,
    payload: Any,
    resumes: list[Any] | tuple[Any, ...] = (),
    *,
    config: dict[str, Any] | None = None,
    max_interrupts: int = 10,
) -> RunResult:
    """Invoke a compiled graph/entrypoint, feeding ``resumes`` to successive interrupts."""
    cfg = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 200}
    if config:
        cfg["configurable"].update(config.get("configurable", {}))
        cfg.update({k: v for k, v in config.items() if k != "configurable"})
    out, pending = _split(graph.invoke(payload, cfg))
    seen: list[Any] = []
    queue = list(resumes)
    while pending:
        if len(seen) >= max_interrupts:
            raise AssertionError(f"more than {max_interrupts} interrupts - infinite approval loop?")
        seen.extend(i.value if hasattr(i, "value") else i for i in pending)
        if not queue:
            return RunResult(out, seen, graph, cfg)  # paused: caller inspects the interrupt
        out, pending = _split(graph.invoke(Command(resume=queue.pop(0)), cfg))
    return RunResult(out, seen, graph, cfg)


def run_example(module: ModuleType | str, *, resumes: list[Any] | None = None) -> RunResult:
    """Run an example module's ``EXAMPLE_INPUT`` (and ``EXAMPLE_RESUMES``) to completion."""
    from deepdive.common.embeddings import make_store

    mod = importlib.import_module(module) if isinstance(module, str) else module
    # make_store() has the same semantic index that langgraph.json configures for Studio's store.
    graph = mod.build_graph(checkpointer=InMemorySaver(), store=make_store(), cache=InMemoryCache())
    return run_graph(graph, mod.EXAMPLE_INPUT, resumes if resumes is not None else getattr(mod, "EXAMPLE_RESUMES", ()))


# --------------------------------------------------------------------------- reuse of a thread
# Studio keeps submitting runs on the SAME thread, so state a run leaves behind (accumulating reducers, fields the
# entry node does not overwrite) must not change the next run: its result has to equal a fresh thread's.
VOLATILE_KEYS = frozenset({"latency_s", "batch_id"})  # wall-clock timings and per-run random ids; nothing else may differ


def normalise(value: Any, volatile: frozenset[str] = VOLATILE_KEYS) -> Any:
    """Make an output comparable between runs: drop the volatile keys and reduce messages to type + content."""
    if isinstance(value, dict):
        return {k: normalise(v, volatile) for k, v in value.items() if k not in volatile}
    if isinstance(value, (list, tuple)):
        return [normalise(v, volatile) for v in value]
    if isinstance(value, BaseMessage):
        return {"type": value.type, "content": normalise(value.content, volatile)}
    return value


def run_twice(
    build: Callable[[], Any],
    first: Any,
    second: Any,
    *,
    first_resumes: list[Any] | tuple[Any, ...] = (),
    second_resumes: list[Any] | tuple[Any, ...] = (),
) -> tuple[RunResult, RunResult]:
    """Run ``first`` then ``second`` on ONE thread, and ``second`` alone on a brand-new graph and thread.

    ``build()`` returns a compiled graph with its own checkpointer (and store, if the example uses one). Returns
    ``(second run on the reused thread, second run on a fresh thread)``: for an example that resets its per-run
    state the two outputs are equal (compare them through :func:`normalise`).
    """
    graph = build()
    cfg = {"configurable": {"thread_id": f"reused-{uuid.uuid4().hex[:8]}"}}
    opening = run_graph(graph, first, first_resumes, config=cfg)
    assert not graph.get_state(opening.config).next, "the first run did not finish (an interrupt was left unanswered)"
    return run_graph(graph, second, second_resumes, config=cfg), run_graph(build(), second, second_resumes)


def run_example_twice(module: ModuleType | str, *, resumes: list[Any] | None = None) -> tuple[dict, dict]:
    """The reuse contract for an example: ``EXAMPLE_INPUT`` (answering ``EXAMPLE_RESUMES``) twice on one thread.

    Returns ``(second run on the reused thread, run on a fresh thread)``, each as ``{"output": ..., "interrupts": ...}``
    after :func:`normalise`: the pauses matter as much as the result (a run that skips its approval is the worst leak).
    The graph gets a checkpointer and a store but no node cache: a node cache is a deliberate cross-run memo.
    """
    from deepdive.common.embeddings import make_store

    mod = importlib.import_module(module) if isinstance(module, str) else module
    answers = resumes if resumes is not None else getattr(mod, "EXAMPLE_RESUMES", ())
    reused, fresh = run_twice(
        lambda: mod.build_graph(checkpointer=InMemorySaver(), store=make_store()),
        mod.EXAMPLE_INPUT,
        mod.EXAMPLE_INPUT,
        first_resumes=answers,
        second_resumes=answers,
    )
    return (
        {"output": normalise(reused.output), "interrupts": normalise(reused.interrupts)},
        {"output": normalise(fresh.output), "interrupts": normalise(fresh.interrupts)},
    )


REQUIRED_SECTIONS = ("PROBLEM", "DESIGN", "LANGGRAPH FEATURES", "TRY IT IN STUDIO")


def check_example(module: ModuleType | str) -> RunResult:
    """The quality contract every example must meet. Raises AssertionError describing the first violation.

    Checks: documentation sections, ``META``, the ``build_graph``/``graph``/``EXAMPLE_INPUT`` surface, that the
    example's declared datasets exist, that it runs end to end offline (answering interrupts from
    ``EXAMPLE_RESUMES``), and that every Pydantic schema it sent to ``with_structured_output`` is compatible
    with OpenAI's strict structured-output mode. Used by tests/test_all_examples.py and by each example's own test.
    """
    from deepdive.common.config import data_dir
    from deepdive.common.fakes import SEEN_SCHEMAS
    from deepdive.common.meta import CATEGORY_ORDER, ExampleMeta

    mod = importlib.import_module(module) if isinstance(module, str) else module
    name = mod.__name__.rsplit(".", 1)[-1]
    doc = mod.__doc__ or ""
    assert doc.count("PROBLEM") >= 1, f"{name}: docstring must contain a PROBLEM section"
    for section in REQUIRED_SECTIONS:
        assert section in doc, f"{name}: docstring is missing a {section} section"
    meta = getattr(mod, "META", None)
    assert isinstance(meta, ExampleMeta), f"{name}: define META = ExampleMeta(...)"
    assert meta.id == name, f"{name}: META.id ({meta.id!r}) must equal the module name"
    assert meta.category in CATEGORY_ORDER, f"{name}: unknown category {meta.category!r}"
    assert meta.problem.strip().endswith(".") and len(meta.problem) > 40, f"{name}: META.problem must be one real sentence"
    assert len(meta.features) >= 3, f"{name}: list at least 3 LangGraph features in META.features"
    assert callable(getattr(mod, "build_graph", None)), f"{name}: define build_graph(**compile_kwargs)"
    assert getattr(mod, "graph", None) is not None, f"{name}: define module-level `graph = build_graph()`"
    assert getattr(mod, "EXAMPLE_INPUT", None), f"{name}: define EXAMPLE_INPUT"
    for ds in meta.datasets:
        assert (data_dir() / ds).exists(), f"{name}: META lists missing dataset {ds!r}"

    # Studio calls these when it merely OPENS a graph (before any run): topology for the canvas, and the input/output
    # schemas for the form. They probe nodes with empty state, so e.g. a cache-key function that assumes a field
    # exists would break the graph view even though runs work.
    try:
        mod.graph.get_graph(xray=1).draw_mermaid()
        mod.graph.get_graph().to_json()
        mod.graph.get_input_jsonschema()
        mod.graph.get_output_jsonschema()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{name}: Studio could not open the graph ({type(e).__name__}: {e})") from e

    SEEN_SCHEMAS.clear()
    result = run_example(mod)
    assert result.output, f"{name}: the example produced no output"
    problems = [p for s in SEEN_SCHEMAS for p in lint_openai_strict(s)]
    assert not problems, f"{name}: schemas incompatible with OpenAI strict structured output: {problems}"
    return result

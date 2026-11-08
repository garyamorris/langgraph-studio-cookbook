# langgraph-deepdive

A playground of **problem-driven LangGraph examples** for exploring architecture patterns in **LangGraph Studio**.

Each example is a designed solution to a concrete problem (escalate support tickets to a human when a keyword lexicon or the model flags a safety or privacy risk, or when the model call fails - recall is not measured on labelled data - rather than answer a
safety incident, gate refunds behind a human, keep a 100-turn chat coherent, ...), not a feature demo. Every module
opens with the problem, the design rationale, the graph, the LangGraph features it uses *and why*, and the exact input
to paste into Studio. Models are OpenAI chat models; the **OpenAI Decisions API** (`client.decisions.create`) is shown
in the few examples where calibrated probabilities are the right tool.

* **[Example catalog](docs/CATALOG.md)** - every example, grouped, with an index by LangGraph feature
* **[Per-example pages](docs/examples/)** - problem, design, diagram of the compiled graph, input to paste
* **[Authoring guide](docs/AUTHORING.md)** - the conventions and the verified LangGraph 1.2.20 facts behind them
* **[Datasets](data/README.md)** - the synthetic "Lumen Home" universe the examples run on

## Quick start

```bash
uv sync                      # Python 3.12 + all dependencies
cp .env.example .env         # then put your OPENAI_API_KEY in .env
uv run langgraph dev         # starts the local server on :2024
```

Open Studio against it: <https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024>, pick a graph from the
dropdown, paste the input from its page in `docs/examples/` (or from the `TRY IT IN STUDIO` section of its docstring),
and run.

> Always start the server with `uv run langgraph dev`. A globally installed `langgraph` binary runs in a different
> Python that cannot import this project and fails with `No module named 'deepdive'`.

## Offline mode (free, deterministic)

Set `DEEPDIVE_OFFLINE=1` - or simply leave `OPENAI_API_KEY` unset - and every example runs with deterministic stand-in
models and a stub Decisions API instead of calling OpenAI. This exists so the whole playground can be tested and
explored without spending money. Treat offline output as **plumbing, not intelligence**: structure, routing,
interrupts, streaming and state all behave for real, but model-written text and classifications are placeholders.
Every example stamps `mode: "offline" | "live"` into its output so you always know which one you are looking at.

## Where to start

| I want to see... | Open |
|---|---|
| a clean routing workflow whose safety path is a deterministic keyword lexicon unioned with the model's risk flag, failing closed on model error (recall not measured on labelled data) | `ticket_router` |
| a human approving money before it moves | `refund_approval`, `ops_copilot` |
| streaming (progress events, tokens, subgraphs) | `live_digest_stream` |
| short-term and long-term memory | `conversation_memory`, `memory_concierge` |
| the OpenAI Decisions API | `credit_decision_gate`, `classifier_bakeoff`, `subject_line_tournament`, `corrective_rag` |
| multi-agent architectures | `research_supervisor`, `support_swarm`, `portfolio_desk_lite` |
| the Functional API, retries, caching, error handlers | `functional_moderation`, `resilient_enrichment` |
| time travel (fork a run and recompute) | `pricing_whatif` |

The complete, generated list is in **[docs/CATALOG.md](docs/CATALOG.md)**.

## Repository layout

```
langgraph.json              Studio registry: one entry per example + the store's semantic index
src/deepdive/
  common/                   shared layer: models, offline stand-ins, Decisions wrapper, datasets, retrieval, embeddings
  examples/<id>.py          one file per example (problem -> design -> graph), `graph = build_graph()`
data/                       synthetic datasets + knowledge bases (see data/README.md)
tests/                      per-example behaviour tests + a contract test over every registered example
scripts/                    make_datasets, gen_docs, verify_studio, check_example
docs/                       generated catalog and example pages, plus the authoring guide
```

## Working with it

```bash
.venv/bin/python -m pytest                       # whole suite, offline, no API calls
.venv/bin/python scripts/check_example.py <id>   # contract + offline run of one example, prints its output
.venv/bin/python scripts/gen_docs.py             # regenerate docs/CATALOG.md and docs/examples/*
.venv/bin/python scripts/verify_studio.py        # boot a real `langgraph dev` and run every example over HTTP
.venv/bin/python scripts/verify_studio.py --live # same, but with real OpenAI calls (costs money)
.venv/bin/python scripts/live_smoke.py           # dry run; add --go to run examples against real OpenAI (costs money)
.venv/bin/python scripts/live_smoke.py --go --all --save out/   # every example, keeping each full output to read (PASS only means "did not crash")
.venv/bin/python scripts/make_datasets.py        # regenerate the seeded datasets
```

## How it was verified

* **1,121 offline tests** (no API calls): per-example behaviour with scripted stand-ins (branches taken, loop caps firing,
  interrupt/resume, idempotency, privacy filters), plus contract tests over every registered example, including that a
  second run on the same thread behaves like a fresh one (Studio keeps submitting on one thread).
* **A real `langgraph dev` server**: all 23 graphs load, Studio's graph and schema endpoints work for each, and every
  example (including the five interrupt/resume flows) runs through the HTTP API (`scripts/verify_studio.py`).
* **Live OpenAI**: every example was run against the real API at least once during development
  (`scripts/live_smoke.py`), which found and fixed real problems offline tests could not (for example a gate question
  that made the trading desk stand aside on every name). Model output varies between runs, so treat the live
  transcripts quoted in docstrings as one observed run, not a guarantee. `scripts/live_smoke.py --go --all --save out/`
  keeps every full output to read: PASS only means "did not crash".
* **An independent expert review** (LangGraph, solutions architecture, applied AI, Studio/test quality) found real
  problems that the above had missed, now fixed: state leaking between runs on a reused thread, an unrecoverable bad
  resume and a non-fail-closed approve rule in `credit_decision_gate`, an overstated safety guarantee in
  `ticket_router`, and a live-only refusal (a long guard-rail sentence made the model refuse the repayment question on a
  strong application). Still open: no capstone that composes several patterns, and a few missing patterns (evaluation
  harness, background work, guardrail layer).

## Design notes worth knowing

* **Models are created inside nodes, never at import**, so Studio can load every graph without an API key.
* **Exported graphs have no checkpointer.** Studio / `langgraph dev` supply persistence and a store; tests and scripts
  pass `checkpointer=` / `store=` / `cache=` into each module's `build_graph(**kwargs)`.
* **Structured-output schemas are checked for OpenAI strict-mode compatibility** by the test suite (offline runs
  cannot catch free-form `dict` fields, which OpenAI rejects).
* **Studio's store has a semantic index** (`store.index` in `langgraph.json`, 256-dim embeddings via
  `common/embeddings.py`: OpenAI when live, hashed bag-of-words offline), so `store.search(ns, query=...)` works in
  the memory examples.
* **Recursion limit:** in LangGraph 1.2 the default is 10007, so every loop in these examples carries its own explicit
  cap in state, and the tests prove the caps fire.
* **`langgraph dev` persists state** under `.langgraph_api/` (threads, and Store items such as saved memories and the
  refund ledger). If an idempotent example reports `already_paid` or remembers things you did not expect, delete that
  directory for a fresh start.

## Data and disclaimers

All data is synthetic. "Lumen Home" and the eight listed companies (`ACMR`, `BORE`, ...) are fictional. Nothing here is
investment, credit or legal advice; the trading and lending examples demonstrate architecture only.

# Ask the shop database (guarded text-to-SQL)

> Let non-technical staff query the shop database in plain English without hallucinated values, wrong joins or unsafe SQL ever reaching the database.

| | |
|---|---|
| **Example id** | `sql_analyst` |
| **Category** | Agents & tools |
| **Difficulty** | intermediate |
| **Models** | OpenAI chat |
| **Data** | `store.sqlite` |
| **LangGraph features** | guarded repair loop, conditional edges with named repair edge, reducer-accumulated attempt history, input/output schemas, deterministic safety layer around a model, structured output |
| **Source** | [`src/deepdive/examples/sql_analyst.py`](../../src/deepdive/examples/sql_analyst.py) |

---

## Problem

Non-technical staff want answers from the shop database ("which region had the most returned orders last
month?") without writing SQL. Naive text-to-SQL fails in three ways that matter. The model invents values
('online' when the channel is really called 'web'), so the query runs fine and silently returns nothing. It
writes a wrong join and returns confident but wrong numbers. And because its output is executed, a careless or
hostile prompt can produce DELETE / DROP / PRAGMA / ATTACH, a second stacked statement, a query that reads
customers' e-mail addresses, or one that never finishes.

## Design

The model writes SQL; code decides whether that SQL may run. Everything that can be decided without a model is
decided without one:

  * `inspect_schema` builds the model's context from the live database: the CREATE statements, the distinct
    values of every low-cardinality text column (status, channel, region, segment, category, product name), the
    date range, and a few house rules (region lives on customers, count DISTINCT orders after joining items,
    what "revenue" means). The model copies real literals instead of guessing them.
  * `write_sql` (structured output: `sql` + a one-line `rationale`) is the only creative step.
  * `guard` is deterministic and deny-by-default. It strips comments, requires exactly one statement that
    starts with SELECT or WITH, and then compiles the statement under SQLite's *authorizer*. The authorizer is
    called by SQLite itself for every operation, table, column and function the statement would touch, so it
    cannot be fooled by a CTE column called `delete_flag` or a string literal containing "DROP TABLE" the way a
    substring check can. Only SELECT / READ / FUNCTION / RECURSIVE actions, the four shop tables, an allow-list
    of functions and non-restricted columns pass. The guard also wraps the query as
    `SELECT * FROM (<query>) LIMIT 101` so the row cap cannot be forgotten or overridden.
  * `execute` runs on the read-only connection with a progress handler (time limit) and fetches at most
    100 rows, so even a legal but runaway query is stopped.
  * Anything that goes wrong - a guard rejection, a SQL error, a timeout, or an empty result (the classic sign
    of a wrong literal) - takes the `repair` edge back to `write_sql` with the exact problem and all earlier
    attempts in the prompt. The loop is capped at 3 attempts. If the model repeats the *same* SQL after an
    empty result, the loop stops early: that is the model standing by "there really are no such rows".
  * `answer` writes prose from the returned rows only and appends the SQL that produced them, so staff can see
    what was asked. When the loop gives up, `give_up` writes a fixed message instead: a model is never asked
    to explain results that do not exist.

## Graph

```text
START -> inspect_schema -> write_sql -> guard --ok--> execute --ok--> answer -> END
                               ^           |             |
                               +- repair --+-------------+    (guard rejection, SQL error, timeout, empty result)
                                           |             |
                                           +-- give_up --+    (3 attempts used, or the same empty query repeated)
                                                 |
                                                 v
                                                END
```

## LangGraph features

* conditional edges with an explicit path map, so the `repair` edge is visible and named in Studio
* a bounded repair loop: `attempts` lives in state, `failed_attempts` accumulates through a reducer
  (`operator.add`; `inspect_schema` resets it with `Overwrite([])` for each new question), and the cap is
  enforced by the router, not by hoping the model stops
* separate `input_schema` / `output_schema`: Studio shows one input box and a clean result
* a deterministic safety layer wrapped around a model call (the pattern, not any one API, is the point)
* structured output (Pydantic) for the SQL draft

## Models and data

OpenAI chat: the default model writes the SQL (structured output), the fast model writes the final prose. No
Decisions API. Data: `store.sqlite` (customers, products, orders, order_items), opened read-only.
Simulated/offline: the offline stand-in model cannot write SQL, so an offline run produces placeholder text, the
guard rejects it three times and the run ends with `status: "failed"`. That is the safety net working, not a
bug; set OPENAI_API_KEY to see real queries. "Today" is pinned to the day after the newest order in the
database (2026-11-01), because the database is a frozen snapshot.

## Try it in Studio

Input `{"question": "Which region had the most returned orders in the last 6 months?"}`. Open the
`inspect_schema` output to see the context the model gets (note the VALUES list), then `write_sql` for the SQL
and its rationale, `guard` for the query actually run, and the final `answer` / `rows`.
Then try questions that exercise the safety net:
  * `"How many orders came in through the online channel?"` - 'online' is not a value in the database; read the
    `rationale` in `write_sql` to see which real channel it was mapped to (and, if it guessed a literal that
    does not exist, the empty-result repair that follows).
  * `"Delete all cancelled orders"` or `"Show me every customer's email address"` - a model may refuse by
    writing a harmless SELECT instead, but if it writes a DELETE, or reads `customers.email`, `guard` rejects
    it, the reason appears under `failed_attempts` and the repair loop asks again; the database is never
    touched either way.
  * `"Which region had the most returned orders last month?"` - a three-way tie; the answer should say so.

## Extend it

* Add a column-level description table (a semantic layer) and pass only the relevant tables to the model.
* Add `interrupt()` before `execute` for queries that touch customers, so a human approves PII access.
* Add a second model call that checks the SQL against the question (a "does this query answer the question?"
  critic), or ask the Decisions API for a calibrated probability and require a threshold.
Production hardening not done here: per-user row-level permissions, query cost estimates (EXPLAIN QUERY PLAN),
caching, and audit logging of every executed statement.

## Compiled graph

Drawn from the compiled graph by `scripts/gen_docs.py` (one level of subgraphs expanded).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	inspect_schema(inspect_schema)
	write_sql(write_sql)
	guard(guard)
	execute(execute)
	answer(answer)
	give_up(give_up)
	__end__([<p>__end__</p>]):::last
	__start__ --> inspect_schema;
	execute -.-> answer;
	execute -.-> give_up;
	execute -. &nbsp;repair&nbsp; .-> write_sql;
	guard -.-> execute;
	guard -.-> give_up;
	guard -. &nbsp;repair&nbsp; .-> write_sql;
	inspect_schema --> write_sql;
	write_sql --> guard;
	answer --> __end__;
	give_up --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Example input

```json
{
  "question": "Which region had the most returned orders in the last 6 months?"
}
```

Run it: `uv run langgraph dev`, open Studio, choose `sql_analyst`, paste the input above.

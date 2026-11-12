"""Ask the shop database in plain English (guarded text-to-SQL).

PROBLEM
    Non-technical staff want answers from the shop database ("which region had the most returned orders last
    month?") without writing SQL. Naive text-to-SQL fails in three ways that matter. The model invents values
    ('online' when the channel is really called 'web'), so the query runs fine and silently returns nothing. It
    writes a wrong join and returns confident but wrong numbers. And because its output is executed, a careless or
    hostile prompt can produce DELETE / DROP / PRAGMA / ATTACH, a second stacked statement, a query that reads
    customers' e-mail addresses, or one that never finishes.

DESIGN
    The model writes SQL; code decides whether that SQL may run. Everything that can be decided without a model is
    decided without one:

      * ``inspect_schema`` builds the model's context from the live database: the CREATE statements, the distinct
        values of every low-cardinality text column (status, channel, region, segment, category, product name), the
        date range, and a few house rules (region lives on customers, count DISTINCT orders after joining items,
        what "revenue" means). The model copies real literals instead of guessing them.
      * ``write_sql`` (structured output: ``sql`` + a one-line ``rationale``) is the only creative step.
      * ``guard`` is deterministic and deny-by-default. It strips comments, requires exactly one statement that
        starts with SELECT or WITH, and then compiles the statement under SQLite's *authorizer*. The authorizer is
        called by SQLite itself for every operation, table, column and function the statement would touch, so it
        cannot be fooled by a CTE column called ``delete_flag`` or a string literal containing "DROP TABLE" the way a
        substring check can. Only SELECT / READ / FUNCTION / RECURSIVE actions, the four shop tables, an allow-list
        of functions and non-restricted columns pass. The guard also wraps the query as
        ``SELECT * FROM (<query>) LIMIT 101`` so the row cap cannot be forgotten or overridden.
      * ``execute`` runs on the read-only connection with a progress handler (time limit) and fetches at most
        100 rows, so even a legal but runaway query is stopped.
      * Anything that goes wrong - a guard rejection, a SQL error, a timeout, or an empty result (the classic sign
        of a wrong literal) - takes the ``repair`` edge back to ``write_sql`` with the exact problem and all earlier
        attempts in the prompt. The loop is capped at 3 attempts. If the model repeats the *same* SQL after an
        empty result, the loop stops early: that is the model standing by "there really are no such rows".
      * ``answer`` writes prose from the returned rows only and appends the SQL that produced them, so staff can see
        what was asked. When the loop gives up, ``give_up`` writes a fixed message instead: a model is never asked
        to explain results that do not exist.

GRAPH
    START -> inspect_schema -> write_sql -> guard --ok--> execute --ok--> answer -> END
                                   ^           |             |
                                   +- repair --+-------------+    (guard rejection, SQL error, timeout, empty result)
                                               |             |
                                               +-- give_up --+    (3 attempts used, or the same empty query repeated)
                                                     |
                                                     v
                                                    END

LANGGRAPH FEATURES
    * conditional edges with an explicit path map, so the ``repair`` edge is visible and named in Studio
    * a bounded repair loop: ``attempts`` lives in state, ``failed_attempts`` accumulates through a reducer
      (``operator.add``; ``inspect_schema`` resets it with ``Overwrite([])`` for each new question), and the cap is
      enforced by the router, not by hoping the model stops
    * separate ``input_schema`` / ``output_schema``: Studio shows one input box and a clean result
    * a deterministic safety layer wrapped around a model call (the pattern, not any one API, is the point)
    * structured output (Pydantic) for the SQL draft

MODELS AND DATA
    OpenAI chat: the default model writes the SQL (structured output), the fast model writes the final prose. No
    Decisions API. Data: ``store.sqlite`` (customers, products, orders, order_items), opened read-only.
    Simulated/offline: the offline stand-in model cannot write SQL, so an offline run produces placeholder text, the
    guard rejects it three times and the run ends with ``status: "failed"``. That is the safety net working, not a
    bug; set OPENAI_API_KEY to see real queries. "Today" is pinned to the day after the newest order in the
    database (2026-11-01), because the database is a frozen snapshot.

TRY IT IN STUDIO
    Open the ``Ready-made case`` dropdown, pick one and press Submit (or leave the form blank on a new thread: that runs
    the first one). Each case is a question the shop database can answer, or a deliberate trap:
      * ``Which region had the most returned orders in the last 6 months?`` - the plain case. Open ``inspect_schema``
        to see the context the model gets (note the VALUES list), then ``write_sql`` for the SQL and its rationale,
        ``guard`` for the query actually run, and the final ``answer`` / ``rows``.
      * ``How many orders came in through the online channel?`` - 'online' is not a value in the database (the channels
        are web, app, retail_partner, marketplace); read the ``rationale`` in ``write_sql`` to see which real channel it
        was mapped to, or the empty-result repair that follows a guessed literal.
      * ``... returned orders last month?`` - a three-way tie (one each); the answer should say so.
      * ``... delivered revenue by product category in the last 3 months?`` - a three-table join with the house rules
        for revenue and discounts.
      * ``Delete all cancelled orders`` and ``Show me every customer's email address`` - a model may refuse by writing
        a harmless SELECT instead, but if it writes a DELETE, or reads ``customers.email``, ``guard`` rejects it, the
        reason appears under ``failed_attempts`` and the repair loop asks again; the database is never touched either way.
      * ``How many orders were placed in 2030?`` - the data ends on 2026-10-31, so the query comes back empty; watch the
        repair loop and whether it ends as ``no_rows``.
    To ask your own, paste ``{"question": "Which region had the most returned orders in the last 6 months?"}`` (any
    question works) instead of picking a case.

EXTEND IT
    * Add a column-level description table (a semantic layer) and pass only the relevant tables to the model.
    * Add ``interrupt()`` before ``execute`` for queries that touch customers, so a human approves PII access.
    * Add a second model call that checks the SQL against the question (a "does this query answer the question?"
      critic), or ask the Decisions API for a calibrated probability and require a threshold.
    Production hardening not done here: per-user row-level permissions, query cost estimates (EXPLAIN QUERY PLAN),
    caching, and audit logging of every executed statement.
"""

from __future__ import annotations

import datetime as dt
import operator
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import db_connect, db_schema
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="sql_analyst",
    title="Ask the shop database (guarded text-to-SQL)",
    category="Agents & tools",
    problem="Let non-technical staff query the shop database in plain English without hallucinated values, wrong "
    "joins or unsafe SQL ever reaching the database.",
    features=[
        "guarded repair loop",
        "conditional edges with named repair edge",
        "reducer-accumulated attempt history",
        "input/output schemas",
        "deterministic safety layer around a model",
        "structured output",
    ],
    models=["OpenAI chat"],
    datasets=["store.sqlite"],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"question": "Which region had the most returned orders in the last 6 months?"}

# Ready-made questions for Studio's dropdown (see deepdive.common.scenarios). The first one runs on a blank form. The
# facts in the labels (channel names, the 32 cancelled orders, the tie, the last order date) come from store.sqlite and
# are pinned by tests/test_sql_analyst.py; what the MODEL does with each question is only ever "watch for".
CASES: dict[str, dict] = {
    "Which region had the most returned orders in the last 6 months? -> read the SQL, its date window and the ranking": {
        "question": "Which region had the most returned orders in the last 6 months?"},
    "How many orders came in through the online channel? -> no channel is called 'online': watch how it maps or repairs": {
        "question": "How many orders came in through the online channel?"},
    "Which region had the most returned orders last month? -> a three-way tie, 1 each: does the answer say so?": {
        "question": "Which region had the most returned orders last month?"},
    "What was delivered revenue by product category in the last 3 months? -> watch the 3-table join and the discount maths": {
        "question": "What was delivered revenue by product category in the last 3 months?"},
    "Delete all cancelled orders -> a write request: nothing is ever written, all 32 cancelled orders stay": {
        "question": "Delete all cancelled orders"},
    "Show me every customer's email address -> email is restricted: watch the guard block it, or the model refuse": {
        "question": "Show me every customer's email address"},
    "How many orders were placed in 2030? -> the data ends 2026-10-31: watch the empty-result repair, then how it ends": {
        "question": "How many orders were placed in 2030?"},
}
SCENARIO = scenario_field(CASES)

# --------------------------------------------------------------------------- limits (all named so tests can see them)
MAX_ATTEMPTS = 3  # write_sql runs at most this many times per question
MAX_ROWS = 100  # rows returned to the caller; the 101st row only tells us the result was truncated
PROMPT_ROWS = 25  # rows the answer model gets to see
QUERY_TIMEOUT_S = 2.0  # wall-clock budget for one query (enforced by a SQLite progress handler)
LOW_CARDINALITY = 12  # text columns with at most this many distinct values are listed for the model

# What the authorizer lets a statement do. Everything else (INSERT, DELETE, DROP, PRAGMA, ATTACH, ...) is denied.
_ALLOWED_ACTIONS = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE})
# Functions are allow-listed too: randomblob()/zeroblob() can exhaust memory and load_extension() has no business here.
ALLOWED_FUNCTIONS = frozenset(
    """count sum avg min max total group_concat abs round coalesce ifnull nullif iif length lower upper substr
    substring trim ltrim rtrim replace instr like glob printf format typeof sign date time datetime julianday strftime
    unixepoch row_number rank dense_rank ntile lag lead first_value last_value nth_value cume_dist percent_rank
    ceil ceiling floor sqrt pow power ln log log10 log2 exp mod concat concat_ws string_agg""".split()
)
# Personal data the shop database holds but this assistant must not hand out in bulk.
RESTRICTED_COLUMNS = frozenset({("customers", "email")})

_ACTION_NAMES = {
    getattr(sqlite3, n): n.removeprefix("SQLITE_")
    for n in (
        "SQLITE_INSERT SQLITE_UPDATE SQLITE_DELETE SQLITE_PRAGMA SQLITE_ATTACH SQLITE_DETACH SQLITE_TRANSACTION "
        "SQLITE_CREATE_TABLE SQLITE_CREATE_INDEX SQLITE_CREATE_VIEW SQLITE_CREATE_TRIGGER SQLITE_DROP_TABLE "
        "SQLITE_DROP_INDEX SQLITE_DROP_VIEW SQLITE_DROP_TRIGGER SQLITE_ALTER_TABLE SQLITE_ANALYZE SQLITE_REINDEX "
        "SQLITE_SAVEPOINT"
    ).split()
    if hasattr(sqlite3, n)
}


# --------------------------------------------------------------------------- the safety layer (pure code, no model)
@dataclass
class GuardResult:
    ok: bool
    sql: str = ""  # the model's query with comments and trailing semicolons removed (this is what we show users)
    run_sql: str = ""  # the statement that will actually run: the query wrapped in a row limit
    problem: str | None = None  # why it was rejected, written so the model can fix it


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


def _clean(sql: str) -> tuple[str, str | None]:
    """Remove comments and trailing semicolons; reject anything that is not exactly one well-formed statement.

    A small scanner (not a regex) so that ``;`` or ``--`` inside a quoted string is left alone while a real
    ``; DROP TABLE`` after a comment is still seen. This is for clear error messages and for making the row-limit
    wrapper safe (a trailing ``--`` would otherwise swallow its closing parenthesis). The hard guarantee comes from
    the authorizer in :func:`check_sql`, not from this function.
    """
    out: list[str] = []
    i, n, depth, after_semicolon = 0, len(sql), 0, False
    while i < n:
        c = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                return "", "unterminated /* comment"
            i = j + 2
            out.append(" ")
            continue
        if c.isspace():
            out.append(c)
            i += 1
            continue
        if c == ";":
            after_semicolon = True
            i += 1
            continue
        if after_semicolon:
            return "", "more than one statement (anything after a semicolon is refused)"
        if c in "'\"`":  # quoted literal or identifier; a doubled quote is an escaped quote
            j = i + 1
            while True:
                if j >= n:
                    return "", f"unterminated {c} quote"
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i : j + 1])
            i = j + 1
            continue
        if c == "[":  # [bracket quoted identifier]
            j = sql.find("]", i + 1)
            if j == -1:
                return "", "unterminated [ identifier"
            out.append(sql[i : j + 1])
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return "", "unbalanced parentheses"
        out.append(c)
        i += 1
    if depth != 0:
        return "", "unbalanced parentheses"
    cleaned = "".join(out).strip()
    return (cleaned, None) if cleaned else ("", "empty query")


def _strip_fence(sql: str) -> str:
    """Models sometimes wrap SQL in a Markdown code fence even inside a structured field; unwrap it instead of failing."""
    m = re.fullmatch(r"\s*```(?:sql|sqlite)?\s*\n?(.*?)\n?\s*```\s*", sql, flags=re.S | re.I)
    return m.group(1) if m else sql


def _open_guarded(denials: list[str]) -> sqlite3.Connection:
    """A read-only connection with the authorizer installed. Reasons for any denial are appended to ``denials``."""
    con = db_connect()
    allowed_tables = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    }

    def authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None, _source: str | None) -> int:
        if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column ("" for COUNT(*))
            if arg1 not in allowed_tables:
                denials.append(f"table '{arg1}' is not available (tables: {', '.join(sorted(allowed_tables))})")
                return sqlite3.SQLITE_DENY
            if (arg1, arg2) in RESTRICTED_COLUMNS:
                denials.append(
                    f"column {arg1}.{arg2} is restricted personal data and cannot be read "
                    "(list the columns you need instead of using *)"
                )
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:  # arg2 = function name
            if (arg2 or "").lower() in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            denials.append(f"function {arg2}() is not allowed")
            return sqlite3.SQLITE_DENY
        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        op = _ACTION_NAMES.get(action, f"operation {action}")
        denials.append(f"{op}{f' on {arg1}' if arg1 else ''} is not allowed: this connection is read-only")
        return sqlite3.SQLITE_DENY

    con.set_authorizer(authorizer)
    return con


def check_sql(sql: str, *, max_rows: int = MAX_ROWS) -> GuardResult:
    """Decide whether model-written SQL may run. Pure code: no model, no side effects, nothing is executed."""
    cleaned, problem = _clean(_strip_fence(sql or ""))
    if problem:
        return GuardResult(False, problem=f"Rejected: {problem}.")
    keyword = re.match(r"[A-Za-z]+", cleaned)
    if not keyword or keyword.group(0).lower() not in ("select", "with"):
        first = keyword.group(0).upper() if keyword else cleaned[:12]
        return GuardResult(False, cleaned, problem=f"Rejected: only a single SELECT (optionally starting with WITH) is allowed, not {first}.")
    run_sql = f"SELECT * FROM ({cleaned}) LIMIT {max_rows + 1}"
    denials: list[str] = []
    con = _open_guarded(denials)
    try:
        con.execute("EXPLAIN " + run_sql)  # compiles under the authorizer; nothing runs
    except (sqlite3.Error, ValueError) as e:  # ValueError: Python refuses embedded NUL characters
        reason = "; ".join(dict.fromkeys(denials)) or f"SQLite cannot compile this query: {e}"
        return GuardResult(False, cleaned, problem=f"Rejected: {reason}.")
    finally:
        con.close()
    return GuardResult(True, cleaned, run_sql)


def _json_safe(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def run_query(run_sql: str, *, max_rows: int = MAX_ROWS, timeout_s: float | None = None) -> QueryResult:
    """Run an already-guarded statement: read-only connection, authorizer again, time limit, row cap.

    Raises ``sqlite3.Error`` (including ``OperationalError('interrupted')`` when the time limit is hit).
    """
    budget = QUERY_TIMEOUT_S if timeout_s is None else timeout_s
    denials: list[str] = []
    con = _open_guarded(denials)
    try:
        deadline = time.monotonic() + budget
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cur = con.execute(run_sql)
        columns = [d[0] for d in cur.description]
        fetched = cur.fetchmany(max_rows + 1)
    finally:
        con.close()
    rows = [{c: _json_safe(v) for c, v in zip(columns, row)} for row in fetched[:max_rows]]
    return QueryResult(columns, rows, truncated=len(fetched) > max_rows)


# --------------------------------------------------------------------------- schema context (deterministic)
HOUSE_RULES = """\
- Region and segment belong to customers: join orders.customer_id = customers.customer_id.
- An order has many order_items (join on order_id). After joining to order_items, count orders with COUNT(DISTINCT orders.order_id).
- Line value = order_items.quantity * order_items.unit_price. orders.discount_pct is a whole-number percentage off the order: multiply by (1 - orders.discount_pct / 100.0).
- Unless the question says otherwise, "revenue" and "sales" mean delivered orders only (status = 'delivered'). Say which statuses you included in the rationale.
- "Returns" / "returned orders" means orders.status = 'returned'. products.unit_price is today's list price; order_items.unit_price is what was charged."""


def build_schema_context() -> str:
    """CREATE statements + distinct values of low-cardinality text columns + date convention + house rules."""
    con = db_connect()
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        values: list[str] = []
        for table in tables:
            for col in con.execute(f'PRAGMA table_info("{table}")').fetchall():  # table names come from sqlite_master, not the model
                name = col["name"]
                if (col["type"] or "").upper() != "TEXT" or (table, name) in RESTRICTED_COLUMNS:
                    continue
                distinct = [
                    r[0]
                    for r in con.execute(f'SELECT DISTINCT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL ORDER BY 1 LIMIT {LOW_CARDINALITY + 1}')
                ]
                if 0 < len(distinct) <= LOW_CARDINALITY:
                    values.append(f"{table}.{name}: " + ", ".join(f"'{v}'" for v in distinct))
        first, last = con.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
    finally:
        con.close()
    today = dt.date.fromisoformat(last) + dt.timedelta(days=1)
    this_month = today.replace(day=1)
    last_month = (this_month - dt.timedelta(days=1)).replace(day=1)
    dates = (
        f"Dates are ISO text ('YYYY-MM-DD'); use date(), strftime() and plain comparisons. Orders span {first} to {last}.\n"
        f"Treat today as {today} (the database is a frozen snapshot). 'Last month' = order_date >= '{last_month}' AND "
        f"order_date < '{this_month}'. 'This month' starts at '{this_month}' and has no orders yet. Compute 'last N months' "
        f"and quarters from today."
    )
    return f"SCHEMA\n{db_schema()}\n\nVALUES (the only legal literals for these columns)\n" + "\n".join(values) + f"\n\nDATES\n{dates}\n\nHOUSE RULES\n{HOUSE_RULES}"


# --------------------------------------------------------------------------- schemas
class SqlDraft(BaseModel):
    """One candidate query."""

    sql: str = Field(description="A single read-only SQLite SELECT (optionally WITH ... SELECT). No semicolon, no comments.")
    rationale: str = Field(description="One short sentence: how you read the question, including any date or revenue assumption")


class QuestionIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made question; replaces `question`
    question: str  # plain-English question about orders, customers, products, returns or revenue


class AnalystOut(TypedDict):
    question: str
    status: str  # answered | no_rows | failed
    answer: str
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    attempts: int
    failed_attempts: list[dict]
    mode: str


class AnalystState(QuestionIn, total=False):
    schema_context: str
    sql: str
    run_sql: str
    rationale: str
    attempts: int
    problem: str | None
    failed_attempts: Annotated[list[dict], operator.add]  # one entry per rejected / failed attempt
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool
    status: str
    answer: str
    mode: str


# --------------------------------------------------------------------------- nodes
WRITE_PROMPT = (
    "You write ONE read-only SQLite query that answers a question about the Lumen Home shop database.\n"
    "Hard rules:\n"
    "- A single SELECT statement, optionally starting with WITH. No semicolons, no comments, no placeholders (? or :name), "
    "no PRAGMA / INSERT / UPDATE / DELETE / DDL.\n"
    "- Use only the tables and columns in the schema. Never SELECT *: name the columns you need. customers.email is "
    "restricted personal data and cannot be read.\n"
    "- For text filters copy a literal EXACTLY from the VALUES list (same spelling and case). Never invent values such "
    "as 'online' or 'Q4'. If the question mentions something that is not in VALUES, use the closest real value and say so in the rationale.\n"
    "- Give every computed column a readable alias (returned_orders, revenue, ...).\n"
    "- Add ORDER BY for rankings. The system limits the rows returned, so only add LIMIT when the question asks for a top-N.\n"
    "- rationale: one short sentence on how you read the question, including date and revenue assumptions."
)

EMPTY_RESULT = (
    "The query ran but returned zero rows. Check every text literal against the VALUES list and the date range. "
    "If you are certain the query is right and the honest answer is 'none', return the same query unchanged."
)


def inspect_schema(state: AnalystState) -> dict:
    """Build the model's context from the live database (schema, real literals, date convention, house rules).

    Also the entry node, so it first resolves a picked ready-made question (or runs the first one on a blank form).
    """
    picked = expand(state, CASES, has_input=lambda s: bool((s.get("question") or "").strip()))
    question = ({**state, **picked}.get("question") or "").strip()
    if not question:  # unreachable while every case has a question; keeps the invariant visible
        raise ValueError("Provide a non-empty `question`.")
    # ``failed_attempts`` appends (operator.add) and Studio reuses the thread: reset it, or the previous question's
    # failures would be shown to the writer as "earlier attempts" and count towards giving up.
    return {**picked, "question": question, "schema_context": build_schema_context(), "attempts": 0, "failed_attempts": Overwrite([])}


def _same_sql(a: str, b: str) -> bool:
    return re.sub(r"\s+", " ", a).strip() == re.sub(r"\s+", " ", b).strip()


def write_sql(state: AnalystState) -> dict:
    """The only creative step. On a retry the prompt carries every earlier attempt and exactly why it failed."""
    attempt = state.get("attempts", 0) + 1
    user = f"Question: {state['question']}"
    previous = state.get("failed_attempts") or []
    if previous:
        user += "\n\nEarlier attempts that did not work:"
        for f in previous:
            user += f"\n- Attempt {f['attempt']} SQL: {f['sql']}\n  Problem: {f['problem']}"
        user += "\nWrite a corrected query. Do not repeat a query that already failed (except as described above)."
    draft = structured_llm(SqlDraft).invoke([SystemMessage(WRITE_PROMPT + "\n\n" + state["schema_context"]), HumanMessage(user)])
    return {"sql": draft.sql.strip(), "rationale": draft.rationale, "attempts": attempt, "problem": None}


def _fail(state: AnalystState, stage: str, problem: str, **extra: Any) -> dict:
    entry = {"attempt": state["attempts"], "stage": stage, "sql": state["sql"], "problem": problem}
    return {"problem": problem, "failed_attempts": [entry], **extra}


def guard(state: AnalystState) -> dict:
    """Deterministic gate: exactly one SELECT/WITH, read-only, allow-listed, row-limited. See :func:`check_sql`."""
    result = check_sql(state["sql"])
    if not result.ok:
        return _fail(state, "guard", result.problem or "Rejected.")
    return {"sql": result.sql, "run_sql": result.run_sql, "problem": None}


def execute(state: AnalystState) -> dict:
    """Run the guarded statement. SQL errors, timeouts and empty results become a problem for the repair edge."""
    try:
        res = run_query(state["run_sql"])
    except sqlite3.Error as e:
        if "interrupted" in str(e):
            problem = f"The query was stopped after {QUERY_TIMEOUT_S:g}s. Make it cheaper: join on keys, filter early, avoid cross joins."
        else:
            problem = f"SQLite error: {e}"
        return _fail(state, "execute", problem, rows=[], row_count=0)
    update = {"columns": res.columns, "rows": res.rows, "row_count": len(res.rows), "truncated": res.truncated}
    if not res.rows:
        return _fail(state, "empty", EMPTY_RESULT, **update)
    return {**update, "problem": None}


def after_guard(state: AnalystState) -> Literal["execute", "repair", "give_up"]:
    if not state.get("problem"):
        return "execute"
    return "give_up" if state["attempts"] >= MAX_ATTEMPTS else "repair"


def after_execute(state: AnalystState) -> Literal["answer", "repair", "give_up"]:
    if not state.get("problem"):
        return "answer"
    failures = state["failed_attempts"]
    repeated_empty = len(failures) >= 2 and failures[-1]["stage"] == failures[-2]["stage"] == "empty" and _same_sql(failures[-1]["sql"], failures[-2]["sql"])
    return "give_up" if (state["attempts"] >= MAX_ATTEMPTS or repeated_empty) else "repair"


ANSWER_PROMPT = (
    "You answer a staff member's question about the Lumen Home shop using ONLY the query result below.\n"
    "- Lead with the direct answer in one or two sentences; add at most three short bullets if a ranking or breakdown helps.\n"
    "- Use numbers exactly as they appear. Never estimate or invent figures the rows do not show.\n"
    "- If several rows tie for first place, say it is a tie. If the result was truncated or you were shown only the first rows, say so.\n"
    "- Mention any assumption from the rationale (for example how 'last month' was read) in one short clause.\n"
    "- Do not mention SQL, tables or columns; the query is shown separately."
)


def _sql_block(sql: str, attempts: int, label: str = "SQL used") -> str:
    return f"\n\n{label} ({attempts} attempt{'s' if attempts != 1 else ''}):\n{sql}"


def answer(state: AnalystState) -> dict:
    """Prose from the returned rows only, followed by the SQL that produced them."""
    shown = state["rows"][:PROMPT_ROWS]
    note = f" Only the first {len(shown)} of {state['row_count']} rows are shown to you." if len(shown) < state["row_count"] else ""
    note += " The full result was truncated at the row cap." if state.get("truncated") else ""
    result_text = f"Columns: {state['columns']}\nRows ({state['row_count']} returned){note}:\n" + "\n".join(str(r) for r in shown)
    msg = get_llm("fast").invoke(
        [
            SystemMessage(ANSWER_PROMPT),
            HumanMessage(f"Question: {state['question']}\nRationale for the query: {state['rationale']}\n\n{result_text}"),
        ]
    )
    return {"status": "answered", "answer": str(msg.text).strip() + _sql_block(state["sql"], state["attempts"]), "mode": mode_label()}


def give_up(state: AnalystState) -> dict:
    """Fixed text, no model: the loop ended without a result worth explaining."""
    last = state["failed_attempts"][-1]
    n = state["attempts"]
    if last["stage"] == "empty":
        status = "no_rows"
        text = (
            f"The query ran without errors but matched no rows ({n} attempt{'s' if n != 1 else ''}). Either nothing in the "
            "database matches, or the question uses a value the database does not contain. Try rephrasing, or ask what "
            "values exist (for example the channels or regions)."
        )
    else:
        status = "failed"
        text = (
            f"I could not produce a safe, working query for this question after {n} attempt{'s' if n != 1 else ''}. "
            f"The last problem was: {last['problem']}"
        )
    return {
        "status": status,
        "answer": text + _sql_block(last["sql"], n, "Last SQL tried"),
        "sql": last["sql"],
        "columns": state.get("columns", []) if status == "no_rows" else [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "mode": mode_label(),
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(AnalystState, input_schema=QuestionIn, output_schema=AnalystOut)
    g.add_node("inspect_schema", inspect_schema)
    g.add_node("write_sql", write_sql)
    g.add_node("guard", guard)
    g.add_node("execute", execute)
    g.add_node("answer", answer)
    g.add_node("give_up", give_up)
    g.add_edge(START, "inspect_schema")
    g.add_edge("inspect_schema", "write_sql")
    g.add_edge("write_sql", "guard")
    # The path maps name the loop: "repair" is the edge back to write_sql.
    g.add_conditional_edges("guard", after_guard, {"execute": "execute", "repair": "write_sql", "give_up": "give_up"})
    g.add_conditional_edges("execute", after_execute, {"answer": "answer", "repair": "write_sql", "give_up": "give_up"})
    g.add_edge("answer", END)
    g.add_edge("give_up", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

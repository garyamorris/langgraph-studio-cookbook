from __future__ import annotations

import json
import re
import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepdive.common.data import read_csv
from deepdive.common.fakes import scripted, set_chat_handler, set_structured_handler
from deepdive.common.testing import check_example, lint_openai_strict, normalise, run_twice
from deepdive.examples import sql_analyst as ex

RETURNS_BY_REGION = (
    "SELECT c.region AS region, COUNT(DISTINCT o.order_id) AS returned_orders "
    "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
    "WHERE o.status = 'returned' GROUP BY c.region ORDER BY returned_orders DESC, region"
)


def drafts(*sqls: str):
    """Script write_sql: the k-th call gets the k-th query. Returns the list of prompts the model was shown."""
    seen: list[str] = []
    nxt = scripted([ex.SqlDraft(sql=s, rationale=f"attempt {i + 1}") for i, s in enumerate(sqls)])

    def handler(text: str):
        seen.append(text)
        return nxt(text)

    set_structured_handler(ex.SqlDraft, handler)
    return seen


def count_chat_calls():
    """Count (and capture) plain chat calls - in this graph only the `answer` node makes them."""
    calls: list[str] = []

    def handler(messages, tools):
        calls.append("\n".join(str(m.content) for m in messages))
        return AIMessage("scripted answer")

    set_chat_handler(handler)
    return calls


def test_contract_and_offline_run():
    res = check_example(ex)
    # The offline stand-in cannot write SQL, so the guard must stop it rather than let placeholder text through.
    assert res.output["status"] == "failed"
    assert res.output["attempts"] == ex.MAX_ATTEMPTS


def test_every_schema_is_openai_strict_compatible():
    # check_example only lints schemas the offline run happens to use; lint them all.
    assert lint_openai_strict(ex.SqlDraft) == []


# --------------------------------------------------------------------------- the guard
@pytest.mark.parametrize(
    "sql, why",
    [
        ("DROP TABLE orders", "not DROP"),
        ("DELETE FROM orders", "not DELETE"),
        ("UPDATE orders SET status = 'delivered'", "not UPDATE"),
        ("INSERT INTO orders VALUES (1, 1, '2026-01-01', 'web', 'delivered', 0)", "not INSERT"),
        ("PRAGMA table_info(orders)", "not PRAGMA"),
        ("ATTACH DATABASE 'x.db' AS x", "not ATTACH"),
        ("EXPLAIN SELECT 1", "not EXPLAIN"),
        ("SELECT 1; DROP TABLE orders", "more than one statement"),
        ("SELECT 1; SELECT 2", "more than one statement"),
        # comment tricks: the DROP is visible once comments are removed, and an unterminated comment is malformed
        ("SELECT 1; -- harmless\nDROP TABLE orders", "more than one statement"),
        ("SELECT 1 /* x */ ; /* y */ DELETE FROM orders", "more than one statement"),
        ("SELECT 1 /* never closed; DROP TABLE orders", "unterminated /* comment"),
        ("SELECT 'never closed", "unterminated"),
        ("SELECT 1) UNION SELECT 2 --", "unbalanced"),
        # start like a query but are not one: the row-limit wrapper makes these syntax errors (see the next test for
        # the authorizer layer, which would stop them even without the wrapper)
        ("WITH t AS (SELECT 1) DELETE FROM orders", "cannot compile"),
        ("WITH t AS (SELECT 1) INSERT INTO orders SELECT * FROM orders", "cannot compile"),
        ("SELECT load_extension('evil')", "function load_extension() is not allowed"),
        ("SELECT length(zeroblob(1000000000))", "function zeroblob() is not allowed"),
        ("SELECT name FROM sqlite_master", "table 'sqlite_master' is not available"),
        ("SELECT email FROM customers", "restricted personal data"),
        ("SELECT * FROM customers", "restricted personal data"),  # * expands to include email
        ("SELECT nosuchcolumn FROM orders", "cannot compile"),
        ("SELECT order_id FROM orders WHERE status = ?", "cannot compile"),  # placeholders have nothing to bind to
        ("SELECT 1\x00", "cannot compile"),  # embedded NUL: a ValueError in Python, not a crash here
        ("", "empty query"),
    ],
)
def test_guard_rejects(sql, why):
    r = ex.check_sql(sql)
    assert not r.ok and r.problem and why.lower() in r.problem.lower(), r
    assert r.run_sql == "", "a rejected statement must never come with something runnable"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) AS n FROM orders",
        "select Status, count(*) from ORDERS group by 1;",
        "  \n SELECT status, COUNT(*) AS n FROM orders -- trailing comment\n GROUP BY status",
        "SELECT /* inline */ COUNT(*) AS n FROM orders",
        # A CTE whose column names look like dangerous keywords. A substring check would reject this.
        "WITH update_log AS (SELECT order_id AS delete_flag, status AS drop_reason FROM orders) "
        "SELECT delete_flag, drop_reason FROM update_log LIMIT 3",
        # Dangerous-looking text inside a string literal is just text.
        "SELECT 'DROP TABLE orders; DELETE FROM orders' AS note FROM orders LIMIT 1",
        'SELECT "order_id" FROM orders LIMIT 1',
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT x FROM c",
        "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n FROM orders GROUP BY month ORDER BY month",
        "SELECT c.region, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rk FROM orders o JOIN customers c USING (customer_id) GROUP BY c.region",
    ],
)
def test_guard_accepts_legitimate_queries(sql):
    r = ex.check_sql(sql)
    assert r.ok, r.problem
    assert r.run_sql.startswith("SELECT * FROM (") and r.run_sql.endswith(f"LIMIT {ex.MAX_ROWS + 1}")
    assert not r.sql.rstrip().endswith(";"), "trailing semicolons are removed so the wrapper stays one statement"
    assert "--" not in r.sql and "/*" not in r.sql, "comments are stripped so they cannot swallow the row-limit wrapper"


@pytest.mark.parametrize("fenced", ["```sql\nSELECT COUNT(*) AS n FROM orders;\n```", "```\nSELECT COUNT(*) AS n FROM orders\n```", " ```SQL\nSELECT COUNT(*) AS n FROM orders\n``` "])
def test_markdown_fences_around_sql_are_unwrapped_not_rejected(fenced):
    r = ex.check_sql(fenced)
    assert r.ok, r.problem
    assert r.sql == "SELECT COUNT(*) AS n FROM orders"
    assert ex.run_query(r.run_sql).rows == [{"n": 845}]


def test_guard_cannot_be_talked_out_of_the_row_limit():
    # The model asks for a huge LIMIT; the wrapper still caps what is fetched.
    r = ex.check_sql("SELECT order_id FROM orders ORDER BY order_id LIMIT 100000")
    assert r.ok
    res = ex.run_query(r.run_sql)
    assert len(res.rows) == ex.MAX_ROWS and res.truncated


# --------------------------------------------------------------------------- execution limits
def test_row_cap_truncates_and_a_small_result_is_not_flagged():
    big = ex.run_query(ex.check_sql("SELECT order_id FROM orders ORDER BY order_id").run_sql)
    assert len(big.rows) == ex.MAX_ROWS and big.truncated and big.rows[0] == {"order_id": 1001}
    small = ex.run_query(ex.check_sql("SELECT order_id FROM orders ORDER BY order_id LIMIT 5").run_sql)
    assert len(small.rows) == 5 and not small.truncated


def test_inner_order_by_survives_the_limit_wrapper():
    res = ex.run_query(ex.check_sql("SELECT order_id FROM orders ORDER BY order_id DESC").run_sql)
    ids = [r["order_id"] for r in res.rows]
    assert ids == sorted(ids, reverse=True)


def test_time_limit_stops_a_runaway_query():
    cross = ex.check_sql("SELECT COUNT(*) AS n FROM order_items a, order_items b, order_items c")
    assert cross.ok, "the query is legal; only the clock can stop it"
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        ex.run_query(cross.run_sql, timeout_s=0.05)


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("WITH t AS (SELECT 1) DELETE FROM orders", "DELETE on orders is not allowed"),
        ("WITH t AS (SELECT 1) INSERT INTO orders SELECT * FROM orders", "INSERT on orders is not allowed"),
        ("UPDATE orders SET status = 'x'", "UPDATE on orders is not allowed"),
        ("DROP TABLE orders", "on sqlite_master is not allowed"),  # SQLite edits its catalogue first
        ("CREATE TABLE evil (x)", "on sqlite_master is not allowed"),
        ("PRAGMA writable_schema = 1", "PRAGMA on writable_schema is not allowed"),
        ("ATTACH DATABASE ':memory:' AS m", "ATTACH on :memory: is not allowed"),
    ],
)
def test_authorizer_is_an_independent_layer_that_names_the_forbidden_operation(sql, reason):
    # Bypass check_sql entirely and hand the raw statement to the guarded connection: SQLite itself must refuse.
    denials: list[str] = []
    con = ex._open_guarded(denials)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            con.execute(sql)
    finally:
        con.close()
    assert any(reason in d for d in denials), denials


def test_execution_layer_refuses_writes_even_if_the_guard_were_bypassed():
    with pytest.raises(sqlite3.Error):
        ex.run_query("DELETE FROM orders")
    with pytest.raises(sqlite3.Error):
        ex.run_query("SELECT * FROM (SELECT email FROM customers) LIMIT 5")


# --------------------------------------------------------------------------- the schema context the model sees
def test_schema_context_lists_real_literals_dates_and_hides_pii():
    ctx = ex.inspect_schema({"question": "anything"})["schema_context"]
    for literal in ("'returned'", "'retail_partner'", "'UK & Ireland'", "'small_business'", "'plugs'", "'Lumen Hub'"):
        assert literal in ctx, f"{literal} should be offered to the model as a real value"
    assert "email" not in ctx.split("VALUES")[1], "restricted columns must not be sampled"
    assert "Treat today as 2026-11-01" in ctx
    assert "order_date >= '2026-10-01' AND order_date < '2026-11-01'" in ctx
    assert "COUNT(DISTINCT orders.order_id)" in ctx


def test_blank_question_is_an_input_error():
    with pytest.raises(ValueError, match="question"):
        ex.graph.invoke({"question": "   "})


# --------------------------------------------------------------------------- the graph
def test_a_real_query_returns_correct_numbers_and_is_json_safe():
    prompts = drafts(RETURNS_BY_REGION)
    chat = count_chat_calls()
    out = ex.graph.invoke({"question": "Which region had the most returned orders?"})

    # Independent source: the order-line CSV, not the SQLite file.
    sales = read_csv("sales.csv")
    expected = (
        sales[sales.status == "returned"].groupby("region").order_id.nunique().sort_values(ascending=False, kind="stable")
    )
    got = {r["region"]: r["returned_orders"] for r in out["rows"]}
    assert got == expected.to_dict()
    assert out["rows"][0] == {"region": "North America", "returned_orders": 17}

    assert out["status"] == "answered" and out["attempts"] == 1 and out["failed_attempts"] == []
    assert out["sql"] == RETURNS_BY_REGION and out["columns"] == ["region", "returned_orders"]
    assert out["row_count"] == 4 and not out["truncated"]
    assert RETURNS_BY_REGION in out["answer"], "the answer must show the SQL that produced it"
    assert out["mode"] == "offline"
    assert "North America" in chat[0] and "17" in chat[0], "the answer model is shown the real rows"
    assert "'retail_partner'" in prompts[0], "the writer is shown the real literals"
    json.dumps(out)  # everything in the result must be JSON-serialisable


def test_invalid_column_goes_through_repair_then_succeeds():
    prompts = drafts(
        "SELECT regionn, COUNT(*) AS n FROM customers GROUP BY regionn",  # typo: does not compile
        "SELECT region, COUNT(*) AS n FROM customers GROUP BY region ORDER BY n DESC",
    )
    count_chat_calls()
    out = ex.graph.invoke({"question": "How many customers per region?"})
    assert out["status"] == "answered" and out["attempts"] == 2
    assert len(prompts) == 2
    assert "Earlier attempts" not in prompts[0]
    assert "no such column: regionn" in prompts[1], "the repair prompt must carry the actual error"
    assert "SELECT regionn" in prompts[1]
    assert [f["stage"] for f in out["failed_attempts"]] == ["guard"]
    assert sum(r["n"] for r in out["rows"]) == 180


def test_guard_rejection_is_fed_back_and_nothing_dangerous_runs():
    prompts = drafts("DELETE FROM orders WHERE status = 'cancelled'", "SELECT COUNT(*) AS cancelled FROM orders WHERE status = 'cancelled'")
    count_chat_calls()
    out = ex.graph.invoke({"question": "Delete all cancelled orders"})
    assert out["attempts"] == 2 and out["status"] == "answered"
    assert "only a single SELECT" in prompts[1] and "DELETE FROM orders" in prompts[1]
    assert out["rows"] == [{"cancelled": 32}]
    # the shop is untouched
    assert ex.run_query(ex.check_sql("SELECT COUNT(*) AS n FROM orders").run_sql).rows == [{"n": 845}]


def test_repair_loop_is_capped_at_three_attempts_and_then_gives_up_without_a_model():
    prompts = drafts("DROP TABLE orders")  # the last value repeats: the model never learns
    chat = count_chat_calls()
    out = ex.graph.invoke({"question": "Please drop the orders table"})
    assert len(prompts) == ex.MAX_ATTEMPTS == 3, "write_sql must run exactly MAX_ATTEMPTS times"
    assert out["status"] == "failed" and out["attempts"] == 3
    assert [f["attempt"] for f in out["failed_attempts"]] == [1, 2, 3]
    assert chat == [], "the answer model must never be asked to explain a result that does not exist"
    assert "could not produce a safe, working query" in out["answer"]
    assert out["rows"] == [] and out["row_count"] == 0


def test_empty_result_triggers_repair_with_a_hint_about_literals():
    prompts = drafts(
        "SELECT COUNT(*) AS n FROM orders WHERE channel = 'online' AND status = 'returned' GROUP BY channel",  # no such channel value
        "SELECT channel, COUNT(*) AS n FROM orders WHERE channel = 'web' GROUP BY channel",
    )
    count_chat_calls()
    out = ex.graph.invoke({"question": "How many orders came in through the online channel?"})
    assert out["attempts"] == 2 and out["status"] == "answered"
    assert "returned zero rows" in prompts[1] and "VALUES list" in prompts[1]
    assert out["failed_attempts"][0]["stage"] == "empty"
    assert out["rows"][0]["channel"] == "web"


def test_empty_result_confirmed_by_the_model_ends_early_as_no_rows():
    same = "SELECT order_id FROM orders WHERE order_date > '2030-01-01'"
    prompts = drafts(same, same + "  ")  # whitespace-only difference = the same query
    chat = count_chat_calls()
    out = ex.graph.invoke({"question": "Orders placed after 2030?"})
    assert out["status"] == "no_rows" and out["attempts"] == 2 and len(prompts) == 2, "no third attempt after the model repeats itself"
    assert chat == [] and "matched no rows" in out["answer"]


def test_empty_result_that_keeps_changing_stops_at_the_cap():
    prompts = drafts(*[f"SELECT order_id FROM orders WHERE order_id = {n}" for n in (1, 2, 3, 4)])
    out = ex.graph.invoke({"question": "Orders with an id below 1000?"})
    assert out["status"] == "no_rows" and out["attempts"] == 3 and len(prompts) == 3


def test_a_timeout_is_repairable(monkeypatch):
    monkeypatch.setattr(ex, "QUERY_TIMEOUT_S", 0.05)
    prompts = drafts(
        "SELECT COUNT(*) AS n FROM order_items a, order_items b, order_items c",
        "SELECT COUNT(*) AS n FROM order_items",
    )
    count_chat_calls()
    out = ex.graph.invoke({"question": "How many order lines are there?"})
    assert out["attempts"] == 2 and out["rows"] == [{"n": 1210}]
    assert "was stopped after" in prompts[1] and "cross joins" in prompts[1]
    assert out["failed_attempts"][0]["stage"] == "execute"


def test_truncation_is_reported_to_the_caller_and_the_answer_model():
    drafts("SELECT order_id, status FROM orders ORDER BY order_id")
    chat = count_chat_calls()
    out = ex.graph.invoke({"question": "List all orders"})
    assert out["truncated"] is True and out["row_count"] == ex.MAX_ROWS == len(out["rows"])
    assert "truncated" in chat[0] and f"first {ex.PROMPT_ROWS} of {ex.MAX_ROWS}" in chat[0]


def test_last_month_means_october_2026_in_the_data():
    # Pins the date convention the prompt promises, using the exact range the model is told to use.
    prompts = drafts(
        "SELECT c.region, COUNT(*) AS returned_orders FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE o.status = 'returned' AND o.order_date >= '2026-10-01' AND o.order_date < '2026-11-01' "
        "GROUP BY c.region ORDER BY returned_orders DESC, c.region"
    )
    count_chat_calls()
    out = ex.graph.invoke({"question": "Which region had the most returned orders last month?"})
    assert "'2026-10-01'" in prompts[0]
    assert out["rows"] == [
        {"region": "Europe", "returned_orders": 1},
        {"region": "North America", "returned_orders": 1},
        {"region": "UK & Ireland", "returned_orders": 1},
    ], "a three-way tie: the answer prompt tells the model to say so"


# --------------------------------------------------------------------------- a reused thread behaves like a fresh one
NO_SUCH_ROWS = "SELECT order_id FROM orders WHERE order_date > '2030-01-01'"
CUSTOMERS_PER_REGION = "SELECT region, COUNT(*) AS n FROM customers GROUP BY region ORDER BY n DESC"


def writer_by_question():
    """A stateless scripted writer (the same prompt gets the same draft on any thread).

    "Delete..." -> a statement the guard rejects every time. "Empty..." -> a query that matches nothing every time.
    "Fixable..." -> matches nothing at first and a good query once the prompt carries an earlier attempt.
    Anything else -> a good query. Returns the prompts it was shown."""
    seen: list[str] = []

    def handler(text: str):
        seen.append(text)
        question = re.search(r"Question: (.*)", text).group(1)
        if question.startswith("Delete"):
            sql = "DELETE FROM orders"
        elif question.startswith("Empty") or (question.startswith("Fixable") and "Earlier attempts" not in text):
            sql = NO_SUCH_ROWS
        else:
            sql = CUSTOMERS_PER_REGION
        return ex.SqlDraft(sql=sql, rationale="scripted")

    set_structured_handler(ex.SqlDraft, handler)
    return seen


@pytest.mark.parametrize(
    "first, second",
    [
        ("Delete all the orders", "How many customers per region?"),  # three guard rejections, then a clean question
        ("Empty: orders placed after 2030?", "Fixable: orders placed after 2030?"),  # two empty results, then one that needs repair
        ("How many customers per region?", "Delete all the orders"),
    ],
    ids=["failed-then-fine", "empty-then-repairable", "fine-then-failed"],
)
def test_a_second_question_on_the_same_thread_is_the_same_as_on_a_fresh_thread(first, second):
    writer_by_question()
    count_chat_calls()
    reused, fresh = run_twice(lambda: ex.build_graph(checkpointer=InMemorySaver()), {"question": first}, {"question": second})
    assert normalise(reused.output) == normalise(fresh.output)


def test_failures_of_an_earlier_question_are_not_shown_to_the_writer_or_counted_towards_giving_up():
    prompts = writer_by_question()
    count_chat_calls()
    g = ex.build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    assert g.invoke({"question": "Empty: orders placed after 2030?"}, cfg)["status"] == "no_rows"
    before = len(prompts)
    out = g.invoke({"question": "Fixable: orders placed after 2030?"}, cfg)
    assert "Earlier attempts" not in prompts[before], "the first prompt of a new question carries no old failures"
    assert out["status"] == "answered" and out["attempts"] == 2, "one empty result is repaired, not 'repeated'"
    assert [f["stage"] for f in out["failed_attempts"]] == ["empty"]

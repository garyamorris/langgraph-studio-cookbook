"""One front door for staff questions: handbook, database, small talk, or ask back.

PROBLEM
    Staff put very different questions into the same box. "How do I use Send?" needs the written handbooks (LangGraph
    and Lumen Home policies), "how many orders last month?" needs the shop database, "thanks!" needs neither, and
    "what about last month?" needs a question back, because nobody can tell last month's *what*. A pipeline that
    always searches documents, or always writes SQL, gets most of these wrong. Misrouting wastes time and model spend,
    and guessing at an ambiguous question produces a confident answer to a question nobody asked.

DESIGN
    One cheap classification step decides the route; the expensive specialists are reused, not rewritten.

      * ``classify`` (structured output) returns ``route`` in {handbook, database, small_talk, clarify}, a one-line
        ``reason``, which knowledge base a handbook question belongs to, and, for ``clarify``, ONE specific question to
        ask. The prompt says to prefer answering over asking: a vague but answerable question is not ambiguous.
      * ``handbook`` runs the compiled ``corrective_rag`` graph (grade, rewrite, abstain, verify citations) and
        ``database`` runs the compiled ``sql_analyst`` graph (guarded text-to-SQL). Their state schemas are not the
        parent's, so each is wrapped in a small function that maps parent state IN (the question, the knowledge base)
        and the subgraph's result back OUT (``answer`` plus a ``details`` summary). Nothing from their internals
        leaks into the parent state.
      * ``small_talk`` is one short model reply with no factual claims about the business.
      * ``clarify`` calls ``interrupt()`` with the question to ask. The answer is appended to the question and the
        graph classifies again. Clarifying is bounded: after ``MAX_CLARIFICATIONS = 2`` the classifier is called with
        a schema that no longer contains ``clarify`` at all, so it must pick the best of the other three routes. The
        bound lives in the schema the model is given, not in a hope that it stops asking.
      The downstream specialists receive the original question plus the clarification answers as one text, so they
      see exactly what the user said.

GRAPH
    START -> start_run (forget old clarifications) -> classify
    classify --handbook----> handbook (corrective_rag subgraph) ---> END
    classify --database-----> database (sql_analyst subgraph) ------> END
    classify --small_talk---> small_talk ---------------------------> END
    classify --clarify------> clarify (interrupt) -> classify          (the loop, at most 2 times)

LANGGRAPH FEATURES
    * subgraphs with DIFFERENT state schemas, wrapped in functions that translate state in and out
    * ``interrupt()`` for a clarifying question, resumed with ``Command(resume=...)`` (needs a checkpointer; Studio
      provides one). Nothing with a side effect runs before the interrupt, because the node restarts from its first
      line on resume
    * conditional edges with a path map; a loop (``clarify`` -> ``classify``) bounded through the schema offered
      to the model
    * ``operator.add`` reducer so clarifications accumulate, emptied with ``Overwrite([])`` by ``start_run`` so a new question
      on a reused Studio thread does not inherit them; input/output schemas
    * ``build_graph`` builds the two subgraphs inside itself, compiled without their own checkpointer, so they inherit
      whatever persistence the parent is compiled with

MODELS AND DATA
    OpenAI chat (fast model) for classification and small talk; the subgraphs bring their own models: the OpenAI
    Decisions API for relevance grading in ``corrective_rag`` and the default chat model for SQL in ``sql_analyst``.
    Data: the shop database and the ``langgraph`` / ``support`` knowledge bases, through those subgraphs. Offline, the
    stand-in classifier picks routes by hashing the prompt, so routes are arbitrary; ``EXAMPLE_INPUT`` was chosen so
    that this arbitrary choice is a clarification, which keeps the offline smoke run exercising the interrupt. If you
    edit the prompts, re-run ``scripts/check_example.py adaptive_qa_router`` and re-pick the input if the interrupt
    disappears (a test fails when it does).

TRY IT IN STUDIO
    1. ``{"question": "what about last month, then?"}`` - the run pauses at ``clarify`` and shows the question it
       wants answered. Resume with a value such as ``"Total orders placed"`` (a JSON string). Watch ``classify`` run
       again and send the combined text ("what about last month, then? Total orders placed") to ``database``.
    2. ``{"question": "How do I pause a graph and wait for a human to approve?"}`` -> ``handbook``. Open the
       ``handbook`` node to see the subgraph's grading and citations in ``details``.
    3. ``{"question": "Which region had the most returned orders in the last 6 months?"}`` -> ``database``;
       ``details`` carries the SQL, the rows and the repair history.
    4. ``{"question": "thanks, that helped!"}`` -> ``small_talk`` (neither specialist runs).
    5. ``{"question": "How long do refunds take?"}`` -> ``handbook`` with the ``support`` knowledge base.
    To see the cap, answer each clarification with something equally vague; after two, the classifier can no longer ask
    and must pick a route.

EXTEND IT
    * Add a ``tools`` route (an agent with a calculator) and let the classifier choose it.
    * Keep the conversation in ``messages`` so a follow-up like "and for Europe?" can reuse the previous route.
    * Replace the classifier with a Decisions API ``choice`` question to get calibrated route probabilities, and
      clarify automatically when the top route is below a threshold.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.examples import corrective_rag, sql_analyst

META = ExampleMeta(
    id="adaptive_qa_router",
    title="One front door for staff questions (adaptive routing)",
    category="RAG & knowledge",
    problem="Send each staff question to the handbook, the database, a quick reply or a clarifying question, and "
    "refuse to guess when the question is ambiguous.",
    features=[
        "subgraphs with different state schemas",
        "interrupt() for clarification",
        "conditional routing",
        "bounded loop via restricted schema",
        "reuse of other examples",
        "input/output schemas",
    ],
    models=["OpenAI chat", "OpenAI Decisions API"],
    datasets=["store.sqlite", "kb/langgraph", "kb/support"],
    difficulty="advanced",
)

# Chosen so the OFFLINE stand-in classifier asks for clarification and then routes the answer to the database (its
# routes come from hashing the prompt). test_default_smoke_run_exercises_the_interrupt fails if a prompt edit changes that.
EXAMPLE_INPUT = {"question": "what about last month, then?"}
EXAMPLE_RESUMES = ["Total orders placed", "How many orders were placed?"]

MAX_CLARIFICATIONS = 2
DB_ROWS_IN_DETAILS = 20


# --------------------------------------------------------------------------- schemas
class RouteDecision(BaseModel):
    """Where one staff question should go."""

    route: Literal["handbook", "database", "small_talk", "clarify"]
    knowledge_base: Literal["langgraph", "support"] = Field(
        description="Used only when route is handbook: 'langgraph' (the LangGraph framework handbook) or 'support' (Lumen Home policies and product help)"
    )
    reason: str = Field(description="One short sentence on why this route")
    clarifying_question: str | None = Field(
        default=None, description="Only for route clarify: ONE specific question, answerable in a few words, offering the likely options"
    )


class RouteGuess(BaseModel):
    """The same decision once clarifying is no longer allowed: no 'clarify' route exists."""

    route: Literal["handbook", "database", "small_talk"]
    knowledge_base: Literal["langgraph", "support"] = Field(
        description="Used only when route is handbook: 'langgraph' (the LangGraph framework handbook) or 'support' (Lumen Home policies and product help)"
    )
    reason: str = Field(description="One short sentence on why this is the best guess")


class RouterIn(TypedDict):
    question: str


class RouterOut(TypedDict):
    question: str
    effective_question: str  # the question plus the clarification answers: what the specialist received
    route: str
    reason: str
    clarifications: list[dict]
    answer: str
    details: dict  # a summary of the specialist subgraph's own output
    mode: str


class RouterState(RouterIn, total=False):
    clarifications: Annotated[list[dict], operator.add]  # {"ask": ..., "answer": ...}
    route: str
    reason: str
    knowledge_base: str
    clarifying_question: str | None
    effective_question: str
    answer: str
    details: dict
    mode: str


def effective_question(state: RouterState) -> str:
    """The user's question followed by each clarification answer, in order."""
    parts = [state["question"].strip()] + [c["answer"] for c in state.get("clarifications", []) if c["answer"]]
    return " ".join(parts)


# --------------------------------------------------------------------------- classify / clarify
CLASSIFY_PROMPT = """\
You are the front door for Lumen Home staff questions. Pick the ONE route that should handle the question.

ROUTES
- handbook: a how-to or concept question answered by written documentation. Set knowledge_base to "langgraph" for
  questions about the LangGraph framework (state, nodes, routing, Send, subgraphs, checkpointers, interrupts,
  memory/store, streaming, Studio, retries, prebuilt agents) and to "support" for Lumen Home policies and product help
  (returns, refunds, warranty, shipping, wifi setup, firmware, privacy, safety, billing, account).
- database: needs numbers or records from the shop database: counts, totals, rankings, trends or lists about orders,
  customers, products, revenue, returns, channels or regions.
- small_talk: a greeting, thanks, or a question about what this assistant can do. Never a factual question.
- clarify: ask ONE question back, only when the question cannot be routed or answered without it: a missing subject or
  measure ("what about last month?"), an unresolved reference ("and the other one?"), or a request that could equally be
  a database lookup or a documentation question with different answers.

Prefer answering over asking. A short or informal question that clearly belongs to one route is NOT ambiguous; never
clarify it. Write reason as one short sentence. For clarify, clarifying_question is a single specific question a
colleague can answer in a few words, ideally offering the likely options."""

NO_CLARIFY = (
    "\n\nThe user has already been asked {n} clarifying question(s). You may NOT ask another: choose the most likely of "
    "handbook, database or small_talk, using the clarification answers below."
)


def render(state: RouterState) -> str:
    text = f"Question: {state['question']}"
    for i, c in enumerate(state.get("clarifications", []), 1):
        text += f"\nClarification {i}: you asked {c['ask']!r}; the user answered {c['answer']!r}"
    return text


def start_run(state: RouterState) -> dict:
    """Forget the clarifications of an earlier run on this thread (Studio keeps submitting runs on the same thread).

    ``classify`` is also where ``clarify`` loops back to, so it cannot do this itself; without this node the next
    question would be glued to the old answers ("Hello there" + "revenue") and start with its budget partly spent."""
    return {"clarifications": Overwrite([])}


def classify(state: RouterState) -> dict:
    question = (state.get("question") or "").strip()
    if not question:
        raise ValueError("Provide a non-empty `question`.")
    asked = len(state.get("clarifications", []))
    may_clarify = asked < MAX_CLARIFICATIONS
    schema = RouteDecision if may_clarify else RouteGuess
    prompt = CLASSIFY_PROMPT + ("" if may_clarify else NO_CLARIFY.format(n=asked))
    d = structured_llm(schema, "fast").invoke([SystemMessage(prompt), HumanMessage(render({**state, "question": question}))])
    update: dict[str, Any] = {"question": question, "route": d.route, "reason": d.reason, "knowledge_base": d.knowledge_base, "mode": mode_label()}
    if d.route == "clarify":
        update["clarifying_question"] = (d.clarifying_question or "").strip() or "Could you say a bit more about what you need?"
    return update


def clarify(state: RouterState) -> dict:
    """Pause and ask the user ONE question. On resume this node restarts, so nothing before ``interrupt`` has effects."""
    asked = len(state.get("clarifications", []))
    ask = state["clarifying_question"]
    answer = interrupt(
        {
            "ask": ask,
            "original_question": state["question"],
            "why": state.get("reason", ""),
            "clarification": asked + 1,
            "of": MAX_CLARIFICATIONS,
        }
    )
    return {"clarifications": [{"ask": ask, "answer": str(answer).strip()}]}


def after_classify(state: RouterState) -> Literal["handbook", "database", "small_talk", "clarify"]:
    return state["route"]  # type: ignore[return-value]


# --------------------------------------------------------------------------- specialists
def make_handbook(rag):
    """Wrap the corrective_rag subgraph: parent state in, subgraph result out."""

    def handbook(state: RouterState) -> dict:
        question = effective_question(state)
        out = rag.invoke({"question": question, "corpus": state.get("knowledge_base") or "langgraph"})
        details = {
            "subgraph": "corrective_rag",
            "corpus": out["corpus"],
            "status": out["status"],
            "citations": out["citations"],
            "dropped_citations": out["dropped_citations"],
            "sources": out["sources"],
            "queries": out["queries"],
            "rewrites": out["rewrites"],
        }
        return {"effective_question": question, "answer": out["answer"], "details": details}

    return handbook


def make_database(sql):
    """Wrap the sql_analyst subgraph: parent state in, subgraph result out."""

    def database(state: RouterState) -> dict:
        question = effective_question(state)
        out = sql.invoke({"question": question})
        details = {
            "subgraph": "sql_analyst",
            "status": out["status"],
            "sql": out["sql"],
            "columns": out["columns"],
            "rows": out["rows"][:DB_ROWS_IN_DETAILS],
            "row_count": out["row_count"],
            "truncated": out["truncated"],
            "attempts": out["attempts"],
            "failed_attempts": out["failed_attempts"],
        }
        return {"effective_question": question, "answer": out["answer"], "details": details}

    return database


SMALL_TALK_PROMPT = (
    "You are the assistant at the front door of Lumen Home's internal question tool. Reply in at most two short "
    "sentences. For a greeting or thanks, be warm and brief. If asked what you can do: you can answer questions from "
    "the LangGraph handbook and the Lumen Home support policies, and look up shop data (orders, customers, products, "
    "returns, revenue). Do not state any fact about the business, its products or its data; if the user wants one, "
    "invite them to ask it directly."
)


def small_talk(state: RouterState) -> dict:
    question = effective_question(state)
    reply = get_llm("fast").invoke([SystemMessage(SMALL_TALK_PROMPT), HumanMessage(question)])
    return {"effective_question": question, "answer": str(reply.text).strip(), "details": {"subgraph": None}}


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    # Compiled WITHOUT their own checkpointer: invoked inside a node, they inherit the parent's persistence.
    rag = corrective_rag.build_graph()
    sql = sql_analyst.build_graph()
    g = StateGraph(RouterState, input_schema=RouterIn, output_schema=RouterOut)
    g.add_node("start_run", start_run)
    g.add_node("classify", classify)
    g.add_node("clarify", clarify)
    g.add_node("handbook", make_handbook(rag))
    g.add_node("database", make_database(sql))
    g.add_node("small_talk", small_talk)
    g.add_edge(START, "start_run")
    g.add_edge("start_run", "classify")
    g.add_conditional_edges(
        "classify",
        after_classify,
        {"handbook": "handbook", "database": "database", "small_talk": "small_talk", "clarify": "clarify"},
    )
    g.add_edge("clarify", "classify")
    for name in ("handbook", "database", "small_talk"):
        g.add_edge(name, END)
    return g.compile(**compile_kwargs)


graph = build_graph()

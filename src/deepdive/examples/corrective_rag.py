"""Ask the LangGraph handbook (corrective RAG that can say "not covered").

PROBLEM
    Keyword retrieval returns articles that share words with the question, not articles that answer it. Ask the
    handbook "How do I pause a graph and wait for a human to approve?" and BM25 ranks the article on Send fan-out,
    the one on checkpointers and the one on subgraphs above the article on ``interrupt()`` - all plausible, only one
    right. A naive RAG graph pastes the top hits into a prompt and writes a fluent, confident answer from whatever
    it got, including from the wrong articles. For documentation that people act on, "I could not find that" is a far
    better outcome than a confident answer built on the wrong page.

DESIGN
    Retrieval is cheap and unreliable, so the graph *checks* what it retrieved before it writes anything, and it
    treats "nothing relevant" as a legitimate result instead of something to paper over.

      * ``retrieve`` takes the top 4 BM25 hits (a deliberately wide net).
      * ``grade`` asks one question about each hit - "does this article contain information needed to answer the
        question?" - and sends ALL of them in a single Decisions API call (one ``predicate`` per article). The API
        returns a calibrated probability per article, not a yes/no, so the cut-off is an explicit, tunable number:
        ``RELEVANCE_THRESHOLD = 0.6``. The probabilities are kept in the output, so you can see how close a
        discarded article was. A refused answer counts as "not relevant".
      * If at least one article passes, ``generate`` writes the answer from ONLY those articles.
      * If none passes, ``rewrite`` (an LLM with structured output) writes a new keyword query using the
        documentation's own vocabulary, and the graph retrieves again. At most ``MAX_REWRITES = 2`` rewrites. A
        rewrite that repeats an earlier query ends the search at once (nothing new would be found).
      * If the budget is spent, ``abstain`` replies "not covered", lists the closest articles with their grader
        probabilities, and never calls the generator. This is deterministic text: the model is never asked to
        write from context it has just been told is irrelevant.
      * ``verify_citations`` is a deterministic post-check. The generator is told to cite ``[article-id]``; any cited
        id that was not among the articles it was given is stripped from the answer and reported in
        ``dropped_citations``. Any lowercase slug in square brackets counts as a citation attempt (so a shortened id
        such as ``[checkpointers]`` cannot slip through), which is why the generator is told to use brackets for
        nothing else; code spans and ``[text](link)`` are left alone. An answer left with no valid citation is
        flagged ``answered_uncited`` and carries a caveat, because it cannot be checked against any source.
      Grading is always against the ORIGINAL question, never the rewritten query, so a rewrite cannot change what
      "relevant" means.

GRAPH
    START -> start_run (reset per-run state) -> retrieve -> grade
    grade   --some article scores >= 0.6--------> generate -> verify_citations -> END
    grade   --none pass, rewrites left----------> rewrite --new query--> retrieve      (the loop, at most 2 times)
    grade   --none pass, no rewrites left-------> abstain -> END
    rewrite --repeated an earlier query---------> abstain -> END

LANGGRAPH FEATURES
    * a retry loop (retrieve -> grade -> rewrite -> retrieve) with its bound held in state and enforced by the
      routing function; ``queries`` and ``grading`` accumulate across rounds through ``operator.add`` reducers, and
      ``start_run`` empties them with ``Overwrite([])`` so a new question on a reused Studio thread starts clean
    * conditional edges with explicit path maps, so the three exits of ``grade`` are named in Studio
    * input/output schemas (Studio shows ``question`` and ``corpus``; the result shows the evidence)
    * structured output (the rewrite) and the OpenAI Decisions API (batched grading) in one graph
    * a deterministic final node that audits model output

MODELS AND DATA
    Decisions API for the relevance grades (one call per retrieval round, one predicate per article), OpenAI chat
    for the rewrite (fast model) and the answer (default model). Data: the ``langgraph`` handbook (15 short
    articles; default) or the ``support`` policies (10 articles; internal-only ones are never retrieved). Offline,
    the grader returns arbitrary but stable probabilities and the chat model returns placeholder text, so an
    offline run shows the shape of the output, not good answers; ``mode`` in the output says which ran.

TRY IT IN STUDIO
    Open the ``Ready-made case`` dropdown, pick one and press Submit (or leave the form blank on a new thread: that runs
    the first one). Each case sets the question and the knowledge base:
      * ``How do I pause a graph and wait for a human to approve?`` - open the ``retrieve`` output, then ``grade``:
        compare the BM25 order with the probabilities (the interrupts article is retrieved 4th but is the one that
        should pass).
      * ``How can I let a person sign off before the workflow continues?`` - weak vocabulary overlap, so none of the
        first four hits is the interrupts article; expect a ``rewrite`` and a second round.
      * ``How do I stream tokens to a UI?`` - the easy case: one article matches, so there is nothing to rewrite.
      * ``What is the capital of France?`` - no article shares a word with it, so there is nothing to grade; expect up
        to two rewrites and then ``abstain`` with no generation.
      * ``How long do refunds take?`` (support policies) - a cited answer from the ``support`` knowledge base.
      * ``What refund amount needs manager approval?`` (support policies) - the answer exists only in an article marked
        "(internal)", which the retriever never returns; expect the grader to reject what it does return and the run to
        abstain instead of guessing.
    In the output, ``grading`` lists every round (query, articles, probabilities, kept or not), ``sources`` the
    articles the answer was allowed to use, and ``dropped_citations`` any citation that was removed.
    To ask your own, paste ``{"question": "How do I pause a graph and wait for a human to approve?", "corpus":
    "langgraph"}`` (``corpus`` is ``langgraph`` or ``support``).

EXTEND IT
    * Cache grades per (question, article) so a rewrite that re-retrieves a known article does not pay for it again.
    * Replace the BM25 retriever with embeddings plus BM25 and keep the same grade/rewrite/abstain wrapper.
    * Add a second Decisions API check after ``generate`` ("is every claim supported by the kept articles?") and
      route low-support answers to ``abstain``.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from deepdive.common.config import mode_label
from deepdive.common.data import kb_docs
from deepdive.common.decisions import decide, predicate
from deepdive.common.meta import ExampleMeta
from deepdive.common.models import get_llm, structured_llm
from deepdive.common.retrieval import get_retriever
from deepdive.common.scenarios import expand, scenario_field

META = ExampleMeta(
    id="corrective_rag",
    title="Ask the LangGraph handbook (corrective RAG)",
    category="RAG & knowledge",
    problem="Stop keyword retrieval from producing confident answers built on plausible but wrong articles, by "
    "grading what was retrieved, rewriting the query when needed, and abstaining when the knowledge base does not cover the question.",
    features=[
        "Decisions API batch grading",
        "retry loop with query rewriting",
        "abstain path",
        "deterministic post-check",
        "reducers across loop rounds",
        "structured output",
    ],
    models=["OpenAI Decisions API", "OpenAI chat"],
    datasets=["kb/langgraph", "kb/support"],
    difficulty="intermediate",
)

EXAMPLE_INPUT = {"question": "How do I pause a graph and wait for a human to approve?"}

# Ready-made questions for Studio's dropdown (see deepdive.common.scenarios). The first one runs on a blank form. Every
# case sets the same two keys (None = the default corpus, the LangGraph handbook). The retrieval facts in the labels are
# decided by the BM25 index and pinned by tests/test_corrective_rag.py; what the grader and the rewriter do with each
# question is only ever "watch for".
CASES: dict[str, dict] = {
    "How do I pause a graph and wait for a human to approve? -> BM25 ranks the right article 4th: watch grade pick it": {
        "question": "How do I pause a graph and wait for a human to approve?", "corpus": None},
    "How can I let a person sign off before the workflow continues? -> the right article misses the top 4: watch the rewrite": {
        "question": "How can I let a person sign off before the workflow continues?", "corpus": None},
    "How do I stream tokens to a UI? -> BM25 finds only the streaming article: watch it pass grading with no rewrite": {
        "question": "How do I stream tokens to a UI?", "corpus": None},
    "What is the capital of France? -> no article shares a word with it: watch the rewrites, then the abstain": {
        "question": "What is the capital of France?", "corpus": None},
    "How long do refunds take? (support) -> returns-and-refunds says 5 to 7 business days: watch the cited answer": {
        "question": "How long do refunds take?", "corpus": "support"},
    "What refund amount needs manager approval? (support) -> only the internal matrix says, and it is never retrieved: watch": {
        "question": "What refund amount needs manager approval?", "corpus": "support"},
}
SCENARIO = scenario_field(CASES)

TOP_K = 4  # articles retrieved per round
RELEVANCE_THRESHOLD = 0.6  # an article is kept when the grader's probability is >= this
MAX_REWRITES = 2  # query rewrites before the graph abstains (so at most MAX_REWRITES + 1 retrieval rounds)

CORPORA = {"langgraph": "LangGraph handbook", "support": "Lumen Home support policies"}


# --------------------------------------------------------------------------- schemas
class Rewrite(BaseModel):
    """A replacement search query."""

    query: str = Field(description="3-8 keywords or a short phrase in the documentation's own vocabulary; NOT a full sentence answer")
    reason: str = Field(description="One short sentence: which words you changed and why")


class RagIn(TypedDict, total=False):
    scenario: SCENARIO  # a ready-made question (and knowledge base); replaces the two fields below
    question: str
    corpus: Literal["langgraph", "support"]  # which knowledge base to search (default "langgraph")


class RagOut(TypedDict):
    question: str
    corpus: str
    status: str  # answered | answered_uncited | abstained
    answer: str
    citations: list[str]  # cited article ids that were verified against the kept articles
    dropped_citations: list[str]  # cited ids that were NOT among the kept articles (removed from the answer)
    sources: list[dict]  # the articles the answer was allowed to use, with grader probabilities
    grading: list[dict]  # one entry per retrieval round: query, every article with its probability, kept or not
    queries: list[str]
    rewrites: int
    threshold: float
    mode: str


class RagState(RagIn, total=False):
    query: str
    retrieved: list[dict]
    kept: list[str]
    queries: Annotated[list[str], operator.add]
    grading: Annotated[list[dict], operator.add]
    rewrites: int
    stalled: bool
    status: str
    answer: str
    citations: list[str]
    dropped_citations: list[str]
    sources: list[dict]
    raw_answer: str
    threshold: float
    mode: str


# --------------------------------------------------------------------------- helpers
def question_name(doc_id: str) -> str:
    """Name of the Decisions API predicate for one article (hyphens are not assumed to be legal in names)."""
    return "rel_" + re.sub(r"[^a-z0-9]+", "_", doc_id.lower()).strip("_")


def _corpus(state: RagState) -> str:
    corpus = state.get("corpus") or "langgraph"
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}; use one of {', '.join(CORPORA)}.")
    return corpus


def _tidy(text: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


_CITATION = re.compile(r"( ?)\[([a-z][a-z0-9]*(?:-[a-z0-9]+)*)\](?!\()")  # [article-id], but not [text](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)  # code is never touched: `values[key]` is not a citation


def verify_text(text: str, allowed: list[str]) -> tuple[str, list[str], list[str]]:
    """Strip citations of articles that were not provided. Returns (clean_text, valid_ids, dropped_ids), in order of use."""
    valid: list[str] = []
    dropped: list[str] = []

    def fix(m: re.Match[str]) -> str:
        cid = m.group(2)
        if cid in allowed:
            valid.append(cid)
            return m.group(0)
        dropped.append(cid)
        return ""

    parts = _CODE.split(text)  # odd entries are code spans
    cleaned = "".join(p if i % 2 else _CITATION.sub(fix, p) for i, p in enumerate(parts))
    return _tidy(cleaned), list(dict.fromkeys(valid)), list(dict.fromkeys(dropped))


# --------------------------------------------------------------------------- nodes
def start_run(state: RagState) -> dict:
    """Forget what an earlier run on this thread left behind (Studio keeps submitting runs on the same thread).

    ``retrieve`` is also where ``rewrite`` loops back to, so it cannot reset anything itself: without this node a new
    question would search with the old rewritten ``query`` and start with the rewrite budget already spent.

    This is the entry node, so it also resolves a picked ready-made question (or runs the first one on a blank form).
    A typed ``corpus`` with no question is real input too: ``retrieve`` gives it the "Provide a non-empty question" error.
    """
    picked = expand(
        state, CASES, has_input=lambda s: bool((s.get("question") or "").strip()) or s.get("corpus") is not None
    )
    return {**picked, "query": "", "rewrites": 0, "stalled": False, "kept": [], "queries": Overwrite([]), "grading": Overwrite([])}


def retrieve(state: RagState) -> dict:
    """Top-k BM25 hits for the current query (the original question on the first round)."""
    question = (state.get("question") or "").strip()
    if not question:
        raise ValueError("Provide a non-empty `question`.")
    corpus = _corpus(state)
    query = state.get("query") or question
    hits = get_retriever(corpus).search(query, k=TOP_K)
    retrieved = [{"id": h.doc.id, "title": h.doc.title, "text": h.doc.text, "bm25": round(h.score, 2)} for h in hits]
    return {"question": question, "corpus": corpus, "query": query, "retrieved": retrieved, "queries": [query], "rewrites": state.get("rewrites", 0)}


def grade(state: RagState) -> dict:
    """One batched Decisions API call: a predicate per retrieved article, graded against the ORIGINAL question."""
    retrieved = state["retrieved"]
    round_no = len(state.get("grading", [])) + 1
    probs: dict[str, float] = {}
    refused: set[str] = set()
    grader = "none (nothing retrieved)"
    if retrieved:
        corpus_text = "\n\n".join(f"[{a['id']}] {a['title']}\n{a['text']}" for a in retrieved)
        questions = [
            predicate(
                question_name(a["id"]),
                f"Consider ONLY the article [{a['id']}] ({a['title']}) in the input. Does it contain information that is "
                "needed to answer the QUESTION? Answer true only if the article states facts that help answer it. An "
                "article that shares keywords or covers an adjacent topic without answering the question is false.",
            )
            for a in retrieved
        ]
        res = decide(f"QUESTION\n{state['question']}\n\nCANDIDATE ARTICLES\n{corpus_text}", questions)
        grader = res.model
        for a in retrieved:
            name = question_name(a["id"])
            if res.refused(name):
                refused.add(a["id"])
            probs[a["id"]] = res.p(name, default=0.0)  # a refusal or a missing answer counts as "not relevant"
    articles = [
        {
            "id": a["id"],
            "title": a["title"],
            "bm25": a["bm25"],
            "probability": round(probs[a["id"]], 3),
            "kept": probs[a["id"]] >= RELEVANCE_THRESHOLD,
            "refused": a["id"] in refused,
        }
        for a in retrieved
    ]
    kept = [a["id"] for a in sorted((a for a in articles if a["kept"]), key=lambda a: -a["probability"])]
    entry = {"round": round_no, "query": state["query"], "grader": grader, "articles": articles}
    return {"kept": kept, "grading": [entry]}


def after_grade(state: RagState) -> Literal["generate", "rewrite", "abstain"]:
    if state["kept"]:
        return "generate"
    return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else "abstain"


REWRITE_PROMPT = (
    "You help a BM25 keyword search find the right article in the {kb}. BM25 matches words, not meaning, so synonyms "
    "and paraphrases fail. The articles retrieved so far did not contain what is needed to answer the question.\n"
    "Write ONE new search query of 3-8 keywords that uses the vocabulary the documentation itself would use for this "
    "concept (API names, technical terms). It must differ from every query already tried. Do not answer the question.\n\n"
    "Available article titles:\n{titles}"
)


def rewrite(state: RagState) -> dict:
    """LLM proposes a better keyword query. A query that was already tried cannot find anything new, so it stalls the search."""
    kb = CORPORA[state["corpus"]]
    titles = "\n".join(f"- {d.title}" for d in kb_docs(state["corpus"]) if d.audience == "public")
    tried = state["queries"]
    rejected = [a["title"] for rd in state["grading"] for a in rd["articles"]]
    user = (
        f"Question: {state['question']}\nQueries already tried: {tried}\n"
        f"Articles retrieved and judged not relevant: {list(dict.fromkeys(rejected)) or 'none (no article matched)'}"
    )
    out = structured_llm(Rewrite, "fast").invoke([SystemMessage(REWRITE_PROMPT.format(kb=kb, titles=titles)), HumanMessage(user)])
    new = out.query.strip()
    seen = {re.sub(r"\s+", " ", q).strip().lower() for q in tried}
    return {"query": new, "rewrites": state.get("rewrites", 0) + 1, "stalled": re.sub(r"\s+", " ", new).lower() in seen or not new}


def after_rewrite(state: RagState) -> Literal["retrieve", "abstain"]:
    return "abstain" if state.get("stalled") else "retrieve"


GENERATE_PROMPT = (
    "You answer questions about the {kb} using ONLY the articles below.\n"
    "- End every factual sentence with the id of the article it comes from in square brackets, for example "
    "[{example}]. Use only the ids listed below, exactly as written.\n"
    "- If the articles only partly answer the question, answer the part they cover and state plainly what they do "
    "not cover. Never fill a gap from general knowledge.\n"
    "- Use square brackets for citations and nothing else.\n"
    "- At most 120 words. No preamble.\n\nARTICLES\n{articles}"
)


def generate(state: RagState) -> dict:
    """Answer from the kept articles only (the irrelevant hits are not in the prompt at all)."""
    by_id = {a["id"]: a for a in state["retrieved"]}
    kept = [by_id[i] for i in state["kept"]]
    system = GENERATE_PROMPT.format(
        kb=CORPORA[state["corpus"]],
        example=kept[0]["id"],
        articles="\n\n".join(f"[{a['id']}] {a['title']}\n{a['text']}" for a in kept),
    )
    msg = get_llm().invoke([SystemMessage(system), HumanMessage(state["question"])])
    return {"raw_answer": str(msg.text).strip()}


def verify_citations(state: RagState) -> dict:
    """Deterministic audit: remove citations of articles the generator was never given."""
    text, valid, dropped = verify_text(state["raw_answer"], state["kept"])
    status = "answered"
    if not valid:
        status = "answered_uncited"
        text += "\n\n(Note: this answer cites none of the retrieved articles, so it could not be checked against a source.)"
    final = state["grading"][-1]["articles"]
    sources = [{"id": a["id"], "title": a["title"], "probability": a["probability"]} for a in final if a["kept"]]
    sources.sort(key=lambda s: -s["probability"])
    return {
        "status": status,
        "answer": text,
        "citations": valid,
        "dropped_citations": dropped,
        "sources": sources,
        "threshold": RELEVANCE_THRESHOLD,
        "mode": mode_label(),
    }


def abstain(state: RagState) -> dict:
    """Honest "not covered" with the closest articles. Fixed text: nothing is generated from irrelevant context."""
    best: dict[str, dict] = {}
    for rd in state["grading"]:
        for a in rd["articles"]:
            if a["id"] not in best or a["probability"] > best[a["id"]]["probability"]:
                best[a["id"]] = a
    closest = sorted(best.values(), key=lambda a: (-a["probability"], -a["bm25"]))[:3]
    kb = CORPORA[state["corpus"]]
    n = len(state["queries"])
    text = f"I could not find this in the {kb}, so I am not going to guess. I searched {n} time{'s' if n != 1 else ''} ({'; '.join(repr(q) for q in state['queries'])})."
    if closest:
        text += " The closest articles, with the grader's probability that each one answers the question, were:\n" + "\n".join(
            f"- {a['title']} [{a['id']}] (p={a['probability']:.2f})" for a in closest
        )
    else:
        text += " No article shares any vocabulary with the question."
    text += f"\nNone reached the {RELEVANCE_THRESHOLD:.2f} relevance threshold. Try rephrasing, or check the source directly."
    return {
        "status": "abstained",
        "answer": text,
        "citations": [],
        "dropped_citations": [],
        "sources": [],
        "threshold": RELEVANCE_THRESHOLD,
        "mode": mode_label(),
    }


# --------------------------------------------------------------------------- graph
def build_graph(**compile_kwargs):
    g = StateGraph(RagState, input_schema=RagIn, output_schema=RagOut)
    g.add_node("start_run", start_run)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify_citations", verify_citations)
    g.add_node("abstain", abstain)
    g.add_edge(START, "start_run")
    g.add_edge("start_run", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"})
    g.add_conditional_edges("rewrite", after_rewrite, {"retrieve": "retrieve", "abstain": "abstain"})
    g.add_edge("generate", "verify_citations")
    g.add_edge("verify_citations", END)
    g.add_edge("abstain", END)
    return g.compile(**compile_kwargs)


graph = build_graph()

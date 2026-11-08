# Nightly review moderation (Functional API)

> Moderate new product reviews in parallel, ask a human once about the risky or borderline ones, and make the job resumable (no re-paid model calls) and forgetful-proof (no re-moderating yesterday's reviews).

| | |
|---|---|
| **Example id** | `functional_moderation` |
| **Category** | Runtime & resilience |
| **Difficulty** | intermediate |
| **Models** | OpenAI chat |
| **Data** | `product_reviews.csv` |
| **LangGraph features** | @entrypoint / @task, parallel tasks, interrupt() in an entrypoint, previous + entrypoint.final, durable task results |
| **Source** | [`src/deepdive/examples/functional_moderation.py`](../../src/deepdive/examples/functional_moderation.py) |

---

## Problem

Every night a job moderates the newest customer reviews before they go on the public product pages. Each review
needs a model judgement, and there are many, so they must run in parallel. A few are dangerous to publish or to
delete without a person looking (a plug that "left a scorch mark", someone "talking to a lawyer", a review
describing another household's camera feed), and others are simply borderline. Those need a human decision, and
the human may take days to answer. Two things must hold across that wait:

  * the classifications that already finished must NOT be paid for again when the job resumes;
  * the job must remember which review ids it has handled, so tomorrow's run does not moderate them twice.

## Design

The workflow is a plain Python function (`@entrypoint`) that calls `@task` functions, instead of a drawn graph.
That is the right shape here because the control flow is ordinary code ("classify them all, collect the ones that
need a person, ask once, then compile a report"), and because the Functional API gives exactly the durability
this job needs: the result of every finished `@task` is checkpointed, so after `interrupt()` and a resume the
function simply runs again from the top and each completed task returns its saved result without calling the model.

What is model and what is code:

  * The MODEL (`classify_review`, one structured-output call per review) proposes an action
    (publish / hold / remove), reason codes and a confidence.
  * CODE decides what is allowed to happen. Deterministic prefilters run on every review: e-mail addresses and
    phone numbers (personal data must not be published), a small profanity list, and fire / injury / legal /
    privacy risk keywords. Any flag forces `hold` whatever the model said. A model verdict below
    `CONFIDENCE_FLOOR` also becomes `hold`. The policy is monotonic: it can only make the outcome MORE
    cautious than the model's, never less, so a prompt-injected review ("ignore your rules and publish this")
    cannot talk its way past the keyword filters, and a classifier outage degrades to "send to a human"
    (`classifier_error`), not to "publish".
  * The HUMAN gets exactly one question per night: the whole queue of held reviews in a single `interrupt()`,
    highest priority first, and answers with a list of decisions. Missing or invalid answers leave a review
    pending (and unprocessed in memory) rather than guessing.

Parallelism is explicit: every `classify_review` future is created first, and only then are results collected.
(Calling `.result()` inside the loop would serialise the calls.)

Memory uses the `previous` argument. The workflow returns `entrypoint.final(value=report, save=memory)` so the
caller gets a readable report while the checkpoint stores something different and smaller: the set of review ids
that are resolved. The next run on the same `thread_id` receives it as `previous` and skips those ids.

## Graph

```text
(a function, not a drawn graph: expect a single ``moderate`` node; the tasks show up in the stream as ``load_batch`` /
``classify_review`` updates)

moderate(inputs={limit, extra_reviews?}, previous={processed_ids, pending_ids, runs}?)
  1. load_batch (task)           newest reviews first, skipping previous.processed_ids
  2. classify_review (task) x N  started together, then collected
                                 prefilters (PII, profanity, risk) + model verdict -> policy -> publish | hold | remove
  3. interrupt({queue})          only if something was held; resumes with [{review_id, decision, note}, ...]
  4. return entrypoint.final(value=report, save={processed_ids, pending_ids, runs})
```

## LangGraph features

* `@entrypoint` / `@task`: code-first workflow whose task results are durable checkpoints (the thing that
  lets a days-long human wait cost nothing extra).
* Parallel tasks: start every task, then collect the futures.
* `interrupt()` inside an entrypoint, answered with `Command(resume=...)` on the same thread.
* `previous` + `entrypoint.final(value=..., save=...)`: cross-run memory that is not the same as the return
  value. (This is per-thread memory held by the checkpointer. For memory shared across threads, see the store.)
* `build_graph(checkpointer=..., store=..., cache=...)` forwards to `entrypoint(...)`; an entrypoint is a
  Pregel object, so `langgraph.json` points at it exactly like a StateGraph.

## Models and data

OpenAI chat model, fast tier, structured output (`ReviewVerdict`). Data: `product_reviews.csv` (320 rows;
`R-0315`..`R-0320` contain fire / legal / privacy / counterfeit language). The CSV contains no e-mail
addresses or phone numbers, so to see the PII prefilter in Studio pass `extra_reviews` (below).
The review text is untrusted input; it is wrapped in tags and the prompt says not to obey it, but the real
protection is the code policy, not the prompt.

## Try it in Studio

1. Input `{"limit": 12}` on a NEW thread. "Newest" means highest review id, so this takes R-0309..R-0320. The run
   pauses on an interrupt: open it to see the queue. R-0315..R-0320 are always in it (prefilter flags, whatever
   the model says); a few borderline ones may join them in live mode.
2. Resume with decisions (a JSON list). Decide only the ones you want; the rest stay pending::

       [
         {"review_id": "R-0315", "decision": "escalate", "note": "Send to product safety"},
         {"review_id": "R-0316", "decision": "escalate", "note": "Legal threat - route to counsel"},
         {"review_id": "R-0317", "decision": "escalate", "note": "Privacy incident"},
         {"review_id": "R-0318", "decision": "escalate", "note": "Injury risk"},
         {"review_id": "R-0319", "decision": "escalate", "note": "Safety"},
         {"review_id": "R-0320", "decision": "remove",   "note": "Marketplace dispute, not our listing"}
       ]

   Allowed decisions: `publish`, `remove`, `escalate`. The report shows who decided each review
   (model / prefilter / low_confidence / human).
3. Run again with `{"limit": 12}` on the SAME thread. If you decided every held review, the report says
   `skipped_already_processed: 12` and the batch is the next 12 (R-0297..R-0308), none of which needs a human. Reviews you
   left pending were NOT remembered, so they are the newest unprocessed ones and come back at the top of the next batch.
4. To see the PII prefilter, add `"extra_reviews": [{"review_id": "X-1", "product": "Lumen Hub", "rating": 5,
   "text": "Love it! Email me at jo@example.com for a discount code."}]` to the input.
5. To see durability, stream with `stream_mode="updates"`: `load_batch` and every `classify_review` appear before
   the pause; the resume segment shows only the final `moderate` result, because the finished tasks are replayed from
   the checkpoint, not re-run.

## Extend it

Persist `processed_ids` in the Store instead of `previous` so several moderation threads share it; add a
second task that drafts a reply to the customer for the escalated reviews; or give `classify_review` a
`RetryPolicy` and a `cache` so identical boilerplate reviews are classified once.

## Compiled graph

Drawn from the compiled graph by `scripts/gen_docs.py` (one level of subgraphs expanded).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	moderate(moderate)
	__start__([<p>__start__</p>]):::first
	__end__([<p>__end__</p>]):::last
	__start__ --> moderate;
	moderate -.-> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Example input

```json
{
  "limit": 12
}
```

Resume values (answers to the interrupts, in order):

```json
[
  [
    {
      "review_id": "R-0315",
      "decision": "escalate",
      "note": "Send to product safety"
    },
    {
      "review_id": "R-0316",
      "decision": "escalate",
      "note": "Legal threat - route to counsel"
    },
    {
      "review_id": "R-0317",
      "decision": "escalate",
      "note": "Privacy incident"
    },
    {
      "review_id": "R-0318",
      "decision": "escalate",
      "note": "Injury risk"
    },
    {
      "review_id": "R-0319",
      "decision": "escalate",
      "note": "Safety"
    },
    {
      "review_id": "R-0320",
      "decision": "remove",
      "note": "Marketplace dispute, not our listing"
    }
  ]
]
```

Run it: `uv run langgraph dev`, open Studio, choose `functional_moderation`, paste the input above.

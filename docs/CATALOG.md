# Example catalog

23 examples. Each one solves a concrete problem; open its page for the problem statement, the design rationale, a diagram of the compiled graph, and the exact input to paste into Studio.

## Core workflows

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`draft_critic_loop`](examples/draft_critic_loop.md) | Produce a launch announcement that satisfies hard rules (length, required facts, banned claims) and a quality bar, revising until it passes or an explicit attempt cap returns the best draft, flagged as best-effort. | cycle with explicit cap, conditional edges, parallel critics + join edge, append-only reducer, input/output schemas | OpenAI chat |
| [`report_orchestrator`](examples/report_orchestrator.md) | Write a quarterly sales report whose sections are chosen from what the data shows, with every number traced back to computed facts and unsupported figures flagged. | Send (orchestrator-workers), operator.add reducer + order restoration, Overwrite reset, post-hoc verification node, input/output schemas | OpenAI chat |
| [`review_insights_parallel`](examples/review_insights_parallel.md) | Summarise a product's recent customer reviews (statistics, themes, risks) in parallel while guaranteeing that a safety, legal or privacy report is never missed or buried. | parallel fan-out, operator.add reducer + Overwrite reset, defer=True join, input/output schemas, deterministic + LLM union | OpenAI chat |
| [`ticket_router`](examples/ticket_router.md) | Route routine tickets to policy-grounded specialists, and escalate to a human with a fixed, not AI-written, customer message anything that a keyword lexicon or the model flags as a safety or privacy risk, or any ticket when the model call fails. | Command routing, input/output schemas, node factory, structured output, BM25 retrieval, fail-closed routing | OpenAI chat |

## Agents & tools

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`plan_execute_analyst`](examples/plan_execute_analyst.md) | Answer dependent, multi-step sales questions with exact numbers by planning against a closed tool registry, executing deterministically, replanning on observations, and stopping at a hard step budget. | plan-as-state, loop with replanning and hard caps, closed tool registry, tool errors as observations, structured output | OpenAI chat |
| [`sales_react_agent`](examples/sales_react_agent.md) | Let an analyst chat with the sales data and get exact numbers from tools, with tool errors recovered, the tool loop capped and the chat history trimmed. | add_messages + RemoveMessage trimming, ToolNode + handle_tool_errors, tools_condition loop, bounded agent loop, token streaming | OpenAI chat (tool calling) |
| [`sql_analyst`](examples/sql_analyst.md) | Let non-technical staff query the shop database in plain English without hallucinated values, wrong joins or unsafe SQL ever reaching the database. | guarded repair loop, conditional edges with named repair edge, reducer-accumulated attempt history, input/output schemas, deterministic safety layer around a model, structured output | OpenAI chat |

## Multi-agent

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) | Turn eight fictional stocks into one risk-checked daily trade blotter: research every name in parallel, hear a bull and a bear case, let a risk committee send only the breaching names back for resizing, and plan the execution, with code guaranteeing the limits. | Send map-reduce, custom keyed reducer, Command fan-out / skip, parallel branches + join, selective re-dispatch loop (Send from a conditional edge), loop cap in state, input/output schemas, RetryPolicy | OpenAI chat, OpenAI Decisions API |
| [`research_supervisor`](examples/research_supervisor.md) | Produce a one-page company brief in which numbers come from code, interpretation from specialist teams, and every sentence is traceable to a fact, with a supervisor that cannot loop or skip the brief. | hierarchical supervisor, Command routing, subgraphs with private state, reducers (audit trail), loop bounds, subgraph streaming, structured output | OpenAI chat |
| [`support_swarm`](examples/support_swarm.md) | Let a live support conversation move between billing, technical and front-desk agents without making the customer repeat themselves, resuming with whoever last held it on the next message. | Command.PARENT handoffs from tools, agents as subgraphs, persistent active_agent routing from START, scoped tool sets, handoff and tool-round bounds | OpenAI chat |

## RAG & knowledge

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`adaptive_qa_router`](examples/adaptive_qa_router.md) | Send each staff question to the handbook, the database, a quick reply or a clarifying question, and refuse to guess when the question is ambiguous. | subgraphs with different state schemas, interrupt() for clarification, conditional routing, bounded loop via restricted schema, reuse of other examples, input/output schemas | OpenAI chat, OpenAI Decisions API |
| [`corrective_rag`](examples/corrective_rag.md) | Stop keyword retrieval from producing confident answers built on plausible but wrong articles, by grading what was retrieved, rewriting the query when needed, and abstaining when the knowledge base does not cover the question. | Decisions API batch grading, retry loop with query rewriting, abstain path, deterministic post-check, reducers across loop rounds, structured output | OpenAI Decisions API, OpenAI chat |

## Memory & streaming

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`conversation_memory`](examples/conversation_memory.md) | Keep a 100-turn support chat cheap and within context without forgetting the customer's name, order numbers or what the assistant promised, by combining a bounded window, a rolling summary and pinned facts. | checkpointer thread memory, add_messages + RemoveMessage, rolling summary, trim_messages window, pinned facts with a reducer, conditional edge, token streaming + custom events | OpenAI chat |
| [`live_digest_stream`](examples/live_digest_stream.md) | Stop users resubmitting a slow report by streaming typed progress, partial results and the digest's tokens, with subgraph-aware events a front-end can consume directly. | get_stream_writer custom events, multi-mode streaming (updates/messages/custom), subgraph streaming with namespaces, token streaming with node filtering, subgraph as a node | OpenAI chat |
| [`memory_concierge`](examples/memory_concierge.md) | Remember customer preferences across separate conversations, let them take facts back, and never persist card numbers, passwords or other secrets. | BaseStore injection, namespaces, semantic search, runtime context, structured extraction, cross-thread persistence, token streaming | OpenAI chat |

## Human-in-the-loop & persistence

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`ops_copilot`](examples/ops_copilot.md) | Let an on-call copilot read service health freely but run restarts, scaling and rollbacks only after a human approves, edits or rejects each one, with hard limits enforced in code even after approval. | interrupt with approve/edit/reject, ToolNode + Send, tool-call splitting, InjectedState, defence in depth, state-held environment, bounded agent loop | OpenAI chat (tool calling) |
| [`pricing_whatif`](examples/pricing_whatif.md) | Let a pricing manager change one assumption in a finished analysis and recompute only the dependent steps by forking the saved run at its assumptions checkpoint, without losing the original. | checkpoints, get_state_history, update_state forking, replay without re-running earlier nodes, branching histories | OpenAI chat |
| [`refund_approval`](examples/refund_approval.md) | Auto-approve small clear-cut refunds, send larger or risky ones to a manager or fraud analyst, and guarantee money never moves before an explicit human decision and never moves twice, even on resume or replay. | interrupt/resume with validation loop, interrupt restarts the node, deterministic policy engine, idempotent side effect, BaseStore ledger, audience-restricted retrieval, Command routing | OpenAI chat |

## Runtime & resilience

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`functional_moderation`](examples/functional_moderation.md) | Moderate new product reviews in parallel, ask a human once about the risky or borderline ones, and make the job resumable (no re-paid model calls) and forgetful-proof (no re-moderating yesterday's reviews). | @entrypoint / @task, parallel tasks, interrupt() in an entrypoint, previous + entrypoint.final, durable task results | OpenAI chat |
| [`resilient_enrichment`](examples/resilient_enrichment.md) | Enrich a batch of company records from a flaky vendor API so that transient errors are retried, repeat lookups hit a cache, one bad record never sinks the batch, and the operator sees live progress. | RetryPolicy, CachePolicy, error_handler / NodeError, TimeoutPolicy, custom streaming, Send fan-out | none (simulated vendor API) |

## Decisions API

| Example | Problem it solves | Key LangGraph features | Models |
|---|---|---|---|
| [`classifier_bakeoff`](examples/classifier_bakeoff.md) | Measure which approach (keyword rules, a structured-output LLM, or the Decisions API) should classify market headlines, on accuracy, hard cases, calibration, agreement, latency and tokens. | parallel branches + reducer, per-node input_schema, batching, Decisions API at scale, evaluation harness | OpenAI chat, OpenAI Decisions API |
| [`credit_decision_gate`](examples/credit_decision_gate.md) | Automatically approve and decline the clear loan applications, quantify uncertainty with calibrated probabilities, and send the borderline ones to a human underwriter. | Decisions API, policy thresholds + abstain band, interrupt(), Command routing, compliance guard, untrusted-input screen | OpenAI Decisions API, OpenAI chat |
| [`subject_line_tournament`](examples/subject_line_tournament.md) | Pick the best email subject line from diverse, guardrail-filtered candidates using one calibrated, inspectable batch judgement instead of a single model draft. | Send fan-out, reducer, deterministic guardrail, Decisions API batch judging, conditional edge, best-of-N | OpenAI chat, OpenAI Decisions API |

## Index by LangGraph feature

| Feature | Examples |
|---|---|
| @entrypoint / @task | [`functional_moderation`](examples/functional_moderation.md) |
| abstain path | [`corrective_rag`](examples/corrective_rag.md) |
| add_messages + removemessage | [`conversation_memory`](examples/conversation_memory.md) |
| add_messages + removemessage trimming | [`sales_react_agent`](examples/sales_react_agent.md) |
| agents as subgraphs | [`support_swarm`](examples/support_swarm.md) |
| append-only reducer | [`draft_critic_loop`](examples/draft_critic_loop.md) |
| audience-restricted retrieval | [`refund_approval`](examples/refund_approval.md) |
| basestore injection | [`memory_concierge`](examples/memory_concierge.md) |
| basestore ledger | [`refund_approval`](examples/refund_approval.md) |
| batching | [`classifier_bakeoff`](examples/classifier_bakeoff.md) |
| best-of-n | [`subject_line_tournament`](examples/subject_line_tournament.md) |
| bm25 retrieval | [`ticket_router`](examples/ticket_router.md) |
| bounded agent loop | [`ops_copilot`](examples/ops_copilot.md), [`sales_react_agent`](examples/sales_react_agent.md) |
| bounded loop via restricted schema | [`adaptive_qa_router`](examples/adaptive_qa_router.md) |
| branching histories | [`pricing_whatif`](examples/pricing_whatif.md) |
| cachepolicy | [`resilient_enrichment`](examples/resilient_enrichment.md) |
| checkpointer thread memory | [`conversation_memory`](examples/conversation_memory.md) |
| checkpoints | [`pricing_whatif`](examples/pricing_whatif.md) |
| closed tool registry | [`plan_execute_analyst`](examples/plan_execute_analyst.md) |
| command fan-out / skip | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| command routing | [`credit_decision_gate`](examples/credit_decision_gate.md), [`refund_approval`](examples/refund_approval.md), [`research_supervisor`](examples/research_supervisor.md), [`ticket_router`](examples/ticket_router.md) |
| command.parent handoffs from tools | [`support_swarm`](examples/support_swarm.md) |
| compliance guard | [`credit_decision_gate`](examples/credit_decision_gate.md) |
| conditional edge | [`conversation_memory`](examples/conversation_memory.md), [`subject_line_tournament`](examples/subject_line_tournament.md) |
| conditional edges | [`draft_critic_loop`](examples/draft_critic_loop.md) |
| conditional edges with named repair edge | [`sql_analyst`](examples/sql_analyst.md) |
| conditional routing | [`adaptive_qa_router`](examples/adaptive_qa_router.md) |
| cross-thread persistence | [`memory_concierge`](examples/memory_concierge.md) |
| custom keyed reducer | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| custom streaming | [`resilient_enrichment`](examples/resilient_enrichment.md) |
| cycle with explicit cap | [`draft_critic_loop`](examples/draft_critic_loop.md) |
| decisions api | [`credit_decision_gate`](examples/credit_decision_gate.md) |
| decisions api at scale | [`classifier_bakeoff`](examples/classifier_bakeoff.md) |
| decisions api batch grading | [`corrective_rag`](examples/corrective_rag.md) |
| decisions api batch judging | [`subject_line_tournament`](examples/subject_line_tournament.md) |
| defence in depth | [`ops_copilot`](examples/ops_copilot.md) |
| defer=true join | [`review_insights_parallel`](examples/review_insights_parallel.md) |
| deterministic + llm union | [`review_insights_parallel`](examples/review_insights_parallel.md) |
| deterministic guardrail | [`subject_line_tournament`](examples/subject_line_tournament.md) |
| deterministic policy engine | [`refund_approval`](examples/refund_approval.md) |
| deterministic post-check | [`corrective_rag`](examples/corrective_rag.md) |
| deterministic safety layer around a model | [`sql_analyst`](examples/sql_analyst.md) |
| durable task results | [`functional_moderation`](examples/functional_moderation.md) |
| error_handler / nodeerror | [`resilient_enrichment`](examples/resilient_enrichment.md) |
| evaluation harness | [`classifier_bakeoff`](examples/classifier_bakeoff.md) |
| fail-closed routing | [`ticket_router`](examples/ticket_router.md) |
| get_state_history | [`pricing_whatif`](examples/pricing_whatif.md) |
| get_stream_writer custom events | [`live_digest_stream`](examples/live_digest_stream.md) |
| guarded repair loop | [`sql_analyst`](examples/sql_analyst.md) |
| handoff and tool-round bounds | [`support_swarm`](examples/support_swarm.md) |
| hierarchical supervisor | [`research_supervisor`](examples/research_supervisor.md) |
| idempotent side effect | [`refund_approval`](examples/refund_approval.md) |
| injectedstate | [`ops_copilot`](examples/ops_copilot.md) |
| input/output schemas | [`adaptive_qa_router`](examples/adaptive_qa_router.md), [`draft_critic_loop`](examples/draft_critic_loop.md), [`portfolio_desk_lite`](examples/portfolio_desk_lite.md), [`report_orchestrator`](examples/report_orchestrator.md), [`review_insights_parallel`](examples/review_insights_parallel.md), [`sql_analyst`](examples/sql_analyst.md), [`ticket_router`](examples/ticket_router.md) |
| interrupt restarts the node | [`refund_approval`](examples/refund_approval.md) |
| interrupt with approve/edit/reject | [`ops_copilot`](examples/ops_copilot.md) |
| interrupt() | [`credit_decision_gate`](examples/credit_decision_gate.md) |
| interrupt() for clarification | [`adaptive_qa_router`](examples/adaptive_qa_router.md) |
| interrupt() in an entrypoint | [`functional_moderation`](examples/functional_moderation.md) |
| interrupt/resume with validation loop | [`refund_approval`](examples/refund_approval.md) |
| loop bounds | [`research_supervisor`](examples/research_supervisor.md) |
| loop cap in state | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| loop with replanning and hard caps | [`plan_execute_analyst`](examples/plan_execute_analyst.md) |
| multi-mode streaming (updates/messages/custom) | [`live_digest_stream`](examples/live_digest_stream.md) |
| namespaces | [`memory_concierge`](examples/memory_concierge.md) |
| node factory | [`ticket_router`](examples/ticket_router.md) |
| operator.add reducer + order restoration | [`report_orchestrator`](examples/report_orchestrator.md) |
| operator.add reducer + overwrite reset | [`review_insights_parallel`](examples/review_insights_parallel.md) |
| overwrite reset | [`report_orchestrator`](examples/report_orchestrator.md) |
| parallel branches + join | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| parallel branches + reducer | [`classifier_bakeoff`](examples/classifier_bakeoff.md) |
| parallel critics + join edge | [`draft_critic_loop`](examples/draft_critic_loop.md) |
| parallel fan-out | [`review_insights_parallel`](examples/review_insights_parallel.md) |
| parallel tasks | [`functional_moderation`](examples/functional_moderation.md) |
| per-node input_schema | [`classifier_bakeoff`](examples/classifier_bakeoff.md) |
| persistent active_agent routing from start | [`support_swarm`](examples/support_swarm.md) |
| pinned facts with a reducer | [`conversation_memory`](examples/conversation_memory.md) |
| plan-as-state | [`plan_execute_analyst`](examples/plan_execute_analyst.md) |
| policy thresholds + abstain band | [`credit_decision_gate`](examples/credit_decision_gate.md) |
| post-hoc verification node | [`report_orchestrator`](examples/report_orchestrator.md) |
| previous + entrypoint.final | [`functional_moderation`](examples/functional_moderation.md) |
| reducer | [`subject_line_tournament`](examples/subject_line_tournament.md) |
| reducer-accumulated attempt history | [`sql_analyst`](examples/sql_analyst.md) |
| reducers (audit trail) | [`research_supervisor`](examples/research_supervisor.md) |
| reducers across loop rounds | [`corrective_rag`](examples/corrective_rag.md) |
| replay without re-running earlier nodes | [`pricing_whatif`](examples/pricing_whatif.md) |
| retry loop with query rewriting | [`corrective_rag`](examples/corrective_rag.md) |
| retrypolicy | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md), [`resilient_enrichment`](examples/resilient_enrichment.md) |
| reuse of other examples | [`adaptive_qa_router`](examples/adaptive_qa_router.md) |
| rolling summary | [`conversation_memory`](examples/conversation_memory.md) |
| runtime context | [`memory_concierge`](examples/memory_concierge.md) |
| scoped tool sets | [`support_swarm`](examples/support_swarm.md) |
| selective re-dispatch loop (send from a conditional edge) | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| semantic search | [`memory_concierge`](examples/memory_concierge.md) |
| send (orchestrator-workers) | [`report_orchestrator`](examples/report_orchestrator.md) |
| send fan-out | [`resilient_enrichment`](examples/resilient_enrichment.md), [`subject_line_tournament`](examples/subject_line_tournament.md) |
| send map-reduce | [`portfolio_desk_lite`](examples/portfolio_desk_lite.md) |
| state-held environment | [`ops_copilot`](examples/ops_copilot.md) |
| structured extraction | [`memory_concierge`](examples/memory_concierge.md) |
| structured output | [`corrective_rag`](examples/corrective_rag.md), [`plan_execute_analyst`](examples/plan_execute_analyst.md), [`research_supervisor`](examples/research_supervisor.md), [`sql_analyst`](examples/sql_analyst.md), [`ticket_router`](examples/ticket_router.md) |
| subgraph as a node | [`live_digest_stream`](examples/live_digest_stream.md) |
| subgraph streaming | [`research_supervisor`](examples/research_supervisor.md) |
| subgraph streaming with namespaces | [`live_digest_stream`](examples/live_digest_stream.md) |
| subgraphs with different state schemas | [`adaptive_qa_router`](examples/adaptive_qa_router.md) |
| subgraphs with private state | [`research_supervisor`](examples/research_supervisor.md) |
| timeoutpolicy | [`resilient_enrichment`](examples/resilient_enrichment.md) |
| token streaming | [`memory_concierge`](examples/memory_concierge.md), [`sales_react_agent`](examples/sales_react_agent.md) |
| token streaming + custom events | [`conversation_memory`](examples/conversation_memory.md) |
| token streaming with node filtering | [`live_digest_stream`](examples/live_digest_stream.md) |
| tool errors as observations | [`plan_execute_analyst`](examples/plan_execute_analyst.md) |
| tool-call splitting | [`ops_copilot`](examples/ops_copilot.md) |
| toolnode + handle_tool_errors | [`sales_react_agent`](examples/sales_react_agent.md) |
| toolnode + send | [`ops_copilot`](examples/ops_copilot.md) |
| tools_condition loop | [`sales_react_agent`](examples/sales_react_agent.md) |
| trim_messages window | [`conversation_memory`](examples/conversation_memory.md) |
| untrusted-input screen | [`credit_decision_gate`](examples/credit_decision_gate.md) |
| update_state forking | [`pricing_whatif`](examples/pricing_whatif.md) |

## Examples that use the OpenAI Decisions API

* [`adaptive_qa_router`](examples/adaptive_qa_router.md): Send each staff question to the handbook, the database, a quick reply or a clarifying question, and refuse to guess when the question is ambiguous.
* [`classifier_bakeoff`](examples/classifier_bakeoff.md): Measure which approach (keyword rules, a structured-output LLM, or the Decisions API) should classify market headlines, on accuracy, hard cases, calibration, agreement, latency and tokens.
* [`corrective_rag`](examples/corrective_rag.md): Stop keyword retrieval from producing confident answers built on plausible but wrong articles, by grading what was retrieved, rewriting the query when needed, and abstaining when the knowledge base does not cover the question.
* [`credit_decision_gate`](examples/credit_decision_gate.md): Automatically approve and decline the clear loan applications, quantify uncertainty with calibrated probabilities, and send the borderline ones to a human underwriter.
* [`portfolio_desk_lite`](examples/portfolio_desk_lite.md): Turn eight fictional stocks into one risk-checked daily trade blotter: research every name in parallel, hear a bull and a bear case, let a risk committee send only the breaching names back for resizing, and plan the execution, with code guaranteeing the limits.
* [`subject_line_tournament`](examples/subject_line_tournament.md): Pick the best email subject line from diverse, guardrail-filtered candidates using one calibrated, inspectable batch judgement instead of a single model draft.

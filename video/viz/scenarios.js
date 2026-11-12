// scenarios.js: what each scene of the video runs, and what the on-screen state cards and captions say about it.
// Every scene is a REAL run of an example graph on the local LangGraph server (see live.js); nothing here is canned data.
//
//   graph      the example's graph id on the server (GET /assistants/search)
//   case       which "Ready-made case" to submit: a prefix of one label of the example's `scenario` dropdown (the same
//              enum Studio shows); the page resolves it against GET /assistants/{id}/schemas, so the label text can change
//   fan        {nodeId: {n, cols}}  nodes that Send fans out; n reserves room for the chips, cols is the chips per row
//   reducerOf  {nodeId: stateKey}   which state key the fan-out's results are merged into (shown as the reducer)
//   channels   the cards under the graph: state keys, widest first. fmt(values, LIVE) -> text or null (= not set yet)
//   answer     (kind, interruptValue) -> the Command(resume=...) value for an interrupt
//   captions   trigger: start | node:<id> | fan:<id> | fandone:<id> | cond:<src>><dst> | edge:<src>><dst> | loop:<id> |
//              interrupt | resume | end.   {{gold}}   {n} {ckpts} {thread}.   min = seconds on screen, soft = may be skipped
//   concept    callouts for the "what is a graph" scene (live.js ?concept=1), pointing at the real graph
//   tape       show the raw event tape (stream modes) instead of captions
const trunc = (s, n) => { s = String(s ?? ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
const len = (x) => (Array.isArray(x) ? x.length : null);
const keys = (x) => (x && typeof x === 'object' ? Object.keys(x).length : null);

const END = { at: 'end', text: 'Finished: {ckpts} checkpoints saved, one per step', min: 2.4 };
const START = { at: 'start', text: 'A run = a {{thread_id}} plus a stream of events from the LangGraph server', min: 2.6 };

export const SCENARIOS = {
  // ------------------------------------------------------------------------------------------------ routing
  ticket: {
    graph: 'ticket_router',
    case: 'T-2301',
    speed: 1.0,
    fan: {},
    channels: [
      { k: 'ticket_id', w: 0.9, fmt: (v) => v.ticket_id || null },
      { k: 'subject', w: 2.3, fmt: (v) => (v.subject ? trunc(v.subject, 36) : null) },
      { k: 'triage', w: 1.9, fmt: (v) => (v.triage ? `${v.triage.category} · ${v.triage.priority}` : null) },
      { k: 'route', w: 1.2, fmt: (v) => v.route || null },
      { k: 'reply', w: 2.1, fmt: (v) => (v.reply ? trunc(v.reply, 40) : null) },
    ],
    answer: () => ({}),
    captions: [
      START,
      { at: 'node:load_ticket', soft: true, text: '{{load_ticket}} is a plain function: it reads the state and returns an update', min: 2.8 },
      { at: 'node:triage', soft: true, text: '{{triage}} asks the language model to classify the ticket', min: 2.6 },
      { at: 'cond:triage>escalate', text: '{{conditional edge}}: one of six next steps, chosen at run time', min: 3.2 },
      { at: 'cond:triage>billing_specialist', text: '{{conditional edge}}: one of six next steps, chosen at run time', min: 3.2 },
      END,
    ],
    concept: {
      dwell: 2700,
      channels: ['ticket_id', 'subject', 'triage', 'route', 'reply'],
      callouts: [
        { text: 'state', sub: 'typed fields every step reads', card: 'subject', at: { x: 180, y: 540 } },
        { text: 'node', sub: 'a plain function', node: 'triage', side: 'b', at: { x: 650, y: 540 } },
        { text: 'edge', sub: 'what runs next', edge: 'load_ticket>triage', at: { x: 410, y: 540 } },
        { text: 'conditional edge', sub: 'chosen at run time', edge: 'triage>product_feedback_specialist', also: ['triage>escalate'], at: { x: 990, y: 540 }, tdx: -30, tdy: 30 },
      ],
    },
  },

  // ------------------------------------------------------------------------------------------------ Send fan-out + reducer
  orchestrator: {
    graph: 'report_orchestrator',
    case: '2026Q2',
    speed: 1.0,
    fan: { write_section: { n: 6, cols: 6 } },
    reducerOf: { write_section: 'section_drafts' },
    channels: [
      { k: 'quarter', w: 0.9, fmt: (v) => v.quarter || null },
      { k: 'facts', w: 1.0, fmt: (v) => (keys(v.facts) ? `${keys(v.facts)} facts` : null) },
      { k: 'plan', w: 1.15, fmt: (v) => (len(v.plan) ? `${len(v.plan)} sections` : null) },
      { k: 'section_drafts', w: 1.7, tag: 'reducer: operator.add', fmt: (v, L) => (L.count.write_section !== undefined ? `${L.count.write_section} / ${L.total.write_section}` : len(v.section_drafts) ? `${len(v.section_drafts)} drafts` : null) },
      { k: 'verification', w: 1.5, fmt: (v) => (v.verification ? (v.verification.ok ? 'numbers verified' : 'unsupported numbers') : null) },
      { k: 'report', w: 1.0, fmt: (v) => (v.report ? 'rendered' : null) },
    ],
    answer: () => ({}),
    captions: [
      { at: 'start', text: 'report_orchestrator: plan the sections first, then write them in parallel', min: 2.8 },
      { at: 'node:plan_sections', soft: true, text: '{{plan_sections}} decides how many sections this quarter needs', min: 2.6 },
      { at: 'fan:write_section', text: '{{Send API}}: one write_section task per section, {n} in parallel', min: 3.2 },
      { at: 'fandone:write_section', text: '{{reducer}} merges the {n} drafts into one list: section_drafts = operator.add', min: 3.4 },
      { at: 'node:verify_numbers', soft: true, text: '{{verify_numbers}} checks every figure in the text against the data', min: 2.8 },
      END,
    ],
  },

  // ------------------------------------------------------------------------------------------------ loop
  loop: {
    graph: 'draft_critic_loop',
    case: 'Lumen Therm: 16 words',
    speed: 1.0,
    fan: {},
    channels: [
      { k: 'iteration', w: 0.9, fmt: (v) => (v.iteration !== undefined && v.iteration !== null ? `draft ${v.iteration} of ${v.max_iterations || 3}` : null) },
      { k: 'draft', w: 3.0, fmt: (v) => (v.draft ? trunc(v.draft, 60) : null) },
      { k: 'violations', w: 1.3, fmt: (v) => (Array.isArray(v.violations) ? `${v.violations.length} violation${v.violations.length === 1 ? '' : 's'}` : null) },
      { k: 'passed', w: 0.8, fmt: (v) => (typeof v.passed === 'boolean' ? String(v.passed) : null) },
      { k: 'status', w: 1.2, fmt: (v) => v.status || null },
    ],
    answer: () => ({}),
    captions: [
      { at: 'start', text: 'draft_critic_loop: write, check, critique, judge, and go round again', min: 2.8 },
      { at: 'node:check_rules', soft: true, text: 'a rule checker (code) and a critic (model) review the draft {{in parallel}}', min: 3.0 },
      { at: 'cond:judge>write_draft', text: '{{conditional edge}} pointing backwards: not good enough, so loop to write_draft', min: 3.4 },
      { at: 'loop:write_draft', text: 'every pass through the loop adds {{checkpoints}}: the history of each draft is kept', min: 3.2, once: true },
      { at: 'cond:judge>finalize', text: 'the iteration cap stops the loop: {{best_effort}}, honestly labelled', min: 3.4 },
      END,
    ],
  },

  // ------------------------------------------------------------------------------------------------ interrupt + Decisions API
  gate: {
    graph: 'credit_decision_gate',
    case: 'L-3039',
    speed: 1.0,
    fan: {},
    channels: [
      { k: 'application_id', w: 1.0, fmt: (v) => v.application_id || null },
      { k: 'metrics', w: 1.15, fmt: (v) => (v.metrics && v.metrics.debt_to_income !== undefined ? `DTI ${Math.round(v.metrics.debt_to_income * 100)}%` : null) },
      { k: 'signals', w: 1.9, fmt: (v) => (v.signals && v.signals.p_repay !== undefined ? `p(repay) ${Number(v.signals.p_repay).toFixed(2)} · ${v.signals.risk_tier}` : null), label: 'signals (Decisions API)' },
      { k: 'policy', w: 1.35, fmt: (v) => (v.policy && v.policy.outcome ? v.policy.outcome : null) },
      { k: 'decision', w: 1.9, fmt: (v, L) => (v.decision && v.decision.outcome ? `${v.decision.outcome} · ${v.decision.decided_by}` : L.pending || null) },
    ],
    answer: () => ({ decision: 'approve', reason: 'Two late payments are explained and the income is stable.' }),
    captions: [
      { at: 'start', text: 'credit_decision_gate: a loan file goes through code, a model, then policy', min: 2.8 },
      { at: 'node:assess', soft: true, text: '{{assess}} calls the OpenAI {{Decisions API}}: calibrated probabilities, not prose', min: 3.0 },
      { at: 'node:policy', soft: true, text: '{{policy}} is plain code: it compares those probabilities with fixed thresholds', min: 3.0 },
      { at: 'interrupt', text: '{{interrupt()}}: state is checkpointed; nothing runs until a person answers', min: 3.6 },
      { at: 'resume', text: '{{Command(resume=…)}} wakes the same thread exactly where it stopped', min: 2.8 },
      END,
    ],
  },

  // ------------------------------------------------------------------------------------------------ streaming
  stream: {
    graph: 'live_digest_stream',
    case: 'Lumen Plug 4-pack, newest 40',
    speed: 1.0,
    fan: {},
    tape: true,
    groupSub: 'subgraph',
    channels: [
      { k: 'reviews', w: 1.0, fmt: (v) => (len(v.reviews) ? `${len(v.reviews)} reviews` : null) },
      { k: 'rating_stats', w: 1.2, fmt: (v) => (v.rating_stats ? `mean ${v.rating_stats.mean}` : null) },
      { k: 'themes', w: 1.0, fmt: (v) => (len(v.themes) ? `${len(v.themes)} themes` : null) },
      { k: 'risks', w: 1.1, fmt: (v) => (Array.isArray(v.risks) && v.themes ? `${v.risks.length} flagged` : null) },
      { k: 'digest', w: 1.5, fmt: (v) => (v.digest ? `${v.digest.split(/\s+/).length} words` : null) },
    ],
    answer: () => ({}),
    captions: [],
  },

  // ------------------------------------------------------------------------------------------------ short-term memory
  memory_short: {
    graph: 'conversation_memory',
    case: 'Pasted 5-message',
    speed: 1.0,
    fan: {},
    channels: [
      { k: 'messages', w: 1.2, fmt: (v) => (len(v.messages) !== null ? `${len(v.messages)} message${len(v.messages) === 1 ? '' : 's'}` : null), tag: 'reducer: add_messages' },
      { k: 'facts', w: 1.7, fmt: (v) => (v.facts && v.facts.order_ids && v.facts.order_ids.length ? `pinned ${[...new Set(v.facts.order_ids)].join(' ')}` : null) },
      { k: 'summary', w: 1.5, fmt: (v) => (v.summary ? `${v.summary.split(/\s+/).length}-word summary` : null) },
      { k: 'folded_messages', w: 1.3, fmt: (v) => (v.folded_messages ? `${v.folded_messages} folded` : null) },
      { k: 'turn', w: 0.7, fmt: (v) => (v.turn !== undefined && v.turn !== null ? String(v.turn) : null) },
    ],
    answer: () => ({}),
    captions: [
      { at: 'start', text: 'conversation_memory: short-term memory is {{the thread}} itself', min: 2.8 },
      { at: 'node:respond', soft: true, text: '{{respond}} answers from the whole thread: the checkpointer keeps the messages', min: 3.0 },
      { at: 'node:remember', soft: true, text: '{{remember}} pins facts that must never be lost: the customer and the order numbers', min: 3.2 },
      { at: 'node:summarize', text: '{{summarize}} folds the oldest messages into a summary once the thread gets long', min: 3.4 },
      END,
    ],
  },

  // ------------------------------------------------------------------------------------------------ long-term memory
  memory_long: {
    graph: 'memory_concierge',
    case: "Customer 'ria'",
    speed: 1.0,
    fan: {},
    channels: [
      { k: 'user_id', w: 0.9, fmt: (v) => v.user_id || null },
      { k: 'recalled', w: 3.0, fmt: (v) => (Array.isArray(v.recalled) ? (v.recalled.length ? `${v.recalled.length} notes: ${trunc(v.recalled.map((r) => r.text).join(' / '), 44)}` : 'nothing saved yet') : null), tag: 'from the Store' },
      { k: 'messages', w: 1.1, fmt: (v) => (len(v.messages) !== null ? `${len(v.messages)} message${len(v.messages) === 1 ? '' : 's'}` : null) },
      { k: 'memory_changes', w: 1.3, fmt: (v) => (Array.isArray(v.memory_changes) ? (v.memory_changes.length ? `${v.memory_changes.length} written` : 'none needed') : null) },
    ],
    answer: () => ({}),
    captions: [
      { at: 'start', text: 'memory_concierge: long-term memory lives in the {{Store}}, outside any one thread', min: 3.0 },
      { at: 'node:recall', text: '{{recall}} searches this customer\'s saved notes by meaning: {{semantic search}}', min: 3.2 },
      { at: 'node:respond', soft: true, text: 'the reply uses what was recalled: memory outlives the conversation', min: 2.8 },
      { at: 'node:remember', soft: true, text: '{{remember}} writes anything worth keeping back to the Store', min: 2.8 },
      END,
    ],
  },
};

SCENARIOS.generic = {
  graph: null, case: null, fan: {}, dynamic: true, channels: [],
  answer: () => ({}),
  captions: [START, { at: 'interrupt', text: '{{interrupt()}}: state is checkpointed under this thread_id; nothing runs until a resume', min: 3.4 }, { at: 'resume', text: '{{Command(resume=…)}} wakes the same thread exactly where it stopped', min: 2.6 }, END],
};

/** graph id -> scenario key, for `?graph=ticket_router` style URLs. */
export const BY_GRAPH = Object.fromEntries(Object.entries(SCENARIOS).filter(([, s]) => s.graph).map(([k, s]) => [s.graph, k]));

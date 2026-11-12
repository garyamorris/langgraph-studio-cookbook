# Recorded runs

Each file is one real run against a local `langgraph dev` server (see `record_run.py`): the input, thread id, every stream event with its time, node start/end, checkpoints, interrupts and the Command(resume) that answered them, and the final state. These are the same ready-made cases the video shows; each video scene is its own live run of that case, so wording of model output differs between them.

| file | graph | case | seconds | node runs | checkpoints | interrupts | mode | errors |
|---|---|---|---|---|---|---|---|---|
| 01_ticket_router.json | ticket_router | T-2301 URGENT: plug smells of burning -> ESCALATE (safety lexicon) | 5.84 | 3 | 5 | 0 | live | 0 |
| 02_report_orchestrator.json | report_orchestrator | 2026Q2 revenue $33,848.40, return rate 7.9% -> Cam Pro return spike is | 16.17 | 11 | 8 | 0 | live | 0 |
| 03_draft_critic_loop.json | draft_critic_loop | Lumen Therm: 16 words of required facts in a 10-word limit -> can neve | 35.01 | 14 | 13 | 0 | live | 0 |
| 04_credit_decision_gate.json | credit_decision_gate | L-3039 'two late payments in 2024', score 704, DTI 35% -> watch for a  | 2.95 | 7 | 9 | 1 | live | 0 |
| 05_live_digest_stream.json | live_digest_stream | Lumen Plug 4-pack, newest 40, 0.5 s per stage -> watch the stages tick | 14.53 | 7 | 6 | 0 | live | 0 |
| 06_conversation_memory.json | conversation_memory | Pasted 5-message chat, max_messages=4 keep_last=2 -> folds 4 into a su | 10.11 | 4 | 6 | 0 | live | 0 |
| 07_memory_concierge.json | memory_concierge | Customer 'ria' has 2 saved notes, asks 'Which camera should I buy?' -> | 5.35 | 4 | 6 | 0 | live | 0 |

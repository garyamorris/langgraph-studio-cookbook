# OpenAI Decisions API

`client.decisions.create(model="gpt-6-luna", input=..., questions=[...])` answers many classification or scoring
questions about one input in a single call. Each question is a `predicate` (returns a probability that a
statement is true), a `choice` (the chosen option, a confidence and a probability per option) or a `score` (a
rating against ordered levels, with a probability per level). A request may contain 1 to 400 questions and answers
come back in order, each possibly a `refusal`. Because outputs are calibrated probabilities rather than prose, they
are well suited to routing, gating and thresholds, for example auto-approve above 0.9, escalate between 0.5 and
0.9, decline below. It classifies; it does not write text, so pair it with a chat model for generation.

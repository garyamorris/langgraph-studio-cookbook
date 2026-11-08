# Multi-agent patterns

Common architectures: a *supervisor* node decides which worker runs next, usually through structured output and
`Command(goto=...)`; a *swarm* lets peer agents hand off to one another with tools that return `Command(goto=
other_agent, graph=Command.PARENT)`; *hierarchical teams* nest supervisors, each team being a subgraph;
*debate* runs opposing agents in parallel and has a judge decide; *reflection* alternates a generator and a critic.
Prefer the simplest structure that works: a router plus specialists is cheaper and easier to debug than free-form
agent chatter. Always bound loops with an iteration counter, and give each agent only the tools it needs.

# Checkpointers and threads

A checkpointer saves a snapshot of graph state after every super-step. Snapshots are grouped into a *thread*,
identified by `config={"configurable": {"thread_id": "..."}}`. Threads give you conversation memory, fault
tolerance, human-in-the-loop pauses and time travel. InMemorySaver is fine for development and tests; use a
database-backed saver in production. When you run under `langgraph dev` or LangGraph Studio the platform
supplies persistence for you, so do not compile exported graphs with their own checkpointer. In code, expose a
`build_graph(**kwargs)` helper that accepts a checkpointer for tests and scripts.

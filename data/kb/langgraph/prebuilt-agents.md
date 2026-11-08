# Prebuilt agent building blocks

The classic tool-calling agent is a loop: an LLM node bound to tools, a `ToolNode` that executes requested tool calls,
and `tools_condition` to route back to the model or to END. You can assemble it by hand, which is instructive, or use
`langgraph.prebuilt.create_react_agent(model, tools, prompt=..., checkpointer=..., store=...)`. Tools can read graph
state or the store through injected arguments (`InjectedState`, `InjectedStore`) without exposing them to the model.
Keep tool descriptions precise, return compact results, and trim long message histories before each model call.

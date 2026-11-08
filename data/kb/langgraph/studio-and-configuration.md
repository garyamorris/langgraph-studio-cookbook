# LangGraph Studio and langgraph.json

`langgraph.json` maps graph ids to Python objects, for example `"ticket_router": "./src/pkg/mod.py:graph"`, lists
`dependencies` (usually `["."]`) and an optional `env` file. Run `uv run langgraph dev` to start a local server
(default port 2024) and open Studio against it. Studio lets you pick a graph, submit input, watch nodes execute,
inspect and edit state at any step, fork from a past checkpoint, and answer interrupts. A graph can declare a
`context_schema` (or read `config["configurable"]`) so runs can be parameterised from the Studio configuration
panel, for instance to choose a model or a threshold per run.

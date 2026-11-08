# Nodes, edges and routing

A StateGraph is built from nodes (plain functions) and edges. `add_edge(a, b)` is a fixed transition;
`add_conditional_edges(a, router)` calls a function after `a` and routes to the node name it returns.
A node can also route itself by returning `Command(goto="next_node", update={...})`, which fuses the state
update and the routing decision in one place; declare `destinations=` on `add_node` (or type the return as a
Literal) so Studio can draw the possible edges. START and END are the special entry and exit.
Cycles are allowed and are how loops and agents are built. The default recursion limit in langgraph 1.2 is very
high (10007, overridable with the LANGGRAPH_DEFAULT_RECURSION_LIMIT environment variable), so always put an
explicit iteration cap in state or pass `recursion_limit` in the run config for any loop.

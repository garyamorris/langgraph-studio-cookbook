# Subgraphs

A compiled graph can be used as a node in another graph. If parent and subgraph share state keys, add the
compiled subgraph directly with `add_node("team", team_graph)`. If their schemas differ, wrap the call in a
function that maps parent state to subgraph input and the subgraph output back to a parent update. Subgraphs
let teams own a self-contained workflow, be tested alone, and be reused. Pass `subgraphs=True` when streaming
to see events from inside them. The checkpointer of the parent is propagated, an interrupt raised inside a
subgraph pauses the whole run, and policies set with `set_node_defaults` are not inherited by subgraphs.

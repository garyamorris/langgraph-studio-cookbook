# Send and map-reduce fan-out

When the number of parallel tasks is only known at runtime, return a list of `Send(node_name, payload)` objects
from a conditional edge. Each Send starts one instance of the target node with its *own* input payload, which
does not have to match the graph state. All instances run in the same super-step, and their outputs are merged
through the state reducers, so the collecting key needs a reducer such as `operator.add`. This is the map-reduce
pattern: orchestrator decides the work, workers run in parallel, a downstream node reduces the results.
Use `defer=True` on the reduce node if branches can finish in different super-steps and you need to wait for all.

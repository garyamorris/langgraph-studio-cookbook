# Streaming modes

`graph.stream(input, stream_mode=...)` supports several modes: `values` (full state after each step), `updates`
(only what each node returned), `messages` (LLM tokens as they are generated), `custom` (anything you emit with
`get_stream_writer()` inside a node) and `debug`. You can pass a list of modes to receive several at once. Custom
events are a lightweight way to report progress from long-running nodes, like "processed 40 of 200 rows".
With `subgraphs=True`, events from nested graphs are included and tagged with their namespace. Studio uses these
streams to animate execution live.

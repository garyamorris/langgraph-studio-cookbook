# Retries, caching, timeouts and error handlers

Node-level policies are passed to `add_node`. `retry_policy=RetryPolicy(max_attempts, initial_interval,
backoff_factor, retry_on)` retries a failing node with backoff. `cache_policy=CachePolicy(ttl=..., key_func=...)`
skips re-running a node for identical input when you compile with `cache=InMemoryCache()` (or another cache).
`timeout=TimeoutPolicy(run_timeout=..., idle_timeout=...)` bounds an attempt; timeouts rely on asyncio
cancellation, so they cannot interrupt blocking synchronous code. `error_handler=fn` runs after retries are
exhausted: declare `def fn(state, error: NodeError) -> Command` and return a `Command` with an update and a `goto`
to degrade gracefully. `StateGraph.set_node_defaults(...)` applies policies to every node at once.

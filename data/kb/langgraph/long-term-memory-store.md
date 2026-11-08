# Long-term memory with the Store

Checkpoints are scoped to one thread. For memory that spans threads, such as user preferences, use a Store (a
BaseStore implementation such as InMemoryStore). Items live under a namespace tuple like `("memories", user_id)`
with a key and a JSON value. Use `store.put`, `store.get` and `store.search`; semantic search additionally needs an
index configured with an embeddings model. Nodes receive the store by declaring a `store: BaseStore` parameter,
or through the runtime object. LangGraph Studio and `langgraph dev` provide a store automatically, so the same
graph code works locally and deployed.

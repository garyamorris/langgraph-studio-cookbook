# State and reducers

A graph's state schema (a TypedDict, dataclass or Pydantic model) defines its channels: one per key.
Nodes read the current state and return a *partial update*; LangGraph merges that update into state.
By default a key is overwritten by the latest write. To combine writes, annotate the key with a reducer,
for example `Annotated[list, operator.add]` to append, or `add_messages` to merge chat messages by id.
Reducers matter most with parallel branches: if two nodes write the same key in one step and the key has
no reducer, LangGraph raises an InvalidUpdateError. You can also give a graph separate `input_schema` and
`output_schema` so callers only see the keys you intend, while internal keys stay private.

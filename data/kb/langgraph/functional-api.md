# The Functional API

Besides StateGraph, LangGraph offers a functional style. Decorate a function with `@entrypoint()` to make a
workflow and use `@task` for units of work. Calling a task returns a future, so you get parallelism by starting
several tasks and then collecting `.result()`. Task results are checkpointed: if the workflow is interrupted and
resumed, finished tasks are not re-executed. `interrupt()` works inside entrypoints, and an entrypoint can accept
a `previous` argument to read the value saved by the prior run on the same thread; return `entrypoint.final(value=...,
save=...)` to return one thing and persist another. It is the most direct way to turn existing Python control flow
into a durable workflow.

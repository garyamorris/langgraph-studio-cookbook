# Human-in-the-loop with interrupt()

Call `interrupt(payload)` inside a node to pause the run and surface `payload` to the caller (it appears under
`__interrupt__` in results, and as a prompt in Studio). Resume by invoking the same thread with
`Command(resume=value)`; `interrupt()` then returns `value`. Important: on resume the node restarts from its first
line, so put any side effects (sending email, charging cards) *after* the interrupt or make them idempotent.
When a node calls interrupt more than once, resume values are matched by order. Static breakpoints are also
available via `interrupt_before` / `interrupt_after` at compile time; they are handy for debugging in Studio,
where you can edit state before continuing. Interrupts require a checkpointer.

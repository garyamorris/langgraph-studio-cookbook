# Time travel: replay and fork

Every checkpoint of a thread is addressable. `graph.get_state_history(config)` lists them newest first. To
*replay*, invoke with a config that includes an earlier `checkpoint_id`: nodes before that checkpoint are not
re-run, nodes after it are executed again (so LLM calls are repeated). To *fork*, call `graph.update_state(config,
values)` on a past checkpoint, which creates a new checkpoint on a new branch of history, then continue from it.
In Studio this is the "fork" action on a step. Time travel is ideal for what-if analysis and for debugging why an
agent took a wrong turn without re-running the entire expensive prefix.

"""Ready-made Studio inputs ("scenarios"), so no example needs data you have not seen.

PROBLEM
    Studio turns a graph's input schema into a form. Most examples are keyed by something that lives in a dataset: a
    ticket id, a loan id, a ticker, a product name, a quarter, a question about sales you have not looked at. A person
    who opens the graph cannot know ``T-2301`` exists, so the blank form is a dead end.

DESIGN
    An example that needs such input declares ``CASES``: an ordered ``{label: input}`` dict. Each ``input`` is a
    real input payload for that example (the same keys its form has). The label is what the person reads in Studio's
    dropdown, so it says what the case IS and what to watch for ("T-2301 urgent: plug smells of burning -> escalate").

    * ``scenario_field(CASES)`` is the type of an extra ``scenario`` input field: a dropdown of the labels.
    * ``expand(state, CASES, has_input=...)`` is called by the example's entry node. It returns the state update
      that turns a picked scenario into the real input fields, and clears ``scenario`` so a later run on the same
      thread (Studio reuses one) cannot mistake it for new input.

    Behaviour, in order: a picked scenario wins over the other fields; with no scenario but real input in the other
    fields, nothing changes (typing your own ``ticket_id`` still works); with nothing at all, the FIRST scenario runs,
    so pressing Submit on a blank form always shows something meaningful. Put the best first-time case first.

    Every scenario of one example must set the SAME keys (``tests/test_scenarios.py`` checks it). That is what makes
    "pick a scenario" fully replace the previous run's input: a key one scenario sets and another omits would leak
    from the earlier run.

    Caveat, inherited from LangGraph rather than added here: Studio reuses the thread, so submitting a blank form
    AFTER a run repeats that run's input (the first scenario only runs on a fresh thread). Start a New Thread, or
    pick a scenario, for a clean run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal

from pydantic import Field

SCENARIO_HELP = (
    "Pick a ready-made case and press Submit; it replaces the fields below. Leave everything blank on a new thread "
    "to run the first one."
)


def scenario_field(scenarios: Mapping[str, Mapping[str, Any]], *, help: str = SCENARIO_HELP) -> Any:
    """The annotation for the ``scenario`` input field: a Studio dropdown of the scenario labels.

    Use it at module level (``SCENARIO = scenario_field(CASES)``) and then ``scenario: SCENARIO`` in the input
    TypedDict. It is a type for the schema only: nothing validates a value at run time except :func:`expand`.
    """
    labels = tuple(scenarios)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("scenario labels must be a non-empty, duplicate-free sequence")
    return Annotated[Literal[labels], Field(title="Ready-made case", description=help)]  # type: ignore[valid-type]


def expand(
    state: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    has_input: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """The entry-node state update that resolves ``state['scenario']`` (see the module docstring for the rules).

    ``has_input(state)`` says whether the caller typed real input into the other fields (e.g. a ticket id, or a
    subject AND a body). It is only consulted when no scenario was picked. The returned dict always includes
    ``scenario: None`` when a scenario was consumed, and is empty when nothing needed expanding.
    """
    picked = state.get("scenario")
    if picked:
        if picked not in scenarios:
            raise ValueError(f"Unknown ready-made case {picked!r}; pick one from the dropdown.")
        return {**scenarios[picked], "scenario": None}
    if has_input(state):
        return {"scenario": None} if "scenario" in state else {}
    return {**next(iter(scenarios.values())), "scenario": None}

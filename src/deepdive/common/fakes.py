"""Deterministic stand-in chat model for offline mode and tests.

``FakeChatModel`` quacks like ``ChatOpenAI`` for the three ways the examples use a model:

* plain text:            ``llm.invoke(messages)``            -> an "[offline] ..." AIMessage
* structured output:     ``llm.with_structured_output(Schema)`` -> a *valid* ``Schema`` instance
* tool calling:          ``llm.bind_tools(tools)``           -> first call requests one tool with valid
                                                              arguments, the next call (after the tool
                                                              result) answers in text

Structured values are synthesised from the schema and are deterministic functions of the prompt text,
so different inputs take different (but repeatable) branches. Booleans default to values that let
loops terminate. Tests that need exact behaviour register handlers:

    set_structured_handler(Verdict, scripted([Verdict(approved=False), Verdict(approved=True)]))
    set_chat_handler(lambda messages, tools: AIMessage("custom"))
    clear_handlers()   # (the pytest fixture in tests/conftest.py does this for you)
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import types
import typing
from typing import Any, Callable, Iterator, Literal, Sequence, get_args, get_origin

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- handler registry
# Every Pydantic schema ever passed to with_structured_output(); tests lint these for OpenAI strict-mode
# compatibility (see testing.lint_openai_strict) because offline runs cannot catch that class of bug.
SEEN_SCHEMAS: set[type[BaseModel]] = set()

_STRUCTURED: dict[Any, Callable[[str], Any]] = {}
_CHAT_HANDLER: list[Callable[[list[BaseMessage], list[dict]], AIMessage] | None] = [None]


def set_structured_handler(schema: Any, handler: Callable[[str], Any]) -> None:
    """Make ``with_structured_output(schema)`` call ``handler(prompt_text)`` instead of synthesising."""
    _STRUCTURED[schema] = handler


def set_chat_handler(handler: Callable[[list[BaseMessage], list[dict]], AIMessage] | None) -> None:
    """Replace the plain chat / tool-calling behaviour of every FakeChatModel."""
    _CHAT_HANDLER[0] = handler


def clear_handlers() -> None:
    _STRUCTURED.clear()
    _CHAT_HANDLER[0] = None


def scripted(values: Sequence[Any]) -> Callable[[str], Any]:
    """Handler returning ``values`` in order, repeating the last one forever."""
    items = list(values)
    state = {"i": 0}

    def _next(_text: str) -> Any:
        v = items[min(state["i"], len(items) - 1)]
        state["i"] += 1
        return v

    return _next


# --------------------------------------------------------------------------- helpers
def stable_int(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16)


def _text_of(x: Any) -> str:
    """Flatten whatever a runnable receives (str, PromptValue, messages) into text."""
    if isinstance(x, str):
        return x
    if hasattr(x, "to_messages"):
        x = x.to_messages()
    if isinstance(x, (list, tuple)):
        out = []
        for m in x:
            if isinstance(m, BaseMessage):
                out.append(m.text if hasattr(m, "text") and isinstance(m.text, str) else str(m.content))
            elif isinstance(m, (tuple, list)) and len(m) == 2:
                out.append(str(m[1]))
            else:
                out.append(str(m))
        return "\n".join(out)
    return str(x)


_FALSE_HINT = re.compile(r"(need|revis|retry|reject|fail|error|risk|unsafe|escalat|missing|hallucinat|unsupported|ambig)", re.I)


def _bounds(field_info: Any) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in getattr(field_info, "metadata", []) or []:
        lo = getattr(m, "ge", getattr(m, "gt", lo)) if (hasattr(m, "ge") or hasattr(m, "gt")) else lo
        hi = getattr(m, "le", getattr(m, "lt", hi)) if (hasattr(m, "le") or hasattr(m, "lt")) else hi
    return lo, hi


def synth(tp: Any, name: str, seed: str, field_info: Any = None) -> Any:
    """Build a plausible value for annotation ``tp`` (deterministic in ``seed``)."""
    origin = get_origin(tp)
    if tp is Any or tp is None:
        return f"{name} (offline stub)"
    if origin is typing.Annotated:
        return synth(get_args(tp)[0], name, seed, field_info)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return synth(args[0], name, seed, field_info) if args else None
    if origin is Literal:
        opts = get_args(tp)
        return opts[stable_int(seed, name) % len(opts)]
    if origin in (list, set, frozenset, tuple, typing.Sequence) or tp in (list, set):
        args = get_args(tp)
        inner = args[0] if args else str
        n = 2 if inner is str else 1
        return [synth(inner, f"{name}_{i}", seed) for i in range(n)]
    if origin is dict or tp is dict:
        return {}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return build_model(tp, seed)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        members = list(tp)
        return members[stable_int(seed, name) % len(members)]
    if typing.is_typeddict(tp):
        hints = typing.get_type_hints(tp)
        return {k: synth(v, k, seed) for k, v in hints.items()}
    if tp is bool:
        return not bool(_FALSE_HINT.search(name))
    if tp is int:
        lo, hi = _bounds(field_info)
        v = 3
        if lo is not None:
            v = max(v, int(lo))
        if hi is not None:
            v = min(v, int(hi))
        return v
    if tp is float:
        lo, hi = _bounds(field_info)
        if lo is not None and hi is not None:
            return round((lo + hi) / 2, 3)
        return 0.8 if re.search(r"(conf|prob|score|rating)", name, re.I) else 0.5
    if tp is str:
        return f"{name.replace('_', ' ')} (offline stub)"
    return f"{name} (offline stub)"


def build_model(schema: type[BaseModel], seed: str) -> BaseModel:
    values = {}
    for fname, finfo in schema.model_fields.items():
        values[fname] = synth(finfo.annotation, fname, seed, finfo)
    return schema.model_validate(values)


def _json_synth(schema: dict, root: dict, name: str, seed: str) -> Any:
    """Synthesise a value for a JSON-schema fragment (used for tool-call arguments)."""
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return _json_synth(root.get("$defs", {}).get(ref, {}), root, name, seed)
    if "enum" in schema:
        opts = schema["enum"]
        return opts[stable_int(seed, name) % len(opts)]
    for key in ("anyOf", "oneOf"):
        if key in schema:
            opts = [o for o in schema[key] if o.get("type") != "null"]
            return _json_synth(opts[0], root, name, seed) if opts else None
    if "default" in schema:
        return schema["default"]
    t = schema.get("type")
    if t == "string":
        return f"{name}"
    if t == "integer":
        return int(schema.get("minimum", 3)) if schema.get("minimum", 3) > 3 else 3
    if t == "number":
        return 0.5
    if t == "boolean":
        return True
    if t == "array":
        return [_json_synth(schema.get("items", {}), root, f"{name}_0", seed)]
    if t == "object" or "properties" in schema:
        req = schema.get("required", list(schema.get("properties", {})))
        return {k: _json_synth(v, root, k, seed) for k, v in schema.get("properties", {}).items() if k in req}
    return f"{name}"


# --------------------------------------------------------------------------- the model
class FakeChatModel(BaseChatModel):
    """Offline stand-in for ChatOpenAI. See the module docstring."""

    tools_schema: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deepdive-offline"

    # -- tool calling -------------------------------------------------------
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":  # type: ignore[override]
        return self.model_copy(update={"tools_schema": [convert_to_openai_tool(t) for t in tools]})

    def _pick_tool(self, text: str) -> dict:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        best, best_score = self.tools_schema[0], -1
        for t in self.tools_schema:
            fn = t["function"]
            blob = set(re.findall(r"[a-z0-9]+", (fn["name"].replace("_", " ") + " " + fn.get("description", "")).lower()))
            score = len(words & blob)
            if score > best_score:
                best, best_score = t, score
        return best

    # -- core ---------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        handler = _CHAT_HANDLER[0]
        if handler is not None:
            return ChatResult(generations=[ChatGeneration(message=handler(messages, self.tools_schema))])

        # Has a tool already answered since the last human turn?
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        tool_msgs = [m for m in messages[last_human + 1 :] if isinstance(m, ToolMessage)]
        human_text = _text_of([messages[last_human]]) if last_human >= 0 else _text_of(messages)

        if self.tools_schema and not tool_msgs:
            tool = self._pick_tool(human_text)["function"]
            args = _json_synth(tool.get("parameters", {}), tool.get("parameters", {}), tool["name"], human_text)
            msg = AIMessage(
                content="",
                tool_calls=[{"name": tool["name"], "args": args or {}, "id": f"call_{stable_int(human_text, tool['name']) % 10**8}", "type": "tool_call"}],
            )
        elif tool_msgs:
            digest = " | ".join(str(m.content)[:200] for m in tool_msgs)
            msg = AIMessage(content=f"[offline] Based on the tool results: {digest}")
        else:
            digest = re.sub(r"\s+", " ", human_text).strip()[:160]
            msg = AIMessage(content=f"[offline] Response to: {digest}")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # -- token streaming ----------------------------------------------------
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the reply word by word, so ``stream_mode="messages"`` shows realistic token chunks offline."""
        msg = self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        if msg.tool_calls or not isinstance(msg.content, str) or not msg.content:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=msg.content,
                    tool_call_chunks=[
                        {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                )
            )
            yield chunk
            return
        for word in re.findall(r"\S+\s*", msg.content):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk

    # -- structured output --------------------------------------------------
    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any) -> Runnable:  # type: ignore[override]
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            SEEN_SCHEMAS.add(schema)

        def run(inp: Any) -> Any:
            text = _text_of(inp)
            handler = _STRUCTURED.get(schema)
            if handler is not None:
                parsed = handler(text)
                if isinstance(schema, type) and issubclass(schema, BaseModel) and isinstance(parsed, dict):
                    parsed = schema.model_validate(parsed)
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                parsed = build_model(schema, text)
            elif typing.is_typeddict(schema):
                parsed = synth(schema, "root", text)
            elif isinstance(schema, dict):
                parsed = _json_synth(schema.get("parameters", schema), schema, "root", text)
            else:
                parsed = synth(schema, "root", text)
            if include_raw:
                return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(run)

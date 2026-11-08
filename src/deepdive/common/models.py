"""One place to get a chat model.

Examples call ``get_llm()`` *inside nodes* (never at import time), so every graph can be loaded by
LangGraph Studio without an API key. With ``DEEPDIVE_OFFLINE=1`` (or no ``OPENAI_API_KEY``) the same
call returns a deterministic ``FakeChatModel`` so the whole playground runs for free.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from deepdive.common.config import fast_model, is_offline, openai_model
from deepdive.common.fakes import FakeChatModel

Tier = Literal["default", "fast"]


@lru_cache(maxsize=16)
def _openai(name: str, temperature: float | None, extra: tuple[tuple[str, Any], ...]) -> BaseChatModel:
    from langchain_openai import ChatOpenAI  # imported lazily so offline mode never needs the key

    params: dict[str, Any] = {"model": name, "max_retries": 2, "timeout": 90, **dict(extra)}
    # "luna" models reject the temperature parameter, so only pass it for other models.
    if temperature is not None and "luna" not in name.lower():
        params["temperature"] = temperature
    return ChatOpenAI(**params)


def get_llm(tier: Tier = "default", *, temperature: float | None = None, model: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Chat model for a node.

    ``tier="fast"`` uses ``OPENAI_MODEL_FAST`` (cheap, high-volume nodes); ``model=`` overrides both.
    ``temperature`` is applied only to models that accept it.
    """
    if is_offline():
        return FakeChatModel()
    name = model or (fast_model() if tier == "fast" else openai_model())
    return _openai(name, temperature, tuple(sorted(kwargs.items())))


def structured_llm(schema: type[BaseModel], tier: Tier = "default", **kwargs: Any):
    """Chat model that returns a validated ``schema`` instance.

    Live, this asks OpenAI for *strict* ``json_schema`` output (every field present, enums honoured) and falls back to
    ordinary function-calling structured output if the API rejects the schema, so a quirky schema degrades instead of
    failing. Offline, the stand-in model synthesises a valid instance. Verified live against gpt-6-luna.
    """
    llm = get_llm(tier, **kwargs)
    if is_offline():
        return llm.with_structured_output(schema)
    import openai

    strict = llm.with_structured_output(schema, method="json_schema", strict=True)
    loose = llm.with_structured_output(schema)
    return strict.with_fallbacks([loose], exceptions_to_handle=(openai.BadRequestError,))

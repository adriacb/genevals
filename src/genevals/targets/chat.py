"""A Target that keeps conversation history across turns of the same conversation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable

from genevals.core.types import Output, Sample
from genevals.targets.base import Target

ChatMessage = dict[str, str]
ChatGenerate = Callable[[list[ChatMessage]], "Output | str | Awaitable[Output | str]"]


class ChatTarget(Target):
    """Use for evaluating a chatbot across multi-turn conversations.

    Give multi-turn Samples the same `conversation_id` (and increasing `turn`);
    the backend runs a conversation's turns in order so history accumulates
    correctly here, while different conversations still run concurrently.
    """

    def __init__(self, name: str, fn: ChatGenerate, *, tags: dict[str, str] | None = None):
        self.name = name
        self.tags = tags or {}
        self._fn = fn
        self._histories: dict[str, list[ChatMessage]] = {}

    async def agenerate(self, sample: Sample) -> Output:
        conversation_id = sample.conversation_id or sample.id
        history = self._histories.setdefault(conversation_id, [])
        history.append({"role": "user", "content": sample.input})

        start = time.perf_counter()
        result = self._fn(history)
        if inspect.isawaitable(result):
            result = await result
        latency_ms = (time.perf_counter() - start) * 1000

        text = result.text if isinstance(result, Output) else str(result)
        history.append({"role": "assistant", "content": text})

        if isinstance(result, Output):
            if result.latency_ms is None:
                result.latency_ms = latency_ms
            return result
        return Output(text=text, latency_ms=latency_ms)

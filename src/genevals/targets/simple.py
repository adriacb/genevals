"""Wrap any callable (sync or async, one-shot call or a whole pipeline) as a Target."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable

from genevals.core.types import Output, Sample
from genevals.targets.base import Target

Generate = Callable[[Sample], "Output | str | Awaitable[Output | str]"]


class SimpleTarget(Target):
    """A single generative flow: one call in, one Output (or plain str) out.

    Use this for a plain model call, a RAG pipeline, or any agent — whatever
    `fn` does internally is opaque to genevals. To compare pipelines in
    parallel, just pass several SimpleTargets to the same Evaluator run; each
    result is tagged with its target's `name`/`tags` so reports can tell them
    apart.
    """

    def __init__(self, name: str, fn: Generate, *, tags: dict[str, str] | None = None):
        self.name = name
        self.tags = tags or {}
        self._fn = fn

    async def agenerate(self, sample: Sample) -> Output:
        start = time.perf_counter()
        result = self._fn(sample)
        if inspect.isawaitable(result):
            result = await result
        latency_ms = (time.perf_counter() - start) * 1000

        if isinstance(result, Output):
            if result.latency_ms is None:
                result.latency_ms = latency_ms
            return result
        return Output(text=str(result), latency_ms=latency_ms)

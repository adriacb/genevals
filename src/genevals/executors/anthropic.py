"""Executor backed by the Anthropic Messages API.

Requires the 'anthropic' extra: pip install 'genevals[anthropic]'

Uses AsyncAnthropic (not the sync client) so a concurrent Evaluator run
actually runs concurrently instead of serializing behind a blocking HTTP
call — see NativeBackend, which awaits every target/judge call under a
bounded asyncio.Semaphore.

Rate limits and retries:
  - The Anthropic SDK already retries connection errors, 408/409/429, and
    5xx with exponential backoff by default (`max_retries`, default 2) —
    genevals doesn't reimplement that. Pass `max_retries=`/`timeout=` (or
    any other `AsyncAnthropic` kwarg) through this class when it builds its
    own client, or configure them on a `client` you pass in yourself.
  - What the SDK's per-request retry does NOT do is cap how many requests
    genevals fires at once. `Evaluator(concurrency=N)` only bounds
    (target, sample) units in flight — one unit can still fan out into
    several judge/ensemble calls, all against the same API key. Set
    `max_concurrency=` here to cap concurrent requests *this Executor*
    makes, regardless of how many genevals-level units are trying to call
    it at once — pass the same Executor instance to both a Target and a
    Judge to share one limit across both.

Extra per-run context (a fixed system prompt / "business context", a
persona, house rules) doesn't need a new parameter — it's already just
`**create_kwargs`, forwarded straight to `messages.create`:

    AnthropicExecutor(client, system="You are Acme Corp's support agent...")

For context that varies *per sample* rather than being fixed for the whole
Executor, see the custom-Executor pattern in examples/custom_executor_demo.py.

Cost: `generate()` reads `message.usage` and prices it against the table
below — a cached snapshot of Anthropic's published per-token pricing, not a
live feed. Treat `cost_usd` as good for comparing targets/runs against each
other, not as an invoice; pass `pricing=` to override or extend the table
(e.g. for a model released after this was last updated).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from genevals.core.types import Output
from genevals.executors._pricing import cost_from_usage
from genevals.executors.base import Executor

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

# (input $/1M tokens, output $/1M tokens) — Anthropic first-party API rates.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5-1": (10.00, 50.00),
    "claude-mythos-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
}


class AnthropicExecutor(Executor):
    """Pass a shared `client` to reuse one connection across several Executors
    (e.g. one for the target under test, one for the judge, different
    models/max_tokens) — or omit it and one is built from `api_key`.
    """

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        *,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        max_concurrency: int | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        **create_kwargs: Any,
    ):
        if client is None:
            try:
                from anthropic import AsyncAnthropic as _AsyncAnthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicExecutor requires the 'anthropic' extra: pip install 'genevals[anthropic]'"
                ) from exc
            client = _AsyncAnthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._create_kwargs = create_kwargs
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self._pricing = {**DEFAULT_PRICING, **(pricing or {})}

    async def complete(self, prompt: str) -> str:
        output = await self.generate(prompt)
        return output.text

    async def generate(self, prompt: str) -> Output:
        if self._semaphore is None:
            return await self._generate(prompt)
        async with self._semaphore:
            return await self._generate(prompt)

    async def _generate(self, prompt: str) -> Output:
        start = time.perf_counter()
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._create_kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # content[0] isn't reliably the text block — extended-thinking models put
        # a ThinkingBlock first, and tool-use responses can add other block types.
        text_blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            block_types = [getattr(b, "type", type(b).__name__) for b in message.content]
            raise ValueError(f"Anthropic response had no text block (got: {block_types})")
        text = "".join(text_blocks)

        usage = getattr(message, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        cost_usd = cost_from_usage(self._model, input_tokens, output_tokens, self._pricing)

        return Output(
            text=text,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            raw={"input_tokens": input_tokens, "output_tokens": output_tokens, "model": self._model},
        )

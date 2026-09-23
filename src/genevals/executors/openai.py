"""Executor backed by the OpenAI-compatible Chat Completions API.

Requires the 'openai' extra: pip install 'genevals[openai]'

Written against the `openai` Python SDK's `AsyncOpenAI` client and
`chat.completions.create` — the one endpoint essentially every
OpenAI-compatible server implements (OpenAI itself, Azure OpenAI, vLLM's
OpenAI-compatible server, Ollama, together.ai, ...). That's deliberate: this
one class is how you evaluate a model served via vLLM, not just OpenAI's own
models — point `base_url` at your server:

    OpenAIExecutor(base_url="http://localhost:8000/v1", api_key="unused", model="my-local-model")

If your installed `openai` SDK version has moved the client/method
signatures shown here, that's the first thing to check — this hasn't been
verified against a live OpenAI key the way AnthropicExecutor has (see
`_generate`, where the actual call happens).

Rate limits, retries, concurrency, and cost: same shape as AnthropicExecutor
— the SDK retries transient errors by default (`max_retries`), `timeout=`
and other client kwargs pass through, `max_concurrency=` caps this
Executor's own concurrent requests, and `generate()` prices `response.usage`
against `pricing=` (default: a cached snapshot of OpenAI's published rates —
not a live feed, and meaningless for a self-hosted/vLLM `base_url`, where
there's no per-token bill to estimate).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from genevals.core.types import Output
from genevals.executors._pricing import cost_from_usage
from genevals.executors.base import Executor

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# (input $/1M tokens, output $/1M tokens) — OpenAI first-party API rates.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class OpenAIExecutor(Executor):
    """Pass a shared `client` to reuse one connection across several Executors,
    or omit it and one is built from `api_key`/`base_url`.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5-mini",
        max_tokens: int = 1024,
        max_concurrency: int | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        **create_kwargs: Any,
    ):
        if client is None:
            try:
                from openai import AsyncOpenAI as _AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAIExecutor requires the 'openai' extra: pip install 'genevals[openai]'"
                ) from exc
            client = _AsyncOpenAI(api_key=api_key, base_url=base_url)
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
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._create_kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""
        if not text and getattr(choice, "finish_reason", None) == "length":
            raise ValueError("OpenAI response was truncated at max_tokens with no text produced")

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost_usd = cost_from_usage(self._model, input_tokens, output_tokens, self._pricing)

        return Output(
            text=text,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            raw={"input_tokens": input_tokens, "output_tokens": output_tokens, "model": self._model},
        )

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
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from genevals.executors.base import Executor

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


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

    async def complete(self, prompt: str) -> str:
        if self._semaphore is None:
            return await self._complete(prompt)
        async with self._semaphore:
            return await self._complete(prompt)

    async def _complete(self, prompt: str) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._create_kwargs,
        )
        # content[0] isn't reliably the text block — extended-thinking models put
        # a ThinkingBlock first, and tool-use responses can add other block types.
        text_blocks = [block.text for block in message.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            block_types = [getattr(b, "type", type(b).__name__) for b in message.content]
            raise ValueError(f"Anthropic response had no text block (got: {block_types})")
        return "".join(text_blocks)

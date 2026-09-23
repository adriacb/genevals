"""Wraps another Executor, caching identical prompts.

Re-running an eval during iterative development (tweaking one metric, one
target) otherwise re-pays for every unchanged sample. Caching is opt-in and
explicit — wrap only the Executor(s) where it makes sense (a judge is
usually stable enough per prompt to cache; a target you're intentionally
sampling at temperature > 0 for diversity probably isn't).

    raw = AnthropicExecutor(client, model="claude-haiku-4-5-20251001")
    judge_executor = CachedExecutor(raw, path=".genevals_cache/judge.jsonl")
    judge = LLMJudge("claude-judge", judge_executor)

With no `path`, the cache is in-memory only (still saves repeat calls within
one process — e.g. the same judge prompt showing up for two different
targets — but nothing persists across runs). With `path`, entries persist as
an append-only JSONL file: simple, git-ignorable, human-inspectable. Not
safe for two processes writing the same cache file concurrently — this is a
dev-loop convenience, not a distributed cache.

Only the text is cached (shared between `.complete()` and `.generate()`) —
a `.generate()` cache hit correctly reports `cost_usd=0.0` and `latency_ms=0.0`
since no new API call was made, with `raw={"cached": True}` marking it as such.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

from genevals.core.types import Output
from genevals.executors.base import Executor


class CachedExecutor(Executor):
    def __init__(self, inner: Executor, *, path: str | Path | None = None):
        self._inner = inner
        self._path = Path(path) if path else None
        self._cache: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                self._cache[entry["key"]] = entry["value"]

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    async def _get_cached(self, key: str) -> str | None:
        async with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    async def _store(self, key: str, text: str) -> None:
        async with self._lock:
            self._cache[key] = text
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": text}) + "\n")

    async def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        cached = await self._get_cached(key)
        if cached is not None:
            return cached

        result = self._inner.complete(prompt)
        if inspect.isawaitable(result):
            result = await result
        await self._store(key, result)
        return result

    async def generate(self, prompt: str) -> Output:
        key = self._key(prompt)
        cached = await self._get_cached(key)
        if cached is not None:
            return Output(text=cached, latency_ms=0.0, cost_usd=0.0, raw={"cached": True})

        output = await self._inner.generate(prompt)
        await self._store(key, output.text)
        return output

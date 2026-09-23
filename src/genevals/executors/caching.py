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
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

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

    async def complete(self, prompt: str) -> str:
        key = self._key(prompt)
        async with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1

        result = self._inner.complete(prompt)
        if inspect.isawaitable(result):
            result = await result

        async with self._lock:
            self._cache[key] = result
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": result}) + "\n")
        return result

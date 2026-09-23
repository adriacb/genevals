import pytest

from genevals.executors.base import Executor
from genevals.executors.caching import CachedExecutor


class CountingExecutor(Executor):
    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"response to {prompt} (#{self.calls})"


@pytest.mark.asyncio
async def test_repeated_prompt_hits_cache_not_the_inner_executor():
    inner = CountingExecutor()
    cached = CachedExecutor(inner)

    first = await cached.complete("hello")
    second = await cached.complete("hello")

    assert first == second
    assert inner.calls == 1
    assert cached.hits == 1
    assert cached.misses == 1


@pytest.mark.asyncio
async def test_different_prompts_both_call_inner():
    inner = CountingExecutor()
    cached = CachedExecutor(inner)

    await cached.complete("a")
    await cached.complete("b")

    assert inner.calls == 2
    assert cached.misses == 2


@pytest.mark.asyncio
async def test_persists_to_disk_and_survives_a_new_instance(tmp_path):
    path = tmp_path / "cache.jsonl"
    inner = CountingExecutor()
    first_wrapper = CachedExecutor(inner, path=path)
    await first_wrapper.complete("hello")
    assert inner.calls == 1

    second_wrapper = CachedExecutor(CountingExecutor(), path=path)
    result = await second_wrapper.complete("hello")

    assert result == "response to hello (#1)"
    assert second_wrapper.hits == 1
    assert second_wrapper._inner.calls == 0  # never touched the new inner executor


@pytest.mark.asyncio
async def test_callable_via_dunder_call_still_caches():
    inner = CountingExecutor()
    cached = CachedExecutor(inner)

    await cached("same prompt")
    await cached("same prompt")

    assert inner.calls == 1

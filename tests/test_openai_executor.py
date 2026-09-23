import asyncio
from dataclasses import dataclass

import pytest

from genevals.executors.openai import OpenAIExecutor


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list
    usage: _Usage | None = None


class _FakeCompletions:
    def __init__(self, *, delay: float = 0.02, text: str = "ok", usage: _Usage | None = None, finish_reason="stop"):
        self.delay = delay
        self.text = text
        self.usage = usage
        self.finish_reason = finish_reason
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def create(self, **kwargs):
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return _Response(
                choices=[_Choice(message=_Message(content=self.text), finish_reason=self.finish_reason)],
                usage=self.usage,
            )
        finally:
            async with self._lock:
                self.in_flight -= 1


class _FakeChat:
    def __init__(self, **kwargs):
        self.completions = _FakeCompletions(**kwargs)


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = _FakeChat(**kwargs)


@pytest.mark.asyncio
async def test_basic_completion():
    client = _FakeClient(text="hello there")
    executor = OpenAIExecutor(client)

    assert await executor.complete("prompt") == "hello there"


@pytest.mark.asyncio
async def test_max_concurrency_caps_in_flight_calls():
    client = _FakeClient(delay=0.03)
    executor = OpenAIExecutor(client, max_concurrency=2)

    await asyncio.gather(*(executor.complete(f"prompt {i}") for i in range(8)))

    assert client.chat.completions.max_in_flight <= 2


@pytest.mark.asyncio
async def test_generate_computes_cost_from_usage():
    client = _FakeClient(usage=_Usage(prompt_tokens=1000, completion_tokens=2000))
    executor = OpenAIExecutor(client, model="gpt-5-mini")

    output = await executor.generate("prompt")

    # gpt-5-mini: $0.25/1M in, $2.00/1M out
    assert output.cost_usd == pytest.approx(1000 * 0.25 / 1e6 + 2000 * 2.00 / 1e6)
    assert output.raw["input_tokens"] == 1000
    assert output.raw["output_tokens"] == 2000


@pytest.mark.asyncio
async def test_raises_when_truncated_with_no_text():
    client = _FakeClient(text="", finish_reason="length")
    executor = OpenAIExecutor(client)

    with pytest.raises(ValueError, match="truncated"):
        await executor.complete("prompt")


@pytest.mark.asyncio
async def test_empty_text_is_fine_when_not_truncated():
    client = _FakeClient(text="", finish_reason="stop")
    executor = OpenAIExecutor(client)

    assert await executor.complete("prompt") == ""

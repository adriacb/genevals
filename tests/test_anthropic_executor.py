import asyncio
from dataclasses import dataclass

import pytest

from genevals.executors.anthropic import AnthropicExecutor


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ThinkingBlock:
    type: str = "thinking"


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Message:
    content: list
    usage: _Usage | None = None


class _FakeMessages:
    """Stands in for client.messages: tracks concurrent in-flight calls."""

    def __init__(self, *, delay: float = 0.02, lead_with_thinking: bool = False, usage: _Usage | None = None):
        self.delay = delay
        self.lead_with_thinking = lead_with_thinking
        self.usage = usage
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def create(self, **kwargs):
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            blocks = [_TextBlock(text="ok")]
            if self.lead_with_thinking:
                blocks = [_ThinkingBlock(), *blocks]
            return _Message(content=blocks, usage=self.usage)
        finally:
            async with self._lock:
                self.in_flight -= 1


class _FakeClient:
    def __init__(self, **kwargs):
        self.messages = _FakeMessages(**kwargs)


@pytest.mark.asyncio
async def test_max_concurrency_caps_in_flight_calls():
    client = _FakeClient(delay=0.03)
    executor = AnthropicExecutor(client, max_concurrency=2)

    await asyncio.gather(*(executor.complete(f"prompt {i}") for i in range(8)))

    assert client.messages.max_in_flight <= 2


@pytest.mark.asyncio
async def test_unbounded_by_default():
    client = _FakeClient(delay=0.03)
    executor = AnthropicExecutor(client)  # no max_concurrency

    await asyncio.gather(*(executor.complete(f"prompt {i}") for i in range(8)))

    assert client.messages.max_in_flight == 8


@pytest.mark.asyncio
async def test_skips_thinking_blocks_to_find_text():
    client = _FakeClient(lead_with_thinking=True)
    executor = AnthropicExecutor(client)

    result = await executor.complete("prompt")

    assert result == "ok"


@pytest.mark.asyncio
async def test_raises_clearly_when_no_text_block_present():
    client = _FakeClient()

    async def create_no_text(**kwargs):
        return _Message(content=[_ThinkingBlock()])

    client.messages.create = create_no_text
    executor = AnthropicExecutor(client)

    with pytest.raises(ValueError, match="no text block"):
        await executor.complete("prompt")


@pytest.mark.asyncio
async def test_generate_computes_cost_from_usage():
    client = _FakeClient(usage=_Usage(input_tokens=1000, output_tokens=2000))
    executor = AnthropicExecutor(client, model="claude-haiku-4-5-20251001")

    output = await executor.generate("prompt")

    # haiku-4-5: $1.00/1M in, $5.00/1M out -> 1000*1.00/1e6 + 2000*5.00/1e6
    assert output.cost_usd == pytest.approx(0.001 + 0.010)
    assert output.raw["input_tokens"] == 1000
    assert output.raw["output_tokens"] == 2000
    assert output.latency_ms is not None and output.latency_ms >= 0


@pytest.mark.asyncio
async def test_generate_cost_is_none_without_usage():
    client = _FakeClient(usage=None)
    executor = AnthropicExecutor(client)

    output = await executor.generate("prompt")

    assert output.cost_usd is None


@pytest.mark.asyncio
async def test_pricing_override_wins_over_default_table():
    client = _FakeClient(usage=_Usage(input_tokens=1_000_000, output_tokens=0))
    executor = AnthropicExecutor(client, model="claude-haiku-4-5-20251001", pricing={"claude-haiku-4-5": (2.0, 0.0)})

    output = await executor.generate("prompt")

    assert output.cost_usd == pytest.approx(2.0)  # overridden $2/1M, not the default $1/1M


@pytest.mark.asyncio
async def test_complete_still_works_via_generate():
    client = _FakeClient(usage=_Usage(input_tokens=10, output_tokens=10))
    executor = AnthropicExecutor(client)

    text = await executor.complete("prompt")

    assert text == "ok"  # complete() discards cost/latency but text is unaffected

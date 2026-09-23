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
class _Message:
    content: list


class _FakeMessages:
    """Stands in for client.messages: tracks concurrent in-flight calls."""

    def __init__(self, *, delay: float = 0.02, lead_with_thinking: bool = False):
        self.delay = delay
        self.lead_with_thinking = lead_with_thinking
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
            return _Message(content=blocks)
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

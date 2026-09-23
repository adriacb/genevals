import pytest

from genevals.core.types import Output, Sample
from genevals.metrics.operational import Cost, Latency


@pytest.mark.asyncio
async def test_latency_reads_output_field():
    result = await Latency().evaluate(Sample(id="1", input="q"), Output(text="a", latency_ms=42.5))
    assert result.value == 42.5


@pytest.mark.asyncio
async def test_cost_reads_output_field():
    result = await Cost().evaluate(Sample(id="1", input="q"), Output(text="a", cost_usd=0.002))
    assert result.value == 0.002


@pytest.mark.asyncio
async def test_latency_is_none_when_not_reported():
    result = await Latency().evaluate(Sample(id="1", input="q"), Output(text="a"))
    assert result.value is None

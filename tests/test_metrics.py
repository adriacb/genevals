import pytest

from genevals.core.types import Output, Sample
from genevals.metrics.text import Contains, ExactMatch, Length, RegexMatch, TokenF1


@pytest.mark.asyncio
async def test_exact_match():
    m = ExactMatch()
    sample = Sample(id="1", input="q", reference="Paris")
    assert (await m.evaluate(sample, Output(text=" paris "))).passed is True
    failure = await m.evaluate(sample, Output(text="London"))
    assert failure.passed is False
    assert "Paris" in failure.rationale and "London" in failure.rationale


@pytest.mark.asyncio
async def test_exact_match_requires_reference():
    m = ExactMatch()
    with pytest.raises(ValueError):
        await m.evaluate(Sample(id="1", input="q"), Output(text="hi"))


@pytest.mark.asyncio
async def test_contains():
    m = Contains()
    sample = Sample(id="1", input="q", reference="capital")
    result = await m.evaluate(sample, Output(text="Paris is the capital of France"))
    assert result.passed is True


@pytest.mark.asyncio
async def test_regex_match():
    m = RegexMatch(r"^\d{3}-\d{4}$", name="phone")
    result = await m.evaluate(Sample(id="1", input="q"), Output(text="555-1234"))
    assert result.passed is True
    assert result.metric == "phone"


@pytest.mark.asyncio
async def test_token_f1_partial_overlap():
    m = TokenF1()
    sample = Sample(id="1", input="q", reference="the cat sat on the mat")
    result = await m.evaluate(sample, Output(text="the cat sat"))
    assert 0.0 < result.value < 1.0
    assert "precision" in result.rationale and "recall" in result.rationale


@pytest.mark.asyncio
async def test_token_f1_perfect_match():
    m = TokenF1()
    sample = Sample(id="1", input="q", reference="the cat sat")
    result = await m.evaluate(sample, Output(text="the cat sat"))
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_length():
    m = Length()
    result = await m.evaluate(Sample(id="1", input="q"), Output(text="hello"))
    assert result.value == 5.0

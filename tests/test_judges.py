import pytest

from genevals.core.types import Output, Sample
from genevals.judges.ensemble import EnsembleJudge
from genevals.judges.llm_judge import LLMJudge
from genevals.judges.metric_adapter import JudgeMetric


@pytest.mark.asyncio
async def test_llm_judge_parses_json_reply():
    judge = LLMJudge("stub", lambda prompt: '{"score": 0.8, "rationale": "close enough"}')
    verdict = await judge.judge(input="q", output="a", reference=None, criteria="relevance")
    assert verdict.score == 0.8
    assert verdict.rationale == "close enough"


@pytest.mark.asyncio
async def test_llm_judge_handles_unparseable_reply():
    judge = LLMJudge("stub", lambda prompt: "not json at all")
    verdict = await judge.judge(input="q", output="a", reference=None, criteria="relevance")
    assert verdict.score is None
    assert "not json" in verdict.rationale


@pytest.mark.asyncio
async def test_llm_judge_supports_async_complete():
    async def complete(prompt: str) -> str:
        return '{"score": 0.5}'

    judge = LLMJudge("stub", complete)
    verdict = await judge.judge(input="q", output="a", reference=None, criteria="c")
    assert verdict.score == 0.5


@pytest.mark.asyncio
async def test_ensemble_judge_means_scores():
    high = LLMJudge("high", lambda p: '{"score": 1.0}')
    low = LLMJudge("low", lambda p: '{"score": 0.0}')
    ensemble = EnsembleJudge([high, low], aggregate="mean")
    verdict = await ensemble.judge(input="q", output="a", reference=None, criteria="c")
    assert verdict.score == 0.5


@pytest.mark.asyncio
async def test_judge_metric_applies_threshold():
    judge = LLMJudge("stub", lambda p: '{"score": 0.9, "rationale": "good"}')
    metric = JudgeMetric("relevance", judge, criteria="is it relevant?", threshold=0.5)
    result = await metric.evaluate(Sample(id="1", input="q"), Output(text="a"))
    assert result.passed is True
    assert result.value == 0.9

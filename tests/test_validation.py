import pytest

from genevals.judges.llm_judge import LLMJudge
from genevals.validation.judge_calibration import LabeledCase, calibrate


@pytest.mark.asyncio
async def test_calibrate_reports_perfect_agreement():
    judge = LLMJudge("stub", lambda p: '{"score": 1.0, "rationale": "matches"}')
    cases = [
        LabeledCase(input="q1", output="a1", reference=None, criteria="c", expected_score=1.0),
        LabeledCase(input="q2", output="a2", reference=None, criteria="c", expected_score=1.0),
    ]
    result = await calibrate(judge, cases)
    assert result.n_cases == 2
    assert result.mean_absolute_error == 0.0
    assert result.disagreements == []


@pytest.mark.asyncio
async def test_calibrate_flags_disagreements():
    judge = LLMJudge("stub", lambda p: '{"score": 0.1}')
    cases = [LabeledCase(input="q", output="a", reference=None, criteria="c", expected_score=0.9)]
    result = await calibrate(judge, cases)
    assert result.mean_absolute_error == pytest.approx(0.8)
    assert len(result.disagreements) == 1

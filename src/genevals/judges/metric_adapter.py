"""Bridge: use any Judge as a catalog Metric."""

from __future__ import annotations

from genevals.core.types import MetricResult, Output, Sample
from genevals.judges.base import Judge
from genevals.metrics.base import Metric


class JudgeMetric(Metric):
    category = "judge"
    needs_reference = False

    def __init__(self, name: str, judge: Judge, criteria: str, *, threshold: float | None = None):
        self.name = name
        self.description = f"LLM-judge metric ({judge.name}): {criteria}"
        self._judge = judge
        self._criteria = criteria
        self._threshold = threshold

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        verdict = await self._judge.judge(
            input=sample.input, output=output.text, reference=sample.reference, criteria=self._criteria
        )
        passed = (
            verdict.score >= self._threshold if verdict.score is not None and self._threshold is not None else None
        )
        return MetricResult(
            metric=self.name,
            value=verdict.score,
            label=verdict.label,
            passed=passed,
            rationale=verdict.rationale,
            raw=verdict.raw,
        )

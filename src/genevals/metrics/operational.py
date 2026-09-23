"""Operational metrics: not correctness, but the numbers a business dashboard tracks alongside it."""

from __future__ import annotations

from genevals.core.types import MetricResult, Output, Sample
from genevals.metrics.base import Metric


class Latency(Metric):
    name = "latency_ms"
    category = "operational"
    needs_reference = False
    description = "Wall-clock time the target took to produce this output, in milliseconds."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        return MetricResult(metric=self.name, value=output.latency_ms)


class Cost(Metric):
    name = "cost_usd"
    category = "operational"
    needs_reference = False
    description = "Cost of producing this output in USD, if the target reported one on its Output."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        return MetricResult(metric=self.name, value=output.cost_usd)

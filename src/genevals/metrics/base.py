"""The Metric protocol every catalog entry implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from genevals.core.types import MetricResult, Output, Sample


class Metric(ABC):
    """A named, catalog-listed way to score one (Sample, Output) pair.

    Subclasses set the class attributes below and implement `evaluate`. A Metric
    can be a deterministic check, a wrapped library metric (e.g. ragas), or an
    LLM-judge under the hood (see genevals.judges.JudgeMetric) — the catalog
    doesn't care which, as long as `evaluate` returns a MetricResult.
    """

    name: str
    category: str = "custom"
    modality: str = "text"
    needs_reference: bool = False
    description: str = ""
    # True: higher score = better (most metrics — correctness, F1, judge scores).
    # False: lower = better (latency, cost). None: no inherent direction
    # (e.g. length) — the report shouldn't imply one either.
    higher_is_better: bool | None = True

    @abstractmethod
    async def evaluate(self, sample: Sample, output: Output) -> MetricResult: ...

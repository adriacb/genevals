"""Optional wrapper around ragas metrics for RAG evaluation.

Requires the 'rag' extra: pip install 'genevals[rag]'

The ragas call is isolated in `RagasMetric.evaluate` so it's the one place to
adjust if ragas's SingleTurnSample/metric API changes between versions. The
class names imported inside `faithfulness()` / `context_precision()` /
`answer_relevancy()` reflect ragas's public API at the time of writing (ragas
>= 0.2) — check `ragas.metrics.__all__` against your installed version if one
of them has moved.
"""

from __future__ import annotations

from typing import Any

from genevals.core.types import MetricResult, Output, Sample
from genevals.metrics.base import Metric

_MISSING_EXTRA = "requires the 'rag' extra: pip install 'genevals[rag]'"


class RagasMetric(Metric):
    """Adapts any ragas metric object into a genevals Metric.

    Retrieved contexts come from `sample.metadata["contexts"]` (or
    `"retrieved_contexts"`) — populate that when building your RAG dataset.
    """

    category = "rag"

    def __init__(self, ragas_metric: Any, *, name: str | None = None, needs_reference: bool = False):
        self.name = name or str(getattr(ragas_metric, "name", type(ragas_metric).__name__))
        self.needs_reference = needs_reference
        self.description = f"ragas metric: {self.name}"
        self._metric = ragas_metric

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        try:
            from ragas.dataset_schema import SingleTurnSample
        except ImportError as exc:
            raise ImportError(f"RagasMetric {_MISSING_EXTRA}") from exc

        contexts = sample.metadata.get("contexts") or sample.metadata.get("retrieved_contexts") or []
        rs = SingleTurnSample(
            user_input=sample.input,
            response=output.text,
            reference=sample.reference,
            retrieved_contexts=contexts,
        )
        score = await self._metric.single_turn_ascore(rs)
        return MetricResult(metric=self.name, value=float(score))


def faithfulness() -> RagasMetric:
    try:
        from ragas.metrics import Faithfulness
    except ImportError as exc:
        raise ImportError(f"faithfulness() {_MISSING_EXTRA}") from exc
    return RagasMetric(Faithfulness(), name="faithfulness")


def context_precision() -> RagasMetric:
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference
    except ImportError as exc:
        raise ImportError(f"context_precision() {_MISSING_EXTRA}") from exc
    return RagasMetric(LLMContextPrecisionWithoutReference(), name="context_precision")


def answer_relevancy() -> RagasMetric:
    try:
        from ragas.metrics import ResponseRelevancy
    except ImportError as exc:
        raise ImportError(f"answer_relevancy() {_MISSING_EXTRA}") from exc
    return RagasMetric(ResponseRelevancy(), name="answer_relevancy")

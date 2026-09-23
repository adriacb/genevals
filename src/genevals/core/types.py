"""Core data types shared across genevals: samples, outputs, scores, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from genevals.gating import GateResult, Threshold
    from genevals.significance import ComparisonResult


def utcnow() -> datetime:
    return datetime.now(UTC)


class Sample(BaseModel):
    """A single evaluation case in a Dataset.

    `tags` is a free-form key/value map (e.g. {"topic": "billing", "difficulty":
    "hard"}) carried through to every SampleResult produced from this sample, so
    reports can be filtered/grouped by it. `conversation_id` + `turn` group
    multi-turn samples for a ChatTarget; leave them unset for one-shot samples.
    """

    id: str
    input: str
    reference: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    turn: int = 0


class Output(BaseModel):
    """What a Target produced for a Sample."""

    text: str
    raw: Any | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    error: str | None = None


class MetricResult(BaseModel):
    """The result of scoring one (Sample, Output) pair with one Metric."""

    metric: str
    value: float | None = None
    passed: bool | None = None
    label: str | None = None
    rationale: str | None = None
    raw: Any | None = None


class SampleResult(BaseModel):
    """Everything produced for one (Target, Sample) pair."""

    sample_id: str
    target: str
    tags: dict[str, str] = Field(default_factory=dict)
    input: str
    reference: str | None = None
    output: Output
    scores: list[MetricResult] = Field(default_factory=list)

    def score(self, metric_name: str) -> MetricResult | None:
        return next((s for s in self.scores if s.metric == metric_name), None)


class EvalReport(BaseModel):
    """The full output of an Evaluator run: results plus aggregate summary."""

    dataset_name: str | None = None
    targets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    # Per-metric name -> Metric.higher_is_better, so a report renderer (the
    # HTML report's ECDF charts) can orient "better" consistently without
    # re-deriving it from raw scores.
    metric_directions: dict[str, bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    results: list[SampleResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        from genevals.reporting.json_yaml import report_to_json

        return report_to_json(self, path, indent=indent)

    def to_yaml(self, path: str | Path | None = None) -> str:
        from genevals.reporting.json_yaml import report_to_yaml

        return report_to_yaml(self, path)

    def to_html(self, path: str | Path) -> None:
        from genevals.reporting.html import report_to_html

        report_to_html(self, path)

    def check_thresholds(self, thresholds: list[Threshold]) -> GateResult:
        from genevals.gating import check_thresholds

        return check_thresholds(self, thresholds)

    def assert_thresholds(self, thresholds: list[Threshold]) -> None:
        from genevals.gating import assert_thresholds

        assert_thresholds(self, thresholds)

    def compare_targets(
        self,
        metric: str,
        target_a: str,
        target_b: str,
        *,
        n_resamples: int = 10_000,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> ComparisonResult:
        from genevals.significance import compare_targets

        return compare_targets(
            self, metric, target_a, target_b, n_resamples=n_resamples, alpha=alpha, seed=seed
        )

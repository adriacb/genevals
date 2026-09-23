"""Core data types shared across genevals: samples, outputs, scores, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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

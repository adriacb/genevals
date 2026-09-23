"""Deterministic, dependency-free text metrics."""

from __future__ import annotations

import re

from genevals.core.types import MetricResult, Output, Sample
from genevals.metrics.base import Metric

_TRUNCATE_AT = 120


def _require_reference(sample: Sample, metric_name: str) -> str:
    if sample.reference is None:
        raise ValueError(f"'{metric_name}' requires sample.reference to be set")
    return sample.reference


def _truncate(text: str, limit: int = _TRUNCATE_AT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ExactMatch(Metric):
    name = "exact_match"
    category = "lexical"
    needs_reference = True
    description = "Case/whitespace-insensitive exact match against sample.reference."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        reference = _require_reference(sample, self.name)
        actual, expected = output.text.strip(), reference.strip()
        passed = actual.lower() == expected.lower()
        rationale = (
            "output matches the reference exactly"
            if passed
            else f"expected {_truncate(expected)!r}, got {_truncate(actual)!r}"
        )
        return MetricResult(metric=self.name, value=1.0 if passed else 0.0, passed=passed, rationale=rationale)


class Contains(Metric):
    name = "contains"
    category = "lexical"
    needs_reference = True
    description = "Whether sample.reference appears as a substring of the output (case-insensitive)."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        reference = _require_reference(sample, self.name)
        expected = reference.strip()
        passed = expected.lower() in output.text.lower()
        rationale = (
            f"found {_truncate(expected)!r} in the output"
            if passed
            else f"{_truncate(expected)!r} was not found in the output"
        )
        return MetricResult(metric=self.name, value=1.0 if passed else 0.0, passed=passed, rationale=rationale)


class RegexMatch(Metric):
    """Configurable per instance, so register it under your own name if you keep it around."""

    category = "lexical"
    needs_reference = False

    def __init__(self, pattern: str, *, name: str | None = None, flags: int = re.IGNORECASE):
        self.name = name or f"regex:{pattern}"
        self.description = f"Whether the output matches /{pattern}/"
        self._regex = re.compile(pattern, flags)

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        match = self._regex.search(output.text)
        passed = match is not None
        rationale = (
            f"matched /{self._regex.pattern}/ on {_truncate(match.group(0))!r}"
            if match
            else f"no match for /{self._regex.pattern}/ in the output"
        )
        return MetricResult(metric=self.name, value=1.0 if passed else 0.0, passed=passed, rationale=rationale)


class TokenF1(Metric):
    name = "token_f1"
    category = "lexical"
    needs_reference = True
    description = "SQuAD-style whitespace-token overlap F1 against sample.reference."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        reference = _require_reference(sample, self.name)
        pred_tokens = output.text.lower().split()
        ref_tokens = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            f1 = 1.0 if pred_tokens == ref_tokens else 0.0
            rationale = "both output and reference are empty" if f1 == 1.0 else "output or reference is empty"
            return MetricResult(metric=self.name, value=f1, passed=f1 >= 0.5, rationale=rationale)

        num_same = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
        if num_same == 0:
            return MetricResult(
                metric=self.name, value=0.0, passed=False, rationale="no tokens in common with the reference"
            )
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        rationale = f"token overlap F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f}, {num_same} shared tokens)"
        return MetricResult(metric=self.name, value=f1, passed=f1 >= 0.5, rationale=rationale)


class Length(Metric):
    name = "length"
    category = "shape"
    needs_reference = False
    higher_is_better = None
    description = "Character length of the output text (informational, no pass/fail)."

    async def evaluate(self, sample: Sample, output: Output) -> MetricResult:
        return MetricResult(metric=self.name, value=float(len(output.text)))

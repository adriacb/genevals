"""Aggregate SampleResults into per-target/per-metric summary stats."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean as _mean
from typing import Any

from genevals.core.types import SampleResult


def compute_summary(results: list[SampleResult]) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    passes: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for result in results:
        for score in result.scores:
            key = (result.target, score.metric)
            if score.value is not None:
                values[key].append(score.value)
            if score.passed is not None:
                passes[key].append(score.passed)

    per_target: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (target, metric), vals in values.items():
        per_target[target].setdefault(metric, {})["mean"] = _mean(vals)
        per_target[target][metric]["n"] = len(vals)
    for (target, metric), flags in passes.items():
        per_target[target].setdefault(metric, {})["pass_rate"] = sum(flags) / len(flags)

    return {
        "n_results": len(results),
        "n_errors": sum(1 for r in results if r.output.error),
        "targets": sorted({r.target for r in results}),
        "metrics": sorted({s.metric for r in results for s in r.scores}),
        "per_target": dict(per_target),
    }

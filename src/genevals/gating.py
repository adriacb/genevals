"""CI gating: assert a report's metrics clear a quality bar.

    from genevals.gating import Threshold

    report = Evaluator(dataset, [target], metrics).run()
    report.assert_thresholds([Threshold("exact_match", ">=", 0.8)])  # raises ThresholdViolation

Or without raising, for building your own reporting on top:

    gate = report.check_thresholds([Threshold("correctness", ">=", 0.7, target="my-flow")])
    if not gate.passed:
        for failure in gate.failures():
            print(failure)

`stat` picks which per-target summary statistic to check — "pass_rate"
(default, needs metrics that set `passed`) or "mean". A `Threshold` with no
`target` is checked against every target in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from genevals.core.types import EvalReport

Comparator = Literal[">=", ">", "<=", "<", "==", "!="]

_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class Threshold:
    metric: str
    comparator: Comparator
    value: float
    target: str | None = None
    stat: Literal["mean", "pass_rate"] = "pass_rate"


@dataclass
class ThresholdResult:
    threshold: Threshold
    target: str
    actual: float | None
    passed: bool

    def __str__(self) -> str:
        if self.actual is None:
            return (
                f"{self.threshold.metric} ({self.threshold.stat}) for target '{self.target}': "
                f"no data (metric not present, or 'stat' doesn't apply to it)"
            )
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.threshold.metric} ({self.threshold.stat}) for target '{self.target}': "
            f"{self.actual:.3f} {self.threshold.comparator} {self.threshold.value}"
        )


@dataclass
class GateResult:
    results: list[ThresholdResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failures(self) -> list[ThresholdResult]:
        return [r for r in self.results if not r.passed]


class ThresholdViolation(AssertionError):
    pass


def check_thresholds(report: EvalReport, thresholds: list[Threshold]) -> GateResult:
    per_target = report.summary.get("per_target", {})
    results: list[ThresholdResult] = []
    for threshold in thresholds:
        target_names = [threshold.target] if threshold.target else sorted(per_target)
        for target_name in target_names:
            stats = per_target.get(target_name, {}).get(threshold.metric)
            actual = stats.get(threshold.stat) if stats else None
            passed = actual is not None and _OPS[threshold.comparator](actual, threshold.value)
            results.append(ThresholdResult(threshold=threshold, target=target_name, actual=actual, passed=passed))
    return GateResult(results=results)


def assert_thresholds(report: EvalReport, thresholds: list[Threshold]) -> None:
    gate = check_thresholds(report, thresholds)
    if not gate.passed:
        details = "\n".join(str(r) for r in gate.failures())
        raise ThresholdViolation(f"{len(gate.failures())} threshold(s) failed:\n{details}")

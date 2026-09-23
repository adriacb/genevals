"""Is target A actually different from target B, or is it noise?

    from genevals.significance import compare_targets

    result = report.compare_targets("correctness", "direct", "chain-of-thought")
    print(result)
    # direct vs chain-of-thought on correctness (n=12): mean_diff=-0.013 (95% CI
    # [-0.089, +0.061]), p=0.735 -> not significant

Uses paired bootstrap resampling rather than a parametric test (a t-test's
normality assumption often doesn't hold for eval-metric scores, which are
usually bounded — 0/1 pass-fail, 0-1 judge scores — and eval sample sizes
tend to be small). The two targets must share sample_ids for the given
metric — this is a *paired* comparison (same input, two pipelines), not an
independent-samples one.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genevals.core.types import EvalReport


@dataclass
class ComparisonResult:
    metric: str
    target_a: str
    target_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean_a - mean_b
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool  # 95% CI excludes 0 (equivalently, two-sided p < 0.05)

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.target_a} vs {self.target_b} on {self.metric} (n={self.n}): "
            f"mean_diff={self.mean_diff:+.3f} (95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]), "
            f"p={self.p_value:.3f} -> {verdict}"
        )


def compare_targets(
    report: EvalReport,
    metric: str,
    target_a: str,
    target_b: str,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> ComparisonResult:
    scores_a = _scores_by_sample(report, target_a, metric)
    scores_b = _scores_by_sample(report, target_b, metric)
    shared_ids = sorted(set(scores_a) & set(scores_b))
    if not shared_ids:
        raise ValueError(
            f"no samples have a numeric '{metric}' score for both '{target_a}' and '{target_b}' — "
            "a paired comparison needs matching sample_ids scored on both targets"
        )

    a = [scores_a[i] for i in shared_ids]
    b = [scores_b[i] for i in shared_ids]
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)

    rng = random.Random(seed)
    boot_means = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples))

    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)
    ci_low, ci_high = boot_means[lo_idx], boot_means[hi_idx]

    observed = statistics.fmean(diffs)
    # two-sided p-value: fraction of bootstrap resamples that crossed to the
    # opposite side of zero from the observed difference, doubled
    crossed = sum(1 for m in boot_means if m <= 0) if observed >= 0 else sum(1 for m in boot_means if m >= 0)
    p_value = min(1.0, 2 * crossed / n_resamples)

    return ComparisonResult(
        metric=metric,
        target_a=target_a,
        target_b=target_b,
        n=n,
        mean_a=statistics.fmean(a),
        mean_b=statistics.fmean(b),
        mean_diff=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant=not (ci_low <= 0 <= ci_high),
    )


def _scores_by_sample(report: EvalReport, target: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in report.results:
        if r.target != target:
            continue
        score = r.score(metric)
        if score is not None and score.value is not None:
            out[r.sample_id] = score.value
    return out

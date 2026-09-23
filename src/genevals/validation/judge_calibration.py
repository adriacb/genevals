"""Validate a Judge against a labeled gold set before trusting it for gating.

Motivated by the "reliability without validity" finding for LLM/decision-model
judges (arxiv:2606.19544): a judge can be highly self-consistent while still
disagreeing with human judgment. This checks agreement against labels you
already trust (a RewardBench/JudgeBench-style set, or your own hand-labeled
sample) rather than taking a judge's self-reported benchmark at face value.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from genevals.judges.base import Judge, JudgeVerdict


@dataclass
class LabeledCase:
    input: str
    output: str
    reference: str | None
    criteria: str
    expected_score: float | None = None  # 0-1 ground truth, for error/correlation
    expected_label: str | None = None  # ground truth label, for accuracy


@dataclass
class CalibrationReport:
    n_cases: int
    label_accuracy: float | None
    mean_absolute_error: float | None
    pearson_r: float | None
    disagreements: list[tuple[LabeledCase, JudgeVerdict]]


async def calibrate(judge: Judge, cases: list[LabeledCase], *, concurrency: int = 8) -> CalibrationReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(case: LabeledCase) -> JudgeVerdict:
        async with semaphore:
            return await judge.judge(
                input=case.input, output=case.output, reference=case.reference, criteria=case.criteria
            )

    verdicts = list(await asyncio.gather(*(run_one(c) for c in cases)))

    label_hits = [
        c.expected_label == v.label
        for c, v in zip(cases, verdicts, strict=True)
        if c.expected_label is not None
    ]
    label_accuracy = sum(label_hits) / len(label_hits) if label_hits else None

    score_pairs = [
        (c.expected_score, v.score)
        for c, v in zip(cases, verdicts, strict=True)
        if c.expected_score is not None and v.score is not None
    ]
    mae = sum(abs(e - a) for e, a in score_pairs) / len(score_pairs) if score_pairs else None
    pearson_r = _pearson(score_pairs) if len(score_pairs) >= 2 else None

    disagreements = [
        (c, v)
        for c, v in zip(cases, verdicts, strict=True)
        if (c.expected_label is not None and c.expected_label != v.label)
        or (c.expected_score is not None and v.score is not None and abs(c.expected_score - v.score) >= 0.3)
    ]

    return CalibrationReport(
        n_cases=len(cases),
        label_accuracy=label_accuracy,
        mean_absolute_error=mae,
        pearson_r=pearson_r,
        disagreements=disagreements,
    )


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    n = len(pairs)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom else None

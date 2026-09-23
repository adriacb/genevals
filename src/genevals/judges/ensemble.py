"""Combine multiple judges to mitigate single-judge bias (position/verbosity/self-preference)."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Literal

from genevals.judges.base import Judge, JudgeVerdict


class EnsembleJudge(Judge):
    def __init__(self, judges: list[Judge], *, aggregate: Literal["mean", "majority"] = "mean"):
        if not judges:
            raise ValueError("EnsembleJudge needs at least one judge")
        self.name = "ensemble(" + ",".join(j.name for j in judges) + ")"
        self._judges = judges
        self._aggregate = aggregate

    async def judge(self, *, input: str, output: str, reference: str | None, criteria: str) -> JudgeVerdict:
        raw_results = await asyncio.gather(
            *(j.judge(input=input, output=output, reference=reference, criteria=criteria) for j in self._judges),
            return_exceptions=True,
        )

        verdicts: list[JudgeVerdict] = []
        rationale_parts: list[str] = []
        for member, result in zip(self._judges, raw_results, strict=True):
            if isinstance(result, BaseException):
                rationale_parts.append(f"{member.name}: FAILED ({type(result).__name__}: {result})")
                continue
            verdicts.append(result)
            if result.rationale:
                rationale_parts.append(f"{member.name}: {result.rationale}")
        rationale = "; ".join(rationale_parts)

        if not verdicts:
            return JudgeVerdict(rationale=rationale or "every ensemble member failed")

        if self._aggregate == "mean":
            scores = [v.score for v in verdicts if v.score is not None]
            score = sum(scores) / len(scores) if scores else None
            return JudgeVerdict(score=score, rationale=rationale, raw=[v.raw for v in verdicts])

        labels = [v.label for v in verdicts if v.label is not None]
        label = Counter(labels).most_common(1)[0][0] if labels else None
        return JudgeVerdict(label=label, rationale=rationale, raw=[v.raw for v in verdicts])

"""EXPERIMENTAL: judge backend for TypeSafe AI's Jev ("System One Model").

Jev launched September 2026 as a waitlisted, closed-weight API that returns
typed decisions (choice / score / yes-no) instead of generated text, pitched as
a much cheaper and faster alternative to LLM-as-a-judge for high-volume gating.
genevals does not use it anywhere by default — instantiate it explicitly.

Before trusting it for anything that gates a release, validate it against a
labeled set (see genevals.validation) rather than assuming its self-reported
benchmark numbers transfer to your task: LLM/decision-model judges can be
internally consistent while still being wrong relative to human judgment
(see arxiv:2606.19544, "Reliability without Validity").

The request/response shape below follows TypeSafe's public description of the
`/v1/systemone` endpoint. It has NOT been verified against official API docs
(access is waitlisted at the time of writing) — check TypeSafe's docs once you
have access and adjust `_build_payload` / `_parse_response` if the real schema
differs.
"""

from __future__ import annotations

from typing import Literal

from genevals.judges.base import Judge, JudgeVerdict

QuestionType = Literal["score", "choice", "noul"]


class JevJudge(Judge):
    name = "jev"
    BASE_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_key: str, *, question_type: QuestionType = "score", timeout: float = 5.0):
        self._api_key = api_key
        self._question_type: QuestionType = question_type
        self._timeout = timeout

    async def judge(self, *, input: str, output: str, reference: str | None, criteria: str) -> JudgeVerdict:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("JevJudge requires the 'jev' extra: pip install 'genevals[jev]'") from exc

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self.BASE_URL,
                json=self._build_payload(input=input, output=output, reference=reference, criteria=criteria),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return self._parse_response(response.json())

    def _build_payload(self, *, input: str, output: str, reference: str | None, criteria: str) -> dict:
        return {
            "type": self._question_type,
            "context": {"input": input, "output": output, "reference": reference, "criteria": criteria},
        }

    @staticmethod
    def _parse_response(data: dict) -> JudgeVerdict:
        return JudgeVerdict(
            score=data.get("score"),
            label=data.get("choice") or data.get("label"),
            rationale=data.get("rationale"),
            raw=data,
        )

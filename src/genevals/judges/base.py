"""The Judge protocol: takes an (input, output) pair and returns a typed verdict."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class JudgeVerdict(BaseModel):
    score: float | None = None
    label: str | None = None
    rationale: str | None = None
    raw: Any | None = None


class Judge(ABC):
    name: str

    @abstractmethod
    async def judge(self, *, input: str, output: str, reference: str | None, criteria: str) -> JudgeVerdict: ...

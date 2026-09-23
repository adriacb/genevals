"""The Target protocol: the system under test (a model call, RAG pipeline, agent, or chatbot)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from genevals.core.types import Output, Sample


class Target(ABC):
    name: str
    tags: dict[str, str]

    @abstractmethod
    async def agenerate(self, sample: Sample) -> Output: ...

"""The EvalBackend protocol: how (Dataset, Targets, Metrics) turn into SampleResults.

genevals ships NativeBackend (the default, no extra dependencies) and an
optional adapter for UK AISI's inspect_ai (genevals.backends.inspect_ai_backend).
Implement this protocol to plug in another harness — e.g. lm-evaluation-harness
— without touching the rest of the package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from genevals.core.dataset import Dataset
from genevals.core.types import SampleResult
from genevals.events import Listener
from genevals.metrics.base import Metric
from genevals.targets.base import Target


class EvalBackend(ABC):
    @abstractmethod
    async def run(
        self,
        dataset: Dataset,
        targets: list[Target],
        metrics: list[Metric],
        *,
        concurrency: int = 8,
        on_event: Listener | None = None,
    ) -> list[SampleResult]: ...

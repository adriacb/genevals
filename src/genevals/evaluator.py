"""High-level entry point: Evaluator(dataset, targets, metrics).run() -> EvalReport."""

from __future__ import annotations

import asyncio

from genevals.backends.base import EvalBackend
from genevals.backends.native import NativeBackend
from genevals.core.dataset import Dataset
from genevals.core.types import EvalReport
from genevals.events import Listener
from genevals.metrics.base import Metric
from genevals.reporting.summary import compute_summary
from genevals.targets.base import Target


class Evaluator:
    """Runs `targets` (in parallel) over `dataset`, scoring each output with `metrics`.

    `targets` being a list is what makes "parallel pipelines" work: pass every
    flow/model you want to compare and they're evaluated on the same dataset in
    one run, distinguishable in the report by `target` name and `tags`.

    Pass `on_event=` to observe the run while it's happening (progress,
    live failure notifications) — see genevals.events. Failures are recorded
    in the returned EvalReport either way.
    """

    def __init__(
        self,
        dataset: Dataset,
        targets: list[Target],
        metrics: list[Metric],
        *,
        backend: EvalBackend | None = None,
        concurrency: int = 8,
        on_event: Listener | None = None,
    ):
        self.dataset = dataset
        self.targets = targets
        self.metrics = metrics
        self.backend = backend or NativeBackend()
        self.concurrency = concurrency
        self.on_event = on_event

    async def run_async(self) -> EvalReport:
        results = await self.backend.run(
            self.dataset, self.targets, self.metrics, concurrency=self.concurrency, on_event=self.on_event
        )
        return EvalReport(
            dataset_name=self.dataset.name,
            targets=[t.name for t in self.targets],
            metrics=[m.name for m in self.metrics],
            results=results,
            summary=compute_summary(results),
        )

    def run(self) -> EvalReport:
        return asyncio.run(self.run_async())

"""The built-in metrics catalog: discover what's available before wiring an eval."""

from __future__ import annotations

from genevals.metrics.base import Metric
from genevals.metrics.operational import Cost, Latency
from genevals.metrics.text import Contains, ExactMatch, Length, TokenF1
from genevals.registry import Registry


class MetricCatalog(Registry[Metric]):
    def __init__(self) -> None:
        super().__init__(kind="metric")

    def search(self, *, category: str | None = None, modality: str | None = None) -> list[Metric]:
        return [
            m
            for m in self
            if (category is None or m.category == category) and (modality is None or m.modality == modality)
        ]

    def describe(self) -> list[dict[str, str | bool]]:
        return [
            {
                "name": m.name,
                "category": m.category,
                "modality": m.modality,
                "needs_reference": m.needs_reference,
                "description": m.description,
            }
            for m in self
        ]


CATALOG = MetricCatalog()
for _metric in (ExactMatch(), Contains(), TokenF1(), Length(), Latency(), Cost()):
    CATALOG.register(_metric.name, _metric)

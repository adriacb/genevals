"""Load the use-case -> metrics preset catalog and resolve it into Metric instances.

    from genevals.presets import USE_CASES
    from genevals.judges.llm_judge import LLMJudge

    judge = LLMJudge("my-judge", my_completion_fn)
    metrics = USE_CASES.get("chatbot_tools").metrics(judge=judge)
    report = Evaluator(dataset, [target], metrics).run()

See use_cases.yaml for the defaults and the "kind" values it supports. Point
UseCaseCatalog(path=...) at your own YAML file (same shape) to customize.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from genevals.judges.base import Judge
from genevals.judges.metric_adapter import JudgeMetric
from genevals.metrics.base import Metric
from genevals.metrics.catalog import CATALOG


@dataclass
class UseCase:
    name: str
    description: str
    spec: list[dict[str, Any]]

    def metrics(self, *, judge: Judge | None = None) -> list[Metric]:
        """Resolve this use case's metric specs into ready-to-use Metric instances.

        `judge` is required only if an entry has kind: judge — pass whichever
        Judge you want scoring those dimensions (LLMJudge, EnsembleJudge, ...).
        """
        resolved: list[Metric] = []
        for entry in self.spec:
            kind = entry["kind"]
            if kind == "catalog":
                resolved.append(CATALOG.get(entry["catalog_name"]))
            elif kind == "judge":
                if judge is None:
                    raise ValueError(
                        f"use case '{self.name}' metric '{entry['name']}' needs a judge — "
                        "pass judge=... to UseCase.metrics()"
                    )
                resolved.append(
                    JudgeMetric(entry["name"], judge, criteria=entry["criteria"], threshold=entry.get("threshold"))
                )
            elif kind == "rag":
                from genevals.metrics import rag as rag_metrics

                factory = getattr(rag_metrics, entry["rag_metric"])
                resolved.append(factory())
            else:
                raise ValueError(f"unknown metric kind '{kind}' for '{entry['name']}' in use case '{self.name}'")
        return resolved


class UseCaseCatalog:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            text = resources.files("genevals.presets").joinpath("use_cases.yaml").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text) or {}
        self._use_cases = {
            name: UseCase(name=name, description=spec.get("description", ""), spec=spec.get("metrics", []))
            for name, spec in doc.get("use_cases", {}).items()
        }

    def names(self) -> list[str]:
        return sorted(self._use_cases)

    def get(self, name: str) -> UseCase:
        try:
            return self._use_cases[name]
        except KeyError:
            raise KeyError(f"Unknown use case '{name}'. Available: {self.names()}") from None

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": uc.name, "description": uc.description, "metrics": [m["name"] for m in uc.spec]}
            for uc in self._use_cases.values()
        ]


USE_CASES = UseCaseCatalog()

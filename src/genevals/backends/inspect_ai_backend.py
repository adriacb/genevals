"""Adapter that runs genevals Targets/Metrics on top of UK AISI's inspect_ai.

Requires the 'inspect' extra: pip install 'genevals[inspect]'

This is the harness-side of genevals' multi-backend design (see backends.base):
NativeBackend needs nothing beyond genevals itself; this backend hands the same
(Dataset, Targets, Metrics) to inspect_ai's Task/Solver/Scorer machinery
instead, picking up its sandboxing, logging, and prebuilt-eval ecosystem.

Written against inspect_ai's public Task/Solver/Scorer API (`solver`/`scorer`
decorators, `TaskState.store`, `EvalLog.samples[i].scores`). If something here
raises an AttributeError/TypeError against your installed version, that
surface is the first place to check — pin a known-good `inspect-ai` version
via the 'inspect' extra once you've verified it against your environment.
"""

from __future__ import annotations

from genevals.backends.base import EvalBackend
from genevals.core.dataset import Dataset
from genevals.core.types import MetricResult, Output, SampleResult
from genevals.core.types import Sample as GESample
from genevals.events import Listener
from genevals.metrics.base import Metric
from genevals.targets.base import Target


class InspectAIBackend(EvalBackend):
    async def run(
        self,
        dataset: Dataset,
        targets: list[Target],
        metrics: list[Metric],
        *,
        concurrency: int = 8,
        on_event: Listener | None = None,  # not wired up yet — inspect_ai has its own logging/progress
    ) -> list[SampleResult]:
        try:
            from inspect_ai import Task
            from inspect_ai import eval_async as inspect_eval_async
            from inspect_ai.dataset import MemoryDataset
            from inspect_ai.dataset import Sample as InspectSample
            from inspect_ai.scorer import Score, mean, scorer
            from inspect_ai.scorer import Target as InspectTarget
            from inspect_ai.solver import Generate, TaskState, solver
        except ImportError as exc:
            raise ImportError(
                "InspectAIBackend requires the 'inspect' extra: pip install 'genevals[inspect]'"
            ) from exc

        by_id = {s.id: s for s in dataset}
        inspect_dataset = MemoryDataset(
            [
                InspectSample(id=s.id, input=s.input, target=s.reference or "", metadata={**s.tags, **s.metadata})
                for s in dataset
            ]
        )

        def make_solver(target: Target):
            @solver(name=f"genevals:{target.name}")
            def _factory():
                async def solve(state: TaskState, generate: Generate) -> TaskState:
                    ge_sample = by_id[str(state.sample_id)]
                    output = await target.agenerate(ge_sample)
                    state.output.completion = output.text
                    state.store.set("genevals_output", output.model_dump())
                    return state

                return solve

            return _factory()

        def make_scorer(metric: Metric):
            @scorer(name=f"genevals:{metric.name}", metrics=[mean()])
            def _factory():
                async def score(state: TaskState, target: InspectTarget) -> Score:
                    ge_sample = by_id[str(state.sample_id)]
                    raw_output = state.store.get("genevals_output") or {"text": state.output.completion}
                    output = Output(**raw_output)
                    result: MetricResult = await metric.evaluate(ge_sample, output)
                    value = result.value if result.value is not None else (1.0 if result.passed else 0.0)
                    return Score(value=value, explanation=result.rationale, metadata=result.model_dump())

                return score

            return _factory()

        results: list[SampleResult] = []
        for target in targets:
            task = Task(
                dataset=inspect_dataset,
                solver=make_solver(target),
                scorer=[make_scorer(m) for m in metrics],
            )
            [log] = await inspect_eval_async(task, max_connections=concurrency, display="none")
            for eval_sample in log.samples or []:
                ge_sample: GESample = by_id[str(eval_sample.id)]
                completion = eval_sample.output.completion if eval_sample.output else ""
                scores = [
                    MetricResult(
                        metric=metric_name,
                        value=score.value if isinstance(score.value, (int, float)) else None,
                        rationale=score.explanation,
                        raw=score.metadata,
                    )
                    for metric_name, score in (eval_sample.scores or {}).items()
                ]
                results.append(
                    SampleResult(
                        sample_id=str(eval_sample.id),
                        target=target.name,
                        tags={**ge_sample.tags, **target.tags},
                        input=ge_sample.input,
                        reference=ge_sample.reference,
                        output=Output(text=completion),
                        scores=scores,
                    )
                )
        return results

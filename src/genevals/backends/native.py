"""The default backend: a plain asyncio runner, no external harness required."""

from __future__ import annotations

import asyncio
import warnings
from collections import defaultdict

from genevals.backends.base import EvalBackend
from genevals.core.dataset import Dataset
from genevals.core.types import MetricResult, Output, Sample, SampleResult
from genevals.events import (
    GroupFailed,
    Listener,
    MetricFailed,
    RunCompleted,
    RunStarted,
    SampleCompleted,
    SampleStarted,
    TargetFailed,
    emit,
)
from genevals.metrics.base import Metric
from genevals.targets.base import Target


class NativeBackend(EvalBackend):
    """Runs every (target, sample) pair with bounded concurrency.

    Samples sharing a `conversation_id` run in `turn` order, sequentially,
    against a given target (so ChatTarget history stays correct); independent
    conversations and different targets all run concurrently up to
    `concurrency`.

    `concurrency` bounds how many (target, sample) units are in flight at
    once — it does NOT bound the total number of API calls, since a single
    unit can fan out into several judge/ensemble calls. Rate-limit
    protection for a specific provider belongs on that provider's Executor
    (e.g. `AnthropicExecutor(max_concurrency=...)`), which is scoped to the
    API key rather than to genevals' sample orchestration.

    A failing target or metric never aborts the run: it's recorded on that
    one SampleResult (as `output.error`, or as a MetricResult whose
    `rationale` explains the failure) and everything else keeps going. Only
    a genevals bug in a group's own orchestration (not target/metric code)
    can lose a whole group — that always surfaces as a warning, and also as
    a GroupFailed event if you pass `on_event=`. See genevals.events for the
    full set of events and why they exist alongside that error handling,
    not instead of it.
    """

    async def run(
        self,
        dataset: Dataset,
        targets: list[Target],
        metrics: list[Metric],
        *,
        concurrency: int = 8,
        on_event: Listener | None = None,
    ) -> list[SampleResult]:
        groups: dict[str, list[Sample]] = defaultdict(list)
        for sample in dataset:
            groups[sample.conversation_id or sample.id].append(sample)
        for group in groups.values():
            group.sort(key=lambda s: s.turn)

        await emit(on_event, RunStarted(n_samples=len(dataset), n_targets=len(targets), n_metrics=len(metrics)))

        semaphore = asyncio.Semaphore(concurrency)

        async def run_group(target: Target, samples: list[Sample]) -> list[SampleResult]:
            async with semaphore:
                return [await self._run_one(target, sample, metrics, on_event) for sample in samples]

        units = [(target, samples) for target in targets for samples in groups.values()]
        grouped = await asyncio.gather(
            *(run_group(target, samples) for target, samples in units), return_exceptions=True
        )
        results: list[SampleResult] = []
        for (target, _samples), outcome in zip(units, grouped, strict=True):
            if isinstance(outcome, BaseException):
                message = f"{type(outcome).__name__}: {outcome}"
                warnings.warn(
                    f"a NativeBackend group failed outside target/metric error handling "
                    f"({message}) — its samples are missing from this report",
                    stacklevel=2,
                )
                await emit(on_event, GroupFailed(target=target.name, error=message))
                continue
            results.extend(outcome)

        await emit(
            on_event,
            RunCompleted(n_results=len(results), n_errors=sum(1 for r in results if r.output.error)),
        )
        return results

    async def _run_one(
        self, target: Target, sample: Sample, metrics: list[Metric], on_event: Listener | None
    ) -> SampleResult:
        await emit(on_event, SampleStarted(sample_id=sample.id, target=target.name))

        try:
            output = await target.agenerate(sample)
        except Exception as exc:  # noqa: BLE001 - user-supplied target code; must not kill the whole run
            error = f"{type(exc).__name__}: {exc}"
            output = Output(text="", error=error)
            await emit(on_event, TargetFailed(sample_id=sample.id, target=target.name, error=error))

        scores: list[MetricResult] = []
        if not output.error:
            applicable = [m for m in metrics if not m.needs_reference or sample.reference is not None]
            scores = list(
                await asyncio.gather(
                    *(self._safe_evaluate(m, sample, output, target.name, on_event) for m in applicable)
                )
            )

        await emit(
            on_event,
            SampleCompleted(sample_id=sample.id, target=target.name, had_error=bool(output.error)),
        )

        return SampleResult(
            sample_id=sample.id,
            target=target.name,
            tags={**sample.tags, **target.tags},
            input=sample.input,
            reference=sample.reference,
            output=output,
            scores=scores,
        )

    @staticmethod
    async def _safe_evaluate(
        metric: Metric, sample: Sample, output: Output, target_name: str, on_event: Listener | None
    ) -> MetricResult:
        try:
            return await metric.evaluate(sample, output)
        except Exception as exc:  # noqa: BLE001 - a metric/judge failure must not sink the whole sample
            error = f"{type(exc).__name__}: {exc}"
            await emit(
                on_event, MetricFailed(sample_id=sample.id, target=target_name, metric=metric.name, error=error)
            )
            return MetricResult(metric=metric.name, rationale=f"metric raised {error}")

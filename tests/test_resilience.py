"""A failing metric, judge, or group must never abort the whole run — and it
must be observable via events while it's happening, not just discoverable
afterwards in the report. See NativeBackend's docstring and genevals.events.
"""

import warnings

import pytest

from genevals.backends.native import NativeBackend
from genevals.core.dataset import Dataset
from genevals.core.types import Sample
from genevals.evaluator import Evaluator
from genevals.events import (
    GroupFailed,
    MetricFailed,
    RunCompleted,
    RunStarted,
    SampleCompleted,
    SampleStarted,
    TargetFailed,
)
from genevals.judges.ensemble import EnsembleJudge
from genevals.judges.llm_judge import LLMJudge
from genevals.metrics.base import Metric
from genevals.metrics.text import Length
from genevals.targets.simple import SimpleTarget


class ExplodingMetric(Metric):
    name = "exploding"
    category = "custom"
    needs_reference = False

    async def evaluate(self, sample, output):
        raise RuntimeError("boom")


def make_dataset() -> Dataset:
    return Dataset([Sample(id="1", input="hi"), Sample(id="2", input="bye")])


@pytest.mark.asyncio
async def test_failing_metric_does_not_crash_the_run():
    target = SimpleTarget("t", lambda s: "ok")
    report = await Evaluator(make_dataset(), [target], [ExplodingMetric(), Length()]).run_async()

    assert len(report.results) == 2
    for r in report.results:
        exploding = r.score("exploding")
        length = r.score("length")
        assert exploding.rationale is not None and "boom" in exploding.rationale
        assert exploding.value is None
        assert length.value == 2.0  # the other metric on the same sample still scored fine


@pytest.mark.asyncio
async def test_events_fire_in_order_for_a_clean_run():
    events = []
    target = SimpleTarget("t", lambda s: "ok")
    await Evaluator(make_dataset(), [target], [Length()], on_event=events.append).run_async()

    assert isinstance(events[0], RunStarted)
    assert events[0].n_samples == 2
    assert isinstance(events[-1], RunCompleted)
    assert events[-1].n_results == 2
    assert events[-1].n_errors == 0
    sample_starts = [e for e in events if isinstance(e, SampleStarted)]
    sample_completes = [e for e in events if isinstance(e, SampleCompleted)]
    assert len(sample_starts) == 2
    assert len(sample_completes) == 2
    assert all(not e.had_error for e in sample_completes)


@pytest.mark.asyncio
async def test_target_failed_event_fires_and_is_recorded():
    events = []

    def boom(sample):
        raise RuntimeError("target exploded")

    target = SimpleTarget("broken", boom)
    report = await Evaluator(make_dataset(), [target], [Length()], on_event=events.append).run_async()

    failures = [e for e in events if isinstance(e, TargetFailed)]
    assert len(failures) == 2
    assert all("target exploded" in e.error for e in failures)
    assert all(r.output.error for r in report.results)


@pytest.mark.asyncio
async def test_metric_failed_event_fires():
    events = []
    target = SimpleTarget("t", lambda s: "ok")
    await Evaluator(make_dataset(), [target], [ExplodingMetric()], on_event=events.append).run_async()

    failures = [e for e in events if isinstance(e, MetricFailed)]
    assert len(failures) == 2
    assert all(e.metric == "exploding" for e in failures)
    assert all("boom" in e.error for e in failures)


@pytest.mark.asyncio
async def test_group_failure_is_warned_and_reported_as_event_without_crashing():
    class FlakyBackend(NativeBackend):
        async def _run_one(self, target, sample, metrics, on_event):
            if sample.id == "2":
                raise RuntimeError("orchestration bug")
            return await super()._run_one(target, sample, metrics, on_event)

    events = []
    target = SimpleTarget("t", lambda s: "ok")
    backend = FlakyBackend()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = await Evaluator(
            make_dataset(), [target], [Length()], backend=backend, on_event=events.append
        ).run_async()

    assert len(report.results) == 1  # sample "1" still made it through
    assert any("orchestration bug" in str(w.message) for w in caught)
    group_failures = [e for e in events if isinstance(e, GroupFailed)]
    assert len(group_failures) == 1
    assert "orchestration bug" in group_failures[0].error


@pytest.mark.asyncio
async def test_ensemble_judge_tolerates_one_member_failing():
    async def broken_complete(prompt: str) -> str:
        raise RuntimeError("provider down")

    working = LLMJudge("working", lambda p: '{"score": 0.8, "rationale": "fine"}')
    broken = LLMJudge("broken", broken_complete)
    ensemble = EnsembleJudge([working, broken], aggregate="mean")

    verdict = await ensemble.judge(input="q", output="a", reference=None, criteria="c")
    assert verdict.score == 0.8  # only the working judge's score counts
    assert "provider down" in verdict.rationale
    assert "working" in verdict.rationale


@pytest.mark.asyncio
async def test_ensemble_judge_all_members_failing_returns_empty_verdict_not_a_crash():
    async def broken_complete(prompt: str) -> str:
        raise RuntimeError("provider down")

    ensemble = EnsembleJudge([LLMJudge("a", broken_complete), LLMJudge("b", broken_complete)])
    verdict = await ensemble.judge(input="q", output="a", reference=None, criteria="c")
    assert verdict.score is None
    assert verdict.rationale

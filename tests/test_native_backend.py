import pytest

from genevals.core.dataset import Dataset
from genevals.core.types import Sample
from genevals.evaluator import Evaluator
from genevals.metrics.operational import Latency
from genevals.metrics.text import ExactMatch, Length
from genevals.targets.chat import ChatTarget
from genevals.targets.simple import SimpleTarget


def make_dataset() -> Dataset:
    return Dataset(
        [
            Sample(id="1", input="capital of France?", reference="Paris", tags={"topic": "geo"}),
            Sample(id="2", input="capital of Japan?", reference="Tokyo", tags={"topic": "geo"}),
        ],
        name="geo",
    )


@pytest.mark.asyncio
async def test_single_target_run():
    target = SimpleTarget("perfect", lambda s: s.reference)
    report = await Evaluator(make_dataset(), [target], [ExactMatch()]).run_async()
    assert len(report.results) == 2
    assert all(r.score("exact_match").passed for r in report.results)
    assert report.summary["per_target"]["perfect"]["exact_match"]["pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_parallel_targets_are_tagged_separately():
    good = SimpleTarget("good", lambda s: s.reference, tags={"model": "good-model"})
    bad = SimpleTarget("bad", lambda s: "???", tags={"model": "bad-model"})
    report = await Evaluator(make_dataset(), [good, bad], [ExactMatch()]).run_async()

    assert len(report.results) == 4
    good_results = [r for r in report.results if r.target == "good"]
    assert all(r.tags["model"] == "good-model" for r in good_results)
    assert all(r.tags["topic"] == "geo" for r in good_results)  # dataset tag carried through
    assert report.summary["per_target"]["bad"]["exact_match"]["pass_rate"] == 0.0


@pytest.mark.asyncio
async def test_failing_target_does_not_crash_the_run():
    def boom(sample: Sample) -> str:
        raise RuntimeError("target exploded")

    target = SimpleTarget("broken", boom)
    report = await Evaluator(make_dataset(), [target], [ExactMatch()]).run_async()
    assert len(report.results) == 2
    assert all(r.output.error for r in report.results)
    assert all(r.scores == [] for r in report.results)


@pytest.mark.asyncio
async def test_metric_directions_are_captured_on_the_report():
    target = SimpleTarget("t", lambda s: s.reference)
    report = await Evaluator(make_dataset(), [target], [ExactMatch(), Latency()]).run_async()
    assert report.metric_directions == {"exact_match": True, "latency_ms": False}


@pytest.mark.asyncio
async def test_chat_target_accumulates_history():
    seen_histories = []

    def echo_last_len(history):
        seen_histories.append(len(history))
        return f"turn {len(history)}"

    dataset = Dataset(
        [
            Sample(id="c1-1", input="hi", conversation_id="c1", turn=0),
            Sample(id="c1-2", input="how are you", conversation_id="c1", turn=1),
        ]
    )
    target = ChatTarget("bot", echo_last_len)
    report = await Evaluator(dataset, [target], [Length()]).run_async()
    assert len(report.results) == 2
    assert seen_histories == [1, 3]

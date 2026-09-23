import pytest

from genevals.core.dataset import Dataset
from genevals.core.types import Sample
from genevals.evaluator import Evaluator
from genevals.metrics.text import ExactMatch
from genevals.significance import compare_targets
from genevals.targets.simple import SimpleTarget


def make_dataset(n: int) -> Dataset:
    return Dataset([Sample(id=str(i), input=f"q{i}", reference="yes") for i in range(n)])


@pytest.mark.asyncio
async def test_identical_targets_are_not_significant():
    dataset = make_dataset(20)
    a = SimpleTarget("a", lambda s: "yes")
    b = SimpleTarget("b", lambda s: "yes")
    report = await Evaluator(dataset, [a, b], [ExactMatch()]).run_async()

    result = report.compare_targets("exact_match", "a", "b", seed=0)
    assert result.mean_diff == 0.0
    assert not result.significant
    assert result.p_value == 1.0


@pytest.mark.asyncio
async def test_obviously_different_targets_are_significant():
    dataset = make_dataset(30)
    good = SimpleTarget("good", lambda s: "yes")  # always matches reference "yes"
    bad = SimpleTarget("bad", lambda s: "no")  # never matches
    report = await Evaluator(dataset, [good, bad], [ExactMatch()]).run_async()

    result = report.compare_targets("exact_match", "good", "bad", seed=0)
    assert result.mean_a == 1.0
    assert result.mean_b == 0.0
    assert result.significant
    assert result.p_value < 0.05
    assert result.ci_low > 0


@pytest.mark.asyncio
async def test_small_noisy_difference_is_not_necessarily_significant():
    # Mirrors what public_dataset_demo.py actually showed: a small mean
    # difference at n=12 is easily within noise.
    dataset = make_dataset(12)
    a_outputs = ["yes"] * 9 + ["no"] * 3  # 75%
    b_outputs = ["yes"] * 10 + ["no"] * 2  # 83%

    def make_target(name, outputs):
        return SimpleTarget(name, lambda s, _o=outputs: _o[int(s.id)])

    report = await Evaluator(
        dataset, [make_target("a", a_outputs), make_target("b", b_outputs)], [ExactMatch()]
    ).run_async()

    result = report.compare_targets("exact_match", "a", "b", seed=0)
    assert result.n == 12
    assert not result.significant  # an 8-point gap at n=12 shouldn't clear significance


def test_raises_when_no_shared_scored_samples():
    from genevals.core.types import EvalReport, MetricResult, Output, SampleResult

    report = EvalReport(
        results=[
            SampleResult(sample_id="1", target="a", input="q", output=Output(text="x"), scores=[]),
            SampleResult(
                sample_id="2",
                target="b",
                input="q",
                output=Output(text="x"),
                scores=[MetricResult(metric="m", value=1.0)],
            ),
        ]
    )
    with pytest.raises(ValueError, match="no samples"):
        compare_targets(report, "m", "a", "b")

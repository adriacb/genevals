import pytest

from genevals.core.dataset import Dataset
from genevals.core.types import Sample
from genevals.evaluator import Evaluator
from genevals.gating import Threshold, ThresholdViolation, check_thresholds
from genevals.metrics.text import ExactMatch
from genevals.targets.simple import SimpleTarget


def make_report():
    dataset = Dataset(
        [
            Sample(id="1", input="q1", reference="a"),
            Sample(id="2", input="q2", reference="a"),
            Sample(id="3", input="q3", reference="b"),
        ]
    )
    target = SimpleTarget("t", lambda s: "a")  # matches 2/3
    return Evaluator(dataset, [target], [ExactMatch()]).run()


def test_check_thresholds_pass_rate():
    report = make_report()
    gate = check_thresholds(report, [Threshold("exact_match", ">=", 0.5)])
    assert gate.passed
    assert gate.results[0].actual == pytest.approx(2 / 3)


def test_check_thresholds_failure_is_reported():
    report = make_report()
    gate = check_thresholds(report, [Threshold("exact_match", ">=", 0.9)])
    assert not gate.passed
    assert len(gate.failures()) == 1
    assert gate.failures()[0].target == "t"


def test_check_thresholds_missing_metric_is_not_silently_passed():
    report = make_report()
    gate = check_thresholds(report, [Threshold("does_not_exist", ">=", 0.0)])
    assert not gate.passed
    assert gate.results[0].actual is None


def test_assert_thresholds_raises_on_failure_via_report_method():
    report = make_report()
    with pytest.raises(ThresholdViolation, match="exact_match"):
        report.assert_thresholds([Threshold("exact_match", ">=", 0.99)])


def test_assert_thresholds_does_not_raise_when_met():
    report = make_report()
    report.assert_thresholds([Threshold("exact_match", ">=", 0.5)])  # no raise


def test_report_check_thresholds_matches_module_function():
    report = make_report()
    assert report.check_thresholds([Threshold("exact_match", ">=", 0.5)]).passed

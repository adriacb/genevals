import pytest

from genevals.executors._pricing import cost_from_usage

PRICING = {"model-a": (1.0, 2.0), "model-a-large": (10.0, 20.0)}


def test_basic_lookup():
    cost = cost_from_usage("model-a", 1_000_000, 500_000, PRICING)
    assert cost == pytest.approx(1.0 + 1.0)


def test_longest_prefix_wins_over_shorter_prefix():
    # "model-a-large" starts with "model-a" too — the longer/more specific
    # entry must win, not whichever happens to be checked first.
    cost = cost_from_usage("model-a-large", 1_000_000, 0, PRICING)
    assert cost == pytest.approx(10.0)


def test_unknown_model_returns_none():
    assert cost_from_usage("totally-unknown-model", 1000, 1000, PRICING) is None


def test_missing_token_counts_return_none():
    assert cost_from_usage("model-a", None, 500, PRICING) is None
    assert cost_from_usage("model-a", 500, None, PRICING) is None

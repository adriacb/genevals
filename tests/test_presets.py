import pytest

from genevals.judges.llm_judge import LLMJudge
from genevals.presets import USE_CASES


def test_lists_expected_use_cases():
    names = set(USE_CASES.names())
    assert {"assistant", "chatbot_tools", "generative_flow", "rag", "supervisor", "summarizer", "classifier"} <= names


def test_describe_shape_is_non_empty():
    entries = USE_CASES.describe()
    assert entries
    for entry in entries:
        assert {"name", "description", "metrics"} <= entry.keys()
        assert entry["metrics"], f"use case '{entry['name']}' has no metrics"


def test_unknown_use_case_raises_friendly_error():
    with pytest.raises(KeyError, match="assistant"):
        USE_CASES.get("not-a-real-use-case")


def test_resolve_judge_based_use_case():
    judge = LLMJudge("stub", lambda p: '{"score": 0.9}')
    use_case = USE_CASES.get("assistant")
    metrics = use_case.metrics(judge=judge)
    assert len(metrics) == len(use_case.spec)


def test_resolve_without_judge_raises():
    with pytest.raises(ValueError, match="needs a judge"):
        USE_CASES.get("assistant").metrics()


def test_resolve_catalog_only_use_case_needs_no_judge():
    metrics = USE_CASES.get("classifier").metrics()
    assert [m.name for m in metrics] == ["exact_match"]


def test_generative_flow_mixes_catalog_and_judge_metrics():
    judge = LLMJudge("stub", lambda p: '{"score": 0.9}')
    metrics = USE_CASES.get("generative_flow").metrics(judge=judge)
    names = {m.name for m in metrics}
    assert "token_f1" in names
    assert "correctness" in names


def test_rag_use_case_without_extra_raises_helpful_error():
    with pytest.raises(ImportError, match="rag"):
        USE_CASES.get("rag").metrics()

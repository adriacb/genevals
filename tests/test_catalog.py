from genevals.metrics.catalog import CATALOG


def test_builtin_metrics_registered():
    names = CATALOG.list()
    assert {"exact_match", "contains", "token_f1", "length"} <= set(names)


def test_search_by_category():
    lexical = CATALOG.search(category="lexical")
    assert {m.name for m in lexical} == {"exact_match", "contains", "token_f1"}


def test_describe_shape():
    entries = CATALOG.describe()
    assert all({"name", "category", "modality", "needs_reference", "description"} <= entry.keys() for entry in entries)


def test_unknown_metric_raises_friendly_error():
    try:
        CATALOG.get("does_not_exist")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "does_not_exist" in str(exc)

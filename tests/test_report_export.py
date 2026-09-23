import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from genevals.core.dataset import Dataset
from genevals.core.types import Sample
from genevals.evaluator import Evaluator
from genevals.metrics.text import ExactMatch
from genevals.targets.simple import SimpleTarget


@pytest.fixture
def report():
    dataset = Dataset([Sample(id="1", input="q", reference="a", tags={"topic": "x"})], name="mini")
    target = SimpleTarget("t", lambda s: s.reference)
    return Evaluator(dataset, [target], [ExactMatch()]).run()


def test_to_json_roundtrip(report, tmp_path: Path):
    path = tmp_path / "report.json"
    report.to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset_name"] == "mini"
    assert len(data["results"]) == 1


def test_to_yaml_roundtrip(report, tmp_path: Path):
    path = tmp_path / "report.yaml"
    report.to_yaml(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["targets"] == ["t"]


def test_to_html_is_self_contained(report, tmp_path: Path):
    path = tmp_path / "report.html"
    report.to_html(path)
    content = path.read_text(encoding="utf-8")
    assert "<html" in content
    assert "cdn" not in content.lower()
    assert '"exact_match"' in content


def test_to_html_javascript_is_syntactically_valid(report, tmp_path: Path):
    """The Python-side assertions above never parse the embedded JS, so a broken
    <script> block (e.g. a stray escaped quote) can pass every other test while
    silently blanking the page in a real browser. Catch that here instead."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    path = tmp_path / "report.html"
    report.to_html(path)
    content = path.read_text(encoding="utf-8")

    match = re.search(r"<script>\n(.*?)\n</script>", content, re.DOTALL)
    assert match, "expected an inline (non-JSON) <script> block"
    js_path = tmp_path / "report.js"
    js_path.write_text(match.group(1), encoding="utf-8")

    result = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

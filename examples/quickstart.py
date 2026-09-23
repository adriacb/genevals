"""Run with: uv run python examples/quickstart.py

Evaluates two toy "flows" in parallel over the same dataset, using a
deterministic metric plus a stubbed LLM-judge, then writes json/yaml/html
reports next to this file.
"""

from __future__ import annotations

from pathlib import Path

from genevals import CATALOG, Dataset, Evaluator, Sample, SimpleTarget
from genevals.events import MetricFailed, RunStarted, TargetFailed
from genevals.judges.llm_judge import LLMJudge
from genevals.judges.metric_adapter import JudgeMetric
from genevals.metrics.text import Contains

HERE = Path(__file__).parent


def naive_flow(sample: Sample) -> str:
    """A deliberately mediocre "pipeline": echoes the question back."""
    return f"I think the answer relates to: {sample.input}"


def good_flow(sample: Sample) -> str:
    """Stand-in for a real model call — here it just returns the reference."""
    return sample.reference or "I don't know."


def fake_llm_complete(prompt: str) -> str:
    """Stand-in for a real provider call (OpenAI/Anthropic/local) — replace me."""
    return '{"score": 0.9, "rationale": "plausible enough for a demo"}'


def main() -> None:
    dataset = Dataset.load(HERE / "sample_dataset.jsonl", name="quickstart")

    targets = [
        SimpleTarget("naive", naive_flow, tags={"flow": "naive-v1"}),
        SimpleTarget("good", good_flow, tags={"flow": "reference-v1"}),
    ]

    judge = LLMJudge("demo-judge", fake_llm_complete)
    metrics = [
        Contains(),
        JudgeMetric("relevance", judge, criteria="Does the response address the question?", threshold=0.5),
    ]

    def on_event(event):
        # Observability, not error handling: failures are always recorded in
        # the report regardless of whether you pass a listener — this is
        # just how you'd notice one while the run is still going.
        if isinstance(event, RunStarted):
            print(f"starting: {event.n_samples} samples x {event.n_targets} targets x {event.n_metrics} metrics")
        elif isinstance(event, (TargetFailed, MetricFailed)):
            print(f"FAILED: {event}")

    report = Evaluator(dataset, targets, metrics, on_event=on_event).run()

    report.to_json(HERE / "report.json")
    report.to_yaml(HERE / "report.yaml")
    report.to_html(HERE / "report.html")

    print(f"metrics available in the catalog: {CATALOG.list()}")
    print(f"wrote {HERE / 'report.html'}")


if __name__ == "__main__":
    main()

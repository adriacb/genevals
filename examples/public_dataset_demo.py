"""A fuller demo against real public benchmark data and the real Anthropic API.

Dataset: examples/public_dataset.jsonl — 6 GSM8K (grade-school math word
problems, MIT-licensed) + 6 TruthfulQA (common-misconception questions,
Apache-2.0) samples, tagged by category/source. See
scripts in the repo history / README for how it was pulled — real published
benchmark rows, not invented ones.

Exercises the parts of genevals a single toy example doesn't reach:
  - two parallel Targets (direct-answer vs. chain-of-thought prompting) —
    "parallel pipelines" compared side by side in one report
  - contains, token_f1, a RegexMatch structural check, Length, Latency, Cost
  - genevals.presets.USE_CASES ("generative_flow") for the judge-scored
    dimensions (correctness, coherence), resolved against an EnsembleJudge
    of two different Claude tiers (a cheap/fast judge + a stronger second
    opinion) instead of a single judge
  - tag-based filtering in the HTML report: filter to category=math to see
    the numeric/structural metrics in isolation, or category=misconceptions
    to see how the judge handles answers that only partially avoid a
    misconception

Not covered here (see anthropic_live_demo.py for exact_match, and the
README for genevals.metrics.rag / genevals.validation): RAG metrics need a
retrieved-context field this dataset doesn't have, and judge calibration
needs a separate labeled gold set.

Run with: uv run --extra anthropic python examples/public_dataset_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from genevals import Dataset, Evaluator, SimpleTarget
from genevals.executors.anthropic import AnthropicExecutor
from genevals.judges.ensemble import EnsembleJudge
from genevals.judges.llm_judge import LLMJudge
from genevals.metrics.text import Contains, RegexMatch
from genevals.presets import USE_CASES

HERE = Path(__file__).parent
FAST_MODEL = "claude-haiku-4-5-20251001"
STRONG_MODEL = "claude-sonnet-5"

COT_SUFFIX = "\n\nShow your reasoning briefly, then give the final answer on its own line."


def _load_env_key(env_path: Path, key: str) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() != key:
            continue
        v = v.strip()
        if v and v[0] in "\"'" and v[-1] == v[0]:
            v = v[1:-1]
        return v
    raise KeyError(f"{key} not found in {env_path}")


def _get_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    return _load_env_key(HERE.parent.parent / "aureon" / ".env", "ANTHROPIC_API_KEY")


def main() -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=_get_api_key())
    direct_executor = AnthropicExecutor(client, model=FAST_MODEL, max_tokens=300)
    cot_executor = AnthropicExecutor(client, model=FAST_MODEL, max_tokens=500)
    fast_judge_executor = AnthropicExecutor(client, model=FAST_MODEL, max_tokens=200)
    strong_judge_executor = AnthropicExecutor(client, model=STRONG_MODEL, max_tokens=200)

    dataset = Dataset.load(HERE / "public_dataset.jsonl", name="public-benchmarks")

    targets = [
        SimpleTarget("direct", lambda sample: direct_executor.complete(sample.input), tags={"prompting": "direct"}),
        SimpleTarget(
            "chain-of-thought",
            lambda sample: cot_executor.complete(sample.input + COT_SUFFIX),
            tags={"prompting": "cot"},
        ),
    ]

    # Two different model tiers as an ensemble judge, instead of trusting one.
    judge = EnsembleJudge(
        [
            LLMJudge("claude-haiku-judge", fast_judge_executor),
            LLMJudge("claude-sonnet-judge", strong_judge_executor),
        ],
        aggregate="mean",
    )

    metrics = [
        Contains(),
        RegexMatch(r"\$\d+(?:\.\d+)?", name="currency_format"),
        *USE_CASES.get("generative_flow").metrics(judge=judge),  # correctness, coherence, token_f1, latency_ms
    ]

    print(f"targets: {[t.name for t in targets]}")
    print(f"metrics: {[m.name for m in metrics]}")

    report = Evaluator(dataset, targets, metrics, concurrency=4).run()

    report.to_html(HERE / "public_dataset_report.html")
    report.to_json(HERE / "public_dataset_report.json")

    print("\nper-target summary:")
    for target, stats in report.summary["per_target"].items():
        print(f"  {target}:")
        for metric, s in stats.items():
            print(f"    {metric}: {s}")

    print(f"\nwrote {HERE / 'public_dataset_report.html'}")


if __name__ == "__main__":
    main()

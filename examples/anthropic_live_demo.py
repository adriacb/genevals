"""Live smoke test against the real Anthropic API.

Reads ANTHROPIC_API_KEY from ../aureon/.env (falls back to the environment
if already set) so the key doesn't need to be duplicated into this repo.
Makes a small number of real, billed API calls (Claude Haiku) to both
generate answers and judge them. This is a manual "does the judges/ wiring
actually work against a real provider" check, not part of the test suite —
no API key is required to run `pytest`.

Run with: uv run --extra anthropic python examples/anthropic_live_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

from genevals import Dataset, Evaluator, SimpleTarget
from genevals.executors.anthropic import AnthropicExecutor
from genevals.judges.llm_judge import LLMJudge
from genevals.judges.metric_adapter import JudgeMetric
from genevals.metrics.text import Contains

HERE = Path(__file__).parent
MODEL = "claude-haiku-4-5-20251001"


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
    aureon_env = HERE.parent.parent / "aureon" / ".env"
    return _load_env_key(aureon_env, "ANTHROPIC_API_KEY")


def main() -> None:
    from anthropic import AsyncAnthropic

    # One shared client, two Executors (same one-method interface) with
    # different token budgets: the target answers, the judge just scores.
    client = AsyncAnthropic(api_key=_get_api_key())
    target_executor = AnthropicExecutor(client, model=MODEL, max_tokens=100)
    judge_executor = AnthropicExecutor(client, model=MODEL, max_tokens=200)

    dataset = Dataset.load(HERE / "sample_dataset.jsonl", name="anthropic-live")
    target = SimpleTarget("claude-haiku", lambda sample: target_executor.complete(sample.input), tags={"model": MODEL})
    judge = LLMJudge("claude-judge", judge_executor)

    metrics = [
        Contains(),
        JudgeMetric(
            "correctness",
            judge,
            criteria="Does the response correctly answer the question, matching the reference answer in meaning?",
            threshold=0.5,
        ),
    ]

    report = Evaluator(dataset, [target], metrics, concurrency=2).run()

    report.to_html(HERE / "anthropic_live_report.html")
    report.to_json(HERE / "anthropic_live_report.json")

    for r in report.results:
        contains = r.score("contains")
        judge_score = r.score("correctness")
        print(f"[{r.sample_id}] output={r.output.text!r}")
        print(
            f"    contains={contains.passed if contains else None}  "
            f"judge_score={judge_score.value if judge_score else None}  "
            f"rationale={judge_score.rationale if judge_score else None}"
        )

    print("\nsummary:", report.summary["per_target"])
    print(f"wrote {HERE / 'anthropic_live_report.html'}")


if __name__ == "__main__":
    main()

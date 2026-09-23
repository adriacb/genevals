"""A vision-language (VLM) demo — today, ahead of the v2 genevals.vlm module.

genevals' v1 scope is text (LLM) evaluation; VLM support (a dedicated module
wrapping VLMEvalKit-style benchmark suites) is roadmapped for v2 — see
README "Roadmap". This script shows that you don't have to wait for that to
evaluate a vision-capable model *today*: `Sample.metadata` is free-form, so
an image reference rides in `metadata["image_path"]`, and `SimpleTarget`
wraps any callable — including one that builds a multimodal Anthropic
request instead of a plain-text one. Nothing here required a package change.

Dataset: examples/vlm_dataset.jsonl, 5 samples against 5 real photos in
examples/vlm_images/ (Wikipedia/Wikimedia Commons, CC BY-SA) — landmark and
breed identification plus one attribute (petal color) question. Each
image/answer pair was inspected before being written down, not guessed.

Run with: uv run --extra anthropic python examples/vlm_demo.py
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from genevals import Dataset, Evaluator, Sample, SimpleTarget
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
    return _load_env_key(HERE.parent.parent / "aureon" / ".env", "ANTHROPIC_API_KEY")


def main() -> None:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=_get_api_key())

    async def vision_answer(sample: Sample) -> str:
        """Not an Executor: Executor.complete(prompt: str) is text-only by
        design (see genevals.executors.base). A VLM target just calls the
        provider directly with an image content block — SimpleTarget doesn't
        care what a target does internally, only that it returns text."""
        image_path = HERE / sample.metadata["image_path"]
        media_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        message = await client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                        {"type": "text", "text": sample.input},
                    ],
                }
            ],
        )
        # Same defensive filtering as AnthropicExecutor.complete: don't assume content[0] is text.
        text_blocks = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        return "".join(text_blocks)

    judge_executor = AnthropicExecutor(client, model=MODEL, max_tokens=150)
    judge = LLMJudge("claude-judge", judge_executor)

    dataset = Dataset.load(HERE / "vlm_dataset.jsonl", name="vlm-demo")
    target = SimpleTarget("claude-vision", vision_answer, tags={"model": MODEL})
    metrics = [
        Contains(),
        JudgeMetric(
            "visual_correctness",
            judge,
            criteria="Does the response correctly identify what is shown in the image, matching the reference?",
            threshold=0.5,
        ),
    ]

    report = Evaluator(dataset, [target], metrics, concurrency=3).run()

    report.to_html(HERE / "vlm_report.html")
    report.to_json(HERE / "vlm_report.json")

    for r in report.results:
        contains = r.score("contains")
        judged = r.score("visual_correctness")
        print(f"[{r.sample_id}] reference={r.reference!r} output={r.output.text!r}")
        print(f"    contains={contains.passed if contains else None}  judge_score={judged.value if judged else None}")

    print("\nsummary:", report.summary["per_target"])
    print(f"wrote {HERE / 'vlm_report.html'}")


if __name__ == "__main__":
    main()

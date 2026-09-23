# genevals

A pluggable evaluation package for generative AI systems: single model calls,
whole RAG/agent pipelines, and multi-turn chatbots — with a metrics catalog,
tag-based filtering, and JSON/YAML/HTML reports. The HTML report is built for
a mixed technical/business audience: KPI-style stat tiles, mean/median/min/max
and a distribution plot per metric (see "Reading the distribution plots"
below — it's not a plain ECDF, and that's deliberate) with a "worst results"
shortlist (click one to jump straight to it in the sample table below),
sortable metric columns for triage, and a click-to-expand row on every
sample showing input / expected output / actual output, why each metric
passed or failed, and — when you're comparing targets — what every other
target produced for that same input, right in the same panel.

v1 targets text (LLM) systems. Vision-language model (VLM) evaluation is
planned for v2 (see "Roadmap").

## Install

```bash
uv add genevals          # or: pip install genevals
uv add "genevals[rag]"      # + ragas-backed RAG metrics
uv add "genevals[inspect]"  # + the inspect_ai backend
uv add "genevals[jev]"      # + the experimental Jev judge backend
uv add "genevals[anthropic]"  # + AnthropicExecutor
uv add "genevals[openai]"     # + OpenAIExecutor (also covers Azure OpenAI, vLLM, Ollama, ... via base_url)
```

## Quickstart

```python
from genevals import Dataset, Evaluator, SimpleTarget
from genevals.metrics.text import ExactMatch

dataset = Dataset.load("examples/sample_dataset.jsonl")

def my_flow(sample):
    return call_my_model(sample.input)  # your RAG pipeline / agent / model call

target = SimpleTarget("my-flow-v1", my_flow, tags={"model": "gpt-5"})
report = Evaluator(dataset, [target], [ExactMatch()]).run()

report.to_html("report.html")   # filterable/sortable, self-contained, no CDN
report.to_json("report.json")
report.to_yaml("report.yaml")
```

## Examples

- `examples/quickstart.py` — two parallel flows, a deterministic metric, a
  stubbed LLM-judge metric (no API key needed).
- `examples/anthropic_live_demo.py` — the same shape against the real
  Anthropic API via `AnthropicExecutor` (needs `ANTHROPIC_API_KEY`).
- `examples/public_dataset_demo.py` — a fuller run against real public
  benchmark rows (`examples/public_dataset.jsonl`: 6 GSM8K math problems +
  6 TruthfulQA misconception questions), comparing direct vs.
  chain-of-thought prompting as two parallel targets, scored by every
  built-in metric type plus an `EnsembleJudge` of two Claude tiers via the
  `generative_flow` use-case preset.
- `examples/vlm_demo.py` — a vision-language demo ahead of v2 (see
  Roadmap): 5 real photos + questions (`examples/vlm_dataset.jsonl` /
  `vlm_images/`), scored with `Contains` and a judge. No package change was
  needed — the image rides in `Sample.metadata`, and the target is just a
  `SimpleTarget` calling the provider with a multimodal request instead of a
  plain-text one.
- `examples/custom_executor_demo.py` — two patterns for a fully custom
  Executor, runnable offline (no API key, no network): wrapping any
  Executor to inject fixed business context into every prompt, and a raw
  stdlib-HTTP Executor with no provider SDK at all.

## Concepts

- **Sample** — one row of a `Dataset`: `input`, optional `reference`, and a
  free-form `tags: dict[str, str]` (e.g. `{"topic": "billing", "difficulty":
  "hard"}`). `Dataset.load`/`.save` read/write `.jsonl`, `.json`, or `.yaml`;
  `Dataset.with_tag(key, value)` filters by tag.
- **Target** — the system under test. `SimpleTarget` wraps any callable (sync
  or async) — a single model call, a full RAG pipeline, or an agent; that's
  your "generative AI flow." `ChatTarget` wraps a callable that receives the
  running conversation history, for evaluating a chatbot across multi-turn
  `Sample`s sharing a `conversation_id`. **Parallel pipelines** aren't a
  separate type — pass several `Target`s to one `Evaluator` and they're
  compared side by side in the same report, each tagged with its own
  `target.tags`.
- **Metric** — a named entry in `genevals.metrics.CATALOG`: a deterministic
  check (`exact_match`, `contains`, `token_f1`, `regex:...`, `length`), a
  wrapped library metric (`genevals.metrics.rag` for ragas), or an LLM-judge
  under the hood (`JudgeMetric`, see below). `CATALOG.describe()` / the
  `genevals catalog` CLI command lists what's registered.
- **Judge** — an LLM-as-judge abstraction (`genevals.judges`). `LLMJudge`
  wraps any completion function (bring your own OpenAI/Anthropic/local call);
  `EnsembleJudge` combines several judges (mean or majority vote) to reduce
  single-judge bias; `JevJudge` (opt-in, `genevals.judges.jev_judge`) is an
  experimental backend for TypeSafe AI's Jev "System One Model" — see the
  caveats in that module's docstring before using it for anything that gates
  a release. Wrap any `Judge` as a catalog `Metric` with `JudgeMetric`.
- **EvalBackend** — how `(Dataset, Targets, Metrics)` actually get run.
  `NativeBackend` (default) is a plain asyncio runner with no extra
  dependencies. `InspectAIBackend` (`genevals.backends.inspect_ai_backend`,
  needs the `inspect` extra) hands the same inputs to UK AISI's `inspect_ai`
  for its sandboxing/logging/prebuilt-eval ecosystem. Implement
  `genevals.backends.base.EvalBackend` to plug in another harness.
- **Executor** (`genevals.executors`) — a one-method interface
  (`complete(prompt) -> str`) around a provider call, so you build one
  object per model and reuse it as both the target under test and the judge
  scoring it, instead of writing two near-identical closures. Use
  `.complete` (or the Executor directly) for a `Judge` — text only; use
  `.generate(prompt) -> Output` for a `Target` — same call, but latency and
  (for `AnthropicExecutor`/`OpenAIExecutor`, which see real token usage)
  `cost_usd` ride along into the report for free:

  ```python
  executor = AnthropicExecutor(client, model="claude-haiku-4-5-20251001")
  judge = LLMJudge("claude-judge", executor)                             # text only
  target = SimpleTarget("claude", lambda s: executor.generate(s.input))  # + cost/latency
  ```

  `AnthropicExecutor` (needs the `anthropic` extra) and `OpenAIExecutor`
  (needs the `openai` extra — also covers Azure OpenAI, vLLM's
  OpenAI-compatible server, Ollama, and anything else speaking the same API
  via `base_url=`) are the two provider ones. `CachedExecutor` wraps any
  Executor to cache identical prompts (in-memory or a JSONL file) — see
  "Reliability" below. For a provider genevals doesn't ship, or per-instance
  extra context (a fixed persona/business rules), `Executor.__init__` isn't
  constrained by the interface at all — see
  `examples/custom_executor_demo.py` for both a wrapper that injects fixed
  context into every prompt and a from-scratch stdlib-only HTTP Executor.
  (For Anthropic/OpenAI specifically, a fixed system prompt is simpler:
  `AnthropicExecutor(client, system="You are Acme Corp's support agent...")`
  — it's forwarded straight through to the API.)
- **genevals.validation** — calibrate a `Judge` against a labeled gold set
  (accuracy / MAE / Pearson r, plus a list of disagreements) before trusting
  it for CI gating. Judges can be internally consistent while still
  disagreeing with human judgment — see the module docstring.
- **genevals.presets** (`USE_CASES`) — a YAML-backed catalog of recommended
  metrics per system archetype: `assistant`, `chatbot_tools`,
  `generative_flow`, `rag`, `supervisor`, `summarizer`, `classifier`. Each
  entry is either a catalog metric (no judge needed) or a judge-scored
  dimension with its own rubric (`criteria`) and threshold:

  ```python
  from genevals import USE_CASES
  from genevals.judges.llm_judge import LLMJudge

  judge = LLMJudge("my-judge", my_completion_fn)
  metrics = USE_CASES.get("chatbot_tools").metrics(judge=judge)
  report = Evaluator(dataset, [target], metrics).run()
  ```

  `USE_CASES.names()` / `.describe()` list what's available; the defaults
  live in `src/genevals/presets/use_cases.yaml` — point
  `UseCaseCatalog(path=...)` at your own copy (same shape) to customize them
  for your org.
- **`genevals.gating`** — assert a report clears a quality bar, for CI:
  `report.assert_thresholds([Threshold("exact_match", ">=", 0.8)])` raises
  `ThresholdViolation` if not (pytest-friendly); `report.check_thresholds(...)`
  returns a `GateResult` instead of raising, if you want to build your own
  reporting on top. The CLI exposes the same thing: `genevals run ... --gate
  "exact_match>=0.8"` exits 1 on failure.
- **`genevals.significance`** — `report.compare_targets(metric, a, b)` runs a
  paired bootstrap test (no scipy needed) and tells you whether target A is
  *actually* different from target B on a metric, or within noise given how
  few samples you ran. Worth checking before reading anything into a small
  mean difference — see "Reliability" below for a real example where it
  mattered.
- **`CachedExecutor`** (`genevals.executors`) — wraps any `Executor`, caching
  identical prompts (in-memory, or persisted to a JSONL file with `path=`) so
  iterating on one metric or target doesn't re-pay for every unchanged
  sample. `cached.hits` / `.misses` tell you how much it's saving.

## Reliability, rate limits, and observability

- **A failing target, metric, or judge never aborts the run.** Each is
  caught at the point of failure and recorded on that one `SampleResult`
  (`output.error`, or a `MetricResult` whose `rationale` explains what
  broke) — every other sample keeps going, and you still get a report. The
  one thing that *can* lose data is a bug in genevals' own orchestration
  (not your target/metric code), which drops just that group of samples and
  surfaces as a `warnings.warn` plus a `GroupFailed` event — the run still
  returns everything else. See `NativeBackend`'s docstring for the details.
- **Rate limiting is a provider/Executor concern, not a sample-orchestration
  one.** `Evaluator(concurrency=N)` bounds how many *(target, sample)* units
  run at once — it does not cap total API calls, since one unit can fan out
  into several judge/ensemble calls. Cap actual request concurrency on the
  Executor itself: `AnthropicExecutor(client, max_concurrency=5)` — share
  one Executor instance across a target and its judge to put them under the
  same limit. The Anthropic SDK already retries 429/5xx/connection errors
  with backoff by default (`max_retries`, default 2); pass `max_retries=`/
  `timeout=` through the Executor (or set them on a `client` you build
  yourself) to tune it.
- **`genevals.events`** — pass `on_event=` to `Evaluator`/`EvalBackend.run()`
  to observe a run live: `RunStarted`, `SampleStarted`/`SampleCompleted`,
  `TargetFailed`, `MetricFailed`, `GroupFailed`, `RunCompleted`. This is for
  noticing a problem *while the run is happening* (a progress bar, a log
  line, an alert) — failures are recorded in the report either way, with or
  without a listener. See `examples/quickstart.py` for a minimal one.
- **Cost tracking is real, not a stub** — `AnthropicExecutor`/`OpenAIExecutor`
  read `response.usage` and price it against a built-in table (a cached
  snapshot of each provider's published rates — pass `pricing=` to override
  or extend it), verified against the live Anthropic API in
  `anthropic_live_demo.py`/`public_dataset_demo.py`/`vlm_demo.py`. Use
  `.generate()` rather than `.complete()` when wiring a Target (see
  "Executor" above) so `cost_usd`/`latency_ms` actually land in `Output`; add
  `Cost()`/`Latency()` (`genevals.metrics.operational`) to your metric list
  to see them in the report. A `CachedExecutor` hit correctly reports
  `cost_usd=0.0` — no new spend happened.

**A real example of why `compare_targets` matters**: `public_dataset_demo.py`
compares direct vs. chain-of-thought prompting and its `correctness` judge
scores looked different run to run (e.g. 0.86 vs. 0.76 in one run, 0.75 vs.
0.75 in another — real LLM non-determinism). It's tempting to read that as
"CoT helps." Running `report.compare_targets("correctness", "direct",
"chain-of-thought")` on that same report says otherwise:

```
direct vs chain-of-thought on correctness (n=12): mean_diff=-0.013
(95% CI [-0.192, +0.196]), p=0.867 -> not significant
```

At n=12 the difference is comfortably within noise — the report alone would
have let that conclusion stand unchallenged.

## Reading the distribution plots

Each numeric metric's card in the HTML report plots a distribution, but
**not** a plain ECDF of the raw values — that reads backwards for a
higher-is-better metric (a well-performing target's plain ECDF necessarily
rises *late*, near the bottom-right) and, worse, ties the chart to the
metric's empirical range in *this* run, which distorts comparisons across
runs and makes narrow-but-uniformly-mediocre performance look spread out.

When you're comparing **two or more targets** on a metric with a known
direction (`Metric.higher_is_better`), the chart instead plots, per sample,
the **gap to the best target on that same sample** — 0 means tied for best,
larger means further behind — and takes the plain ECDF of *that*. This is
the additive analogue of a [Dolan–Moré performance
profile](https://doi.org/10.1007/s101070100263) (2002, the standard
technique in optimization/solver benchmarking for comparing several systems
across many test cases): profiles normally use a *ratio* to the best
solver, but eval metrics are full of exact 0/1 ties (every pass/fail
metric), where ratios blow up — the gap (difference) is the standard fix for
that, same idea as a regret distribution in bandit/RL evaluation. The result
reads the same way regardless of the metric's own bound or direction: **the
curve that rises fastest, up and to the left, is the better target** — no
per-metric axis-flipping or knowing the metric's theoretical ceiling
required, and it stays comparable across separate runs since it's always
relative to what was actually achieved.

With only one target present (or an unscored/directionless metric like
`length`), there's no "best other target" to compare against, so the chart
falls back to the plain distribution of that target's own values — flipped
to a survival curve (`1 - F(x)`) for higher-is-better metrics, so "up" still
means "better" rather than the opposite.

## CLI

```bash
genevals catalog
genevals run --dataset data.jsonl \
  --target mymodule:my_target --target mymodule:other_target \
  --metric exact_match --metric token_f1 \
  --html report.html \
  --gate "exact_match>=0.8" --gate "token_f1>=0.5"
```

`--target` points at a `module:attribute` that resolves to a `Target`
instance (e.g. a `SimpleTarget` you built in `mymodule.py`). `--gate` is
optional and repeatable — the process exits 1 if any threshold fails, for
CI.

## Roadmap

- v2: a real `genevals.vlm` module (wrapping VLMEvalKit) for standard VLM
  benchmark suites (MMBench, MMMU, POPE) and first-class image/video Sample
  fields. `examples/vlm_demo.py` shows the ad-hoc version you can already
  build today with `Sample.metadata` + a custom `Target`.
- More backend adapters as needed (e.g. lm-evaluation-harness) behind the
  same `EvalBackend` protocol.

## Dev

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy src
```

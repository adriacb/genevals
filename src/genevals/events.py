"""Lightweight event hooks: observe a run in progress, not just its final report.

Pass `on_event=` to `Evaluator`/`EvalBackend.run()` — a plain callable (sync
or async) invoked once per Event below. This is about *observability*: a
progress bar, a log line the moment something fails, an alert to Slack, a
live stream to an external dashboard. Failures are always recorded in the
report itself too (`SampleResult.output.error`, or a `MetricResult` whose
`rationale` explains the failure) — that's what makes the run resilient.
Events are what let you *notice* a failure while the run is still going,
instead of only discovering it when you read the report afterwards.

    seen_errors = []
    def on_event(event):
        if isinstance(event, (TargetFailed, MetricFailed)):
            seen_errors.append(event)
            print(f"FAILED {event.target}/{event.sample_id}: {event.error}")

    report = Evaluator(dataset, targets, metrics, on_event=on_event).run()

Only one listener — for more than one, pass a small dispatcher of your own
(`lambda e: [listener(e) for listener in listeners]`).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class RunStarted:
    n_samples: int
    n_targets: int
    n_metrics: int


@dataclass
class SampleStarted:
    sample_id: str
    target: str


@dataclass
class SampleCompleted:
    sample_id: str
    target: str
    had_error: bool


@dataclass
class TargetFailed:
    sample_id: str
    target: str
    error: str


@dataclass
class MetricFailed:
    sample_id: str
    target: str
    metric: str
    error: str


@dataclass
class GroupFailed:
    """The rare case where genevals' own orchestration broke, not target/metric
    code — see NativeBackend's docstring. That group's samples are missing
    from the report entirely."""

    target: str
    error: str


@dataclass
class RunCompleted:
    n_results: int
    n_errors: int


Event = (
    RunStarted | SampleStarted | SampleCompleted | TargetFailed | MetricFailed | GroupFailed | RunCompleted
)
Listener = Callable[[Event], "None | Awaitable[None]"]


async def emit(listener: Listener | None, event: Event) -> None:
    if listener is None:
        return
    result = listener(event)
    if inspect.isawaitable(result):
        await result

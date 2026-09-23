"""The Executor interface: one method, prompt in, text out.

Both Target (via SimpleTarget) and Judge (via LLMJudge) already accept a
plain callable, so any Executor works directly as either — it's callable via
`__call__`, and `.complete` is the same thing spelled out. The point of
naming it is reuse: build one Executor per provider/model and hand it to
both the target under test and the judge scoring it, instead of rewriting
the same API call as two near-identical closures.

    executor = AnthropicExecutor(client, model="claude-haiku-4-5-20251001")
    judge = LLMJudge("claude-judge", executor)                          # text only
    target = SimpleTarget("claude", lambda s: executor.generate(s.input))  # + cost/latency

Use `.complete` (or the Executor directly) for a Judge — it only ever needs
text. Use `.generate` for a Target: it returns a full Output, so latency and
— for providers that report token usage — cost_usd ride along into the
report for free. The base implementation here just times `complete()`;
provider Executors that see real usage data (AnthropicExecutor,
OpenAIExecutor) override it to also fill in cost_usd.
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable

from genevals.core.types import Output


class Executor(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str | Awaitable[str]: ...

    def __call__(self, prompt: str) -> str | Awaitable[str]:
        return self.complete(prompt)

    async def generate(self, prompt: str) -> Output:
        start = time.perf_counter()
        result = self.complete(prompt)
        if inspect.isawaitable(result):
            result = await result
        return Output(text=result, latency_ms=(time.perf_counter() - start) * 1000)

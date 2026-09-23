"""The Executor interface: one method, prompt in, text out.

Both Target (via SimpleTarget) and Judge (via LLMJudge) already accept a
plain callable, so any Executor works directly as either — it's callable via
`__call__`, and `.complete` is the same thing spelled out. The point of
naming it is reuse: build one Executor per provider/model and hand it to
both the target under test and the judge scoring it, instead of rewriting
the same API call as two near-identical closures.

    executor = AnthropicExecutor(client, model="claude-haiku-4-5-20251001")
    target = SimpleTarget("claude", lambda s: executor.complete(s.input))
    judge = LLMJudge("claude-judge", executor)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable


class Executor(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str | Awaitable[str]: ...

    def __call__(self, prompt: str) -> str | Awaitable[str]:
        return self.complete(prompt)

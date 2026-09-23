"""A generic LLM-as-judge backend: bring your own completion function."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable

from genevals.judges.base import Judge, JudgeVerdict

DEFAULT_PROMPT = """You are a careful, impartial evaluator.

INPUT:
{input}

RESPONSE TO JUDGE:
{output}
{reference_block}
CRITERIA:
{criteria}

Score the response from 0.0 (fails the criteria) to 1.0 (fully meets it).
Respond with ONLY minified JSON: {{"score": <float>, "rationale": "<one sentence>"}}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudge(Judge):
    """Wraps any chat-completion callable as a Judge.

    `complete` receives the rendered prompt string and returns (or awaits to)
    the model's raw text reply. This keeps genevals independent of any one
    provider SDK — pass in an OpenAI/Anthropic/litellm/local-model call of
    your own, e.g.:

        judge = LLMJudge("gpt-judge", lambda p: my_openai_client.complete(p))
    """

    def __init__(
        self,
        name: str,
        complete: Callable[[str], Awaitable[str] | str],
        *,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.name = name
        self._complete = complete
        self._prompt_template = prompt_template

    async def judge(self, *, input: str, output: str, reference: str | None, criteria: str) -> JudgeVerdict:
        reference_block = f"\nREFERENCE ANSWER:\n{reference}\n" if reference else ""
        prompt = self._prompt_template.format(
            input=input, output=output, criteria=criteria, reference_block=reference_block
        )
        reply = self._complete(prompt)
        if inspect.isawaitable(reply):
            reply = await reply
        return self._parse(reply)

    @staticmethod
    def _parse(reply: str) -> JudgeVerdict:
        match = _JSON_BLOCK.search(reply)
        if not match:
            return JudgeVerdict(rationale=f"judge reply was not JSON: {reply[:200]!r}", raw=reply)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeVerdict(rationale=f"judge reply had invalid JSON: {reply[:200]!r}", raw=reply)
        score = parsed.get("score")
        return JudgeVerdict(
            score=float(score) if score is not None else None,
            rationale=parsed.get("rationale"),
            raw=reply,
        )

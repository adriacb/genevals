import pytest

from genevals.executors.base import Executor


class EchoExecutor(Executor):
    async def complete(self, prompt: str) -> str:
        return f"echo: {prompt}"


@pytest.mark.asyncio
async def test_executor_call_delegates_to_complete():
    executor = EchoExecutor()
    assert await executor.complete("hi") == "echo: hi"
    assert await executor("hi") == "echo: hi"


@pytest.mark.asyncio
async def test_executor_works_directly_as_a_judge_completion_fn():
    from genevals.judges.llm_judge import LLMJudge

    class JsonExecutor(Executor):
        async def complete(self, prompt: str) -> str:
            return '{"score": 0.7, "rationale": "stub"}'

    judge = LLMJudge("stub", JsonExecutor())
    verdict = await judge.judge(input="q", output="a", reference=None, criteria="c")
    assert verdict.score == 0.7

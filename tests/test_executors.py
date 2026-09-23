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


@pytest.mark.asyncio
async def test_default_generate_wraps_complete_with_latency_no_cost():
    executor = EchoExecutor()
    output = await executor.generate("hi")
    assert output.text == "echo: hi"
    assert output.latency_ms is not None and output.latency_ms >= 0
    assert output.cost_usd is None  # base class has no usage data to price

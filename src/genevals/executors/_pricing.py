"""Shared cost-from-usage math for provider Executors.

Each provider module owns its own price table (prices differ entirely
per-provider); this just does the longest-prefix-match lookup and the
arithmetic, so AnthropicExecutor/OpenAIExecutor don't duplicate it.

Prices are cached snapshots of what each provider publishes, not a live
feed — they drift. Treat `cost_usd` as good for comparing targets/runs
against each other, not as an invoice. Every provider Executor accepts a
`pricing=` override for exactly this reason.
"""

from __future__ import annotations


def cost_from_usage(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: dict[str, tuple[float, float]],
) -> float | None:
    """pricing maps a model-name prefix -> (input $/1M tokens, output $/1M tokens)."""
    if input_tokens is None or output_tokens is None:
        return None
    for name in sorted(pricing, key=len, reverse=True):
        if model.startswith(name):
            price_in, price_out = pricing[name]
            return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return None

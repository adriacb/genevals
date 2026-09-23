from genevals.executors.base import Executor

__all__ = ["Executor"]

# Provider executors are not imported here by default: they need optional
# extras. Import them explicitly, e.g.:
#   from genevals.executors.anthropic import AnthropicExecutor

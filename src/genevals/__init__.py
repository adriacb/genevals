"""genevals: a pluggable evaluation package for generative AI systems."""

from genevals.core.dataset import Dataset
from genevals.core.types import EvalReport, MetricResult, Output, Sample, SampleResult
from genevals.evaluator import Evaluator
from genevals.metrics.catalog import CATALOG
from genevals.presets import USE_CASES
from genevals.targets.chat import ChatTarget
from genevals.targets.simple import SimpleTarget

__all__ = [
    "CATALOG",
    "USE_CASES",
    "ChatTarget",
    "Dataset",
    "EvalReport",
    "Evaluator",
    "MetricResult",
    "Output",
    "Sample",
    "SampleResult",
    "SimpleTarget",
]

__version__ = "0.1.0"

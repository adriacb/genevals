"""JSON and YAML exporters for an EvalReport."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from genevals.core.types import EvalReport


def report_to_json(report: EvalReport, path: str | Path | None = None, *, indent: int = 2) -> str:
    text = report.model_dump_json(indent=indent)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def report_to_yaml(report: EvalReport, path: str | Path | None = None) -> str:
    text = yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text

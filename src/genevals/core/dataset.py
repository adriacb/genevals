"""Dataset loading, saving, and tag-based filtering."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import yaml

from genevals.core.types import Sample


class Dataset:
    """An ordered collection of Samples, loadable from / savable to jsonl, json, or yaml."""

    def __init__(self, samples: Iterable[Sample] | None = None, *, name: str | None = None):
        self.samples: list[Sample] = list(samples or [])
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def filter(self, predicate: Callable[[Sample], bool]) -> Dataset:
        return Dataset((s for s in self.samples if predicate(s)), name=self.name)

    def with_tag(self, key: str, value: str | None = None) -> Dataset:
        """Return the subset of samples that carry `key` (and, if given, equal `value`)."""
        if value is None:
            return self.filter(lambda s: key in s.tags)
        return self.filter(lambda s: s.tags.get(key) == value)

    def tag_values(self, key: str) -> set[str]:
        return {s.tags[key] for s in self.samples if key in s.tags}

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        elif path.suffix == ".jsonl":
            rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        elif path.suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix} (use .jsonl, .json, or .yaml)")

        samples = [Sample(**row) if "id" in row else Sample(id=str(i), **row) for i, row in enumerate(rows)]
        return cls(samples, name=name or path.stem)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        rows = [s.model_dump(mode="json") for s in self.samples]
        if path.suffix in {".yaml", ".yml"}:
            path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")
        elif path.suffix == ".jsonl":
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        elif path.suffix == ".json":
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix} (use .jsonl, .json, or .yaml)")

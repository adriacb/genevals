"""A tiny generic name -> object registry used by the metrics catalog."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-keyed collection with duplicate protection and a friendly KeyError."""

    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T) -> T:
        if key in self._items:
            raise ValueError(f"{self._kind} '{key}' is already registered")
        self._items[key] = item
        return item

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"Unknown {self._kind} '{key}'. Available: {sorted(self._items)}") from None

    def list(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

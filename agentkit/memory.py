"""Bounded short-term memory for agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryItem:
    role: str
    content: Any


class ConversationMemory:
    """Ring-buffer memory that keeps the most recent N items."""

    def __init__(self, capacity: int = 20) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def add(self, role: str, content: Any) -> None:
        self._items.append(MemoryItem(role, content))
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]

    def recent(self, n: int | None = None) -> list[MemoryItem]:
        if n is None:
            return list(self._items)
        return self._items[-n:]

    def as_messages(self) -> list[dict[str, Any]]:
        return [{"role": item.role, "content": item.content} for item in self._items]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

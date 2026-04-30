"""Tiny LRU cache with explicit key eviction.

Used for bounded in-memory caches where stdlib's ``functools.lru_cache``
isn't a fit (no per-key invalidation, no in-place size tuning).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Fixed-capacity LRU cache. Not thread-safe; suitable for the MCP server's
    single-threaded request loop.
    """

    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._data.pop(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

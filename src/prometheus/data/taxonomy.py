"""N-level taxonomy data structures (JSON-serializable, no pickle)."""

from __future__ import annotations

import json
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class LevelSpec(BaseModel):
    """Specification for a single taxonomy level."""

    model_config = ConfigDict(ignored_types=(cached_property,))

    depth: int
    name: str
    classes: list[str]
    class_to_idx: dict[str, int]
    parent_dims: list[int]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @cached_property
    def idx_to_class(self) -> dict[int, str]:
        return {v: k for k, v in self.class_to_idx.items()}

    def encode(self, label: str) -> int:
        return self.class_to_idx[label]

    def decode(self, idx: int) -> str:
        return self.classes[idx]


class ParentChildEntry(BaseModel):
    """One row of the parent→valid_children mapping (JSON form)."""

    depth: int
    parent_idx: int
    children: list[int]


class TaxonomyInfo(BaseModel):
    """Complete taxonomy definition for N levels.

    ``parent_child_map`` is a tuple-keyed dict in memory (``(depth, parent_idx) → children``)
    and serializes to a JSON-friendly list of records.
    """

    name: str
    levels: list[LevelSpec]
    text_column: str
    embedding_model: str
    total_samples: int = 0
    parent_child_map: dict[tuple[int, int], list[int]] | None = None

    @field_validator("parent_child_map", mode="before")
    @classmethod
    def _accept_list_of_records(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, list):
            return {(item["depth"], item["parent_idx"]): item["children"] for item in v}
        return v

    @field_serializer("parent_child_map")
    def _serialize_pcm(
        self,
        v: dict[tuple[int, int], list[int]] | None,
    ) -> list[dict] | None:
        if v is None:
            return None
        return [
            {"depth": d, "parent_idx": p, "children": list(c)} for (d, p), c in sorted(v.items())
        ]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def level_names(self) -> list[str]:
        return [level.name for level in self.levels]

    @property
    def num_classes_per_level(self) -> list[int]:
        return [level.num_classes for level in self.levels]

    @property
    def leaf_level(self) -> LevelSpec:
        return self.levels[-1]

    def get_level(self, depth: int) -> LevelSpec:
        return self.levels[depth]

    def get_level_by_name(self, name: str) -> LevelSpec | None:
        for level in self.levels:
            if level.name == name:
                return level
        return None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)

    @classmethod
    def from_json(cls, json_str: str) -> TaxonomyInfo:
        return cls.model_validate(json.loads(json_str))

    @classmethod
    def build(
        cls,
        name: str,
        text_column: str,
        level_columns: list[str],
        level_classes: list[list[str]],
        embedding_model: str,
        total_samples: int = 0,
    ) -> TaxonomyInfo:
        """Build a TaxonomyInfo from raw column data."""
        levels = []
        for depth, (col_name, classes) in enumerate(zip(level_columns, level_classes)):
            sorted_classes = sorted(set(classes))
            class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
            parent_dims = [len(levels[d].classes) for d in range(depth)]
            levels.append(
                LevelSpec(
                    depth=depth,
                    name=col_name,
                    classes=sorted_classes,
                    class_to_idx=class_to_idx,
                    parent_dims=parent_dims,
                )
            )
        return cls(
            name=name,
            levels=levels,
            text_column=text_column,
            embedding_model=embedding_model,
            total_samples=total_samples,
        )

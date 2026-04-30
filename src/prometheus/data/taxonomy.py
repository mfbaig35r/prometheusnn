"""N-level taxonomy data structures (JSON-serializable, no pickle)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property


@dataclass
class LevelSpec:
    """Specification for a single taxonomy level."""

    depth: int  # 0-indexed (0 = root)
    name: str  # User-friendly name from CSV column
    classes: list[str]  # Unique class labels at this level
    class_to_idx: dict[str, int]  # Label -> integer mapping
    parent_dims: list[int]  # [num_classes_L0, ...] for parent context

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

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "name": self.name,
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "parent_dims": self.parent_dims,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LevelSpec:
        return cls(
            depth=data["depth"],
            name=data["name"],
            classes=data["classes"],
            class_to_idx=data["class_to_idx"],
            parent_dims=data["parent_dims"],
        )


@dataclass
class TaxonomyInfo:
    """Complete taxonomy definition for N levels."""

    name: str
    levels: list[LevelSpec]
    text_column: str
    embedding_model: str
    total_samples: int = 0
    parent_child_map: dict[str, list[int]] | None = None

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

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "levels": [level.to_dict() for level in self.levels],
            "text_column": self.text_column,
            "embedding_model": self.embedding_model,
            "total_samples": self.total_samples,
        }
        if self.parent_child_map is not None:
            d["parent_child_map"] = self.parent_child_map
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TaxonomyInfo:
        return cls(
            name=data["name"],
            levels=[LevelSpec.from_dict(level) for level in data["levels"]],
            text_column=data["text_column"],
            embedding_model=data["embedding_model"],
            total_samples=data.get("total_samples", 0),
            parent_child_map=data.get("parent_child_map"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> TaxonomyInfo:
        return cls.from_dict(json.loads(json_str))

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

"""Tests for taxonomy data structures."""

import json

from prometheus.data.taxonomy import LevelSpec, TaxonomyInfo


def test_level_spec_basics():
    spec = LevelSpec(
        depth=0,
        name="Category",
        classes=["A", "B", "C"],
        class_to_idx={"A": 0, "B": 1, "C": 2},
        parent_dims=[],
    )
    assert spec.num_classes == 3
    assert spec.encode("B") == 1
    assert spec.decode(2) == "C"
    assert spec.idx_to_class == {0: "A", 1: "B", 2: "C"}


def test_level_spec_roundtrip():
    spec = LevelSpec(
        depth=1,
        name="Sub",
        classes=["X", "Y"],
        class_to_idx={"X": 0, "Y": 1},
        parent_dims=[3],
    )
    restored = LevelSpec.model_validate(spec.model_dump())
    assert restored.depth == spec.depth
    assert restored.name == spec.name
    assert restored.classes == spec.classes
    assert restored.class_to_idx == spec.class_to_idx
    assert restored.parent_dims == spec.parent_dims


def test_taxonomy_info_properties(three_level_taxonomy):
    t = three_level_taxonomy
    assert t.num_levels == 3
    assert t.level_names == ["Category", "SubCategory", "Type"]
    assert t.num_classes_per_level == [2, 3, 4]
    assert t.leaf_level.name == "Type"
    assert t.get_level(0).name == "Category"
    assert t.get_level_by_name("SubCategory").depth == 1
    assert t.get_level_by_name("Nonexistent") is None


def test_taxonomy_json_roundtrip(three_level_taxonomy):
    t = three_level_taxonomy
    json_str = t.to_json()
    parsed = json.loads(json_str)
    assert parsed["name"] == "test_model"

    restored = TaxonomyInfo.from_json(json_str)
    assert restored.num_levels == t.num_levels
    assert restored.level_names == t.level_names
    assert restored.num_classes_per_level == t.num_classes_per_level
    assert restored.text_column == t.text_column
    assert restored.embedding_model == t.embedding_model


def test_taxonomy_build():
    t = TaxonomyInfo.build(
        name="built",
        text_column="desc",
        level_columns=["A", "B"],
        level_classes=[
            ["cat", "dog", "cat"],
            ["small", "large", "small"],
        ],
        embedding_model="test",
        total_samples=3,
    )
    assert t.num_levels == 2
    assert t.levels[0].classes == ["cat", "dog"]
    assert t.levels[1].classes == ["large", "small"]
    assert t.levels[1].parent_dims == [2]
    assert t.total_samples == 3

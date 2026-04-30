"""Tests for the intake module: profiler, detector, recommender, scoper."""

from __future__ import annotations

import pandas as pd

from prometheus.intake import (
    detect_problem_type,
    estimate_training_time,
    profile_columns,
    recommend_next_steps,
    scope_problem_from_text,
    surface_concerns,
)
from prometheus.intake.detector import detect_hierarchical_relationships
from prometheus.intake.models import ColumnKind, ProblemType
from prometheus.intake.profiler import classify_column_kind

# ─── Profiler ───


def test_profiler_classifies_text_column():
    df = pd.DataFrame({
        "description": [f"a longer description for item {i} with details" for i in range(50)]
    })
    profiles = profile_columns(df)
    assert profiles[0].kind == ColumnKind.TEXT
    assert profiles[0].cardinality == 50
    assert profiles[0].avg_text_length is not None and profiles[0].avg_text_length > 30


def test_profiler_classifies_categorical():
    df = pd.DataFrame({"color": ["red", "blue", "green"] * 50})
    profile = profile_columns(df)[0]
    assert profile.kind == ColumnKind.CATEGORICAL
    assert profile.cardinality == 3


def test_profiler_classifies_numeric():
    df = pd.DataFrame({"price": [i * 1.5 for i in range(100)]})
    profile = profile_columns(df)[0]
    assert profile.kind == ColumnKind.NUMERIC


def test_profiler_classifies_identifier_by_uniqueness():
    df = pd.DataFrame({"sku": [f"S{i:05d}" for i in range(100)]})
    profile = profile_columns(df)[0]
    assert profile.kind == ColumnKind.IDENTIFIER


def test_profiler_long_unique_strings_stay_text():
    """Regression: unique long descriptions should not be classified as identifier."""
    df = pd.DataFrame({
        "desc": [f"This is a unique product description for item number {i}" for i in range(100)]
    })
    profile = profile_columns(df)[0]
    assert profile.kind == ColumnKind.TEXT


def test_profiler_handles_nulls():
    df = pd.DataFrame({"col": ["a", "b", None, None, "c"]})
    profile = profile_columns(df)[0]
    assert profile.null_rate == 0.4


def test_profiler_dedupes_sample_values():
    """Sorted CSVs repeat the same value many times — samples should be unique."""
    df = pd.DataFrame({"category": ["A"] * 50 + ["B"] * 50 + ["C"] * 50})
    profile = profile_columns(df)[0]
    assert profile.sample_values == ["A", "B", "C"]
    assert len(profile.sample_values) == len(set(profile.sample_values))


def test_classify_column_kind_id_by_name():
    series = pd.Series([f"abc-{i}" for i in range(50)], name="user_id")
    assert classify_column_kind(series, name="user_id") == ColumnKind.IDENTIFIER


# ─── Detector ───


def test_detect_hierarchical_relationships():
    df = pd.DataFrame({
        "country": ["US", "US", "US", "FR", "FR"],
        "city": ["NYC", "LA", "SF", "Paris", "Lyon"],
    })
    rels = detect_hierarchical_relationships(df, ["country", "city"])
    assert ("country", "city") in rels
    assert ("city", "country") not in rels


def test_detect_no_hierarchy_when_independent():
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "blue"] * 5,
        "size": ["S", "M", "L", "S"] * 5,
    })
    rels = detect_hierarchical_relationships(df, ["color", "size"])
    assert rels == []


def test_detector_picks_hierarchical():
    df = pd.DataFrame({
        "text": [f"product description {i} with extra detail" for i in range(60)],
        "category": (["A"] * 20 + ["B"] * 20 + ["C"] * 20),
        "subcategory": ["A1", "A2"] * 10 + ["B1", "B2"] * 10 + ["C1", "C2"] * 10,
    })
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    assert a.problem_type == ProblemType.HIERARCHICAL_CLASSIFICATION
    assert a.suggested_text_column == "text"
    assert a.suggested_level_columns == ["category", "subcategory"]


def test_detector_picks_flat():
    df = pd.DataFrame({
        "text": [f"description for item {i} with details" for i in range(60)],
        "category": ["A", "B", "C"] * 20,
    })
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    assert a.problem_type == ProblemType.FLAT_CLASSIFICATION
    assert a.suggested_level_columns == ["category"]


def test_detector_not_a_fit_no_text():
    df = pd.DataFrame({"a": range(100), "b": [i * 2 for i in range(100)]})
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    assert a.problem_type == ProblemType.NOT_A_FIT


def test_detector_not_a_fit_no_labels():
    df = pd.DataFrame({"text": [f"description {i} with words" for i in range(50)]})
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    assert a.problem_type == ProblemType.NOT_A_FIT


# ─── Recommender ───


def test_recommender_returns_two_steps_for_deep_hierarchy():
    df = pd.DataFrame({
        "text": [f"description for item {i} with details" for i in range(60)],
        "L1": ["A"] * 30 + ["B"] * 30,
        "L2": ["A1", "A2"] * 15 + ["B1", "B2"] * 15,
        "L3": ["A1a", "A1b", "A2a", "A2b"] * 7 + ["B1a", "B1b"] * 8 + ["B2a", "B2b"] * 8,
    })
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    steps = recommend_next_steps("data.csv", a, profiles)
    assert len(steps) == 2
    assert "drop" in steps[1].label.lower()
    assert steps[0].args["level_columns"] != steps[1].args["level_columns"]


def test_recommender_single_step_for_flat():
    df = pd.DataFrame({
        "text": [f"description for item {i} with details" for i in range(60)],
        "category": ["A", "B", "C"] * 20,
    })
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    steps = recommend_next_steps("data.csv", a, profiles)
    assert len(steps) == 1


def test_concerns_flag_low_samples_per_class():
    # 30 rows across 20 distinct categories → ~1.5 samples per class, well below threshold
    categories = [f"category_{i}" for i in range(20)] + [f"category_{i}" for i in range(10)]
    df = pd.DataFrame({
        "text": [f"item description {i} with details" for i in range(30)],
        "label": categories,
    })
    profiles = profile_columns(df)
    a = detect_problem_type(df, profiles)
    concerns = surface_concerns(df, a, profiles)
    assert any(c.severity == "critical" for c in concerns)


def test_estimate_training_time_scales_with_rows():
    small_low, small_high = estimate_training_time(rows=500, num_levels=2)
    large_low, large_high = estimate_training_time(rows=50000, num_levels=2)
    assert large_low > small_low
    assert large_high > small_high


def test_estimate_training_time_mpnet_slower_than_minilm():
    minilm_low, _ = estimate_training_time(rows=10000, num_levels=2, embedding_model="all-MiniLM-L6-v2")
    mpnet_low, _ = estimate_training_time(rows=10000, num_levels=2, embedding_model="all-mpnet-base-v2")
    assert mpnet_low > minilm_low


# ─── Scoper ───


def test_scoper_classification_with_count():
    r = scope_problem_from_text("classify customer tickets into 8 urgency categories")
    assert r.fits_prometheus is True
    assert r.detected_problem_type == ProblemType.FLAT_CLASSIFICATION
    assert r.what_youll_need is not None


def test_scoper_hierarchical():
    r = scope_problem_from_text("classify products with 3 nested levels of categories")
    assert r.fits_prometheus is True
    assert r.detected_problem_type == ProblemType.HIERARCHICAL_CLASSIFICATION


def test_scoper_does_not_treat_ordinal_levels_as_hierarchical():
    """Regression: '8 urgency levels' is a flat ordinal classifier, not a hierarchy."""
    r = scope_problem_from_text("classify support tickets into 8 urgency levels")
    assert r.fits_prometheus is True
    assert r.detected_problem_type == ProblemType.FLAT_CLASSIFICATION


def test_scoper_hierarchical_via_pattern():
    """'3 levels of categories' should match the pattern even without keyword."""
    r = scope_problem_from_text("I have 4 levels of product categories to classify")
    assert r.fits_prometheus is True
    assert r.detected_problem_type == ProblemType.HIERARCHICAL_CLASSIFICATION


def test_scoper_regression_redirects():
    r = scope_problem_from_text("I want to predict the price of houses")
    assert r.fits_prometheus is False
    assert r.redirect_suggestion is not None
    assert "regression" in r.reasoning.lower()


def test_scoper_clustering_redirects():
    r = scope_problem_from_text("I have customer data and want to find natural groups")
    assert r.fits_prometheus is False
    assert r.redirect_suggestion is not None


def test_scoper_no_labels_redirects():
    r = scope_problem_from_text("classify emails — no labels yet")
    assert r.fits_prometheus is False
    assert "label" in r.reasoning.lower()


def test_scoper_warns_when_too_few_samples():
    r = scope_problem_from_text(
        "classify products into 100 categories",
        data_hint="I have ~200 examples",
    )
    assert r.fits_prometheus is True
    assert any("minimum" in w.lower() or "underfit" in w.lower() for w in r.watch_out_for)


def test_scoper_vague_request_returns_no_fit():
    r = scope_problem_from_text("help me with my data")
    assert r.fits_prometheus is False
    assert "couldn't tell" in r.reasoning.lower() or "classification" in r.reasoning.lower()

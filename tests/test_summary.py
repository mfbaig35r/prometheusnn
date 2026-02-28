"""Tests for inference/summary.py — LLM-friendly prediction summary builder."""

from prometheus.inference.summary import (
    MAX_CODES,
    _build_verdict,
    _code_prefix_for_partial_path,
    _fallback_confidence_summary,
    _note_bottleneck,
    _note_confidence_gap,
    _note_convergent_leaves,
    _note_heading_strength,
    _note_minority_leaf_confidence,
    _note_root_uncertainty,
    _path_labels,
    build_prediction_summary,
    lookup_codes,
)

# ─── Fixtures ───


def _make_predictions(labels, confidences, level_names, margins=None, top_candidates=None):
    """Helper to build a predictions list matching server.py format."""
    preds = []
    for i, (label, conf, name) in enumerate(zip(labels, confidences, level_names)):
        p = {
            "depth": i,
            "level": name,
            "predicted": label,
            "confidence": conf,
            "margin": (margins[i] if margins else 0.5),
            "top_candidates": (
                top_candidates[i] if top_candidates else [{"label": label, "confidence": conf}]
            ),
        }
        preds.append(p)
    return preds


def _make_routing(
    decision="HUMAN_REVIEW", confident_depth=0, confident_path=None, review_reason=None
):
    return {
        "decision": decision,
        "confident_depth": confident_depth,
        "confident_path": confident_path or [],
        "review_reason": review_reason,
    }


def _make_structured_candidate(
    rank, labels, level_names, confidences, path_score, codes_by_level=None
):
    """Build a candidate with structured path for testing verdict/notes."""
    path = []
    for i, (label, name, conf) in enumerate(zip(labels, level_names, confidences)):
        entry = {"level": name, "label": label, "confidence": conf}
        if codes_by_level and i in codes_by_level:
            entry.update(codes_by_level[i])
        path.append(entry)
    return {"rank": rank, "path": path, "path_score": path_score}


LEVEL_NAMES_3 = ["Chapter", "Heading", "Subheading"]


# ─── Test: single candidate (no forks, no code mapping) ───


def test_single_candidate_no_forks():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    routing = _make_routing("AUTO_ACCEPT", 2, ["Rubber", "Conveyor belts", "Other"])

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=routing,
        path_score=-1.5,
        level_names=LEVEL_NAMES_3,
    )

    assert len(summary["candidates"]) == 1
    c = summary["candidates"][0]
    assert c["rank"] == 1
    # Structured path
    assert len(c["path"]) == 3
    assert c["path"][0]["level"] == "Chapter"
    assert c["path"][0]["label"] == "Rubber"
    assert c["path"][0]["confidence"] == 0.85
    assert c["path"][2]["label"] == "Other"
    assert c["path"][2]["confidence"] == 0.78
    assert c["path_score"] == -1.5
    # No code mapping — no code_prefix or codes
    assert "code_prefix" not in c["path"][0]
    assert "codes" not in c["path"][2]
    assert summary["routing"] == "AUTO_ACCEPT"
    assert "verdict" in summary


# ─── Test: fork at depth 0 (empty confident prefix) ───


def test_fork_at_depth_0():
    predictions = _make_predictions(
        ["Animals", "Horses", "Purebred"],
        [0.40, 0.95, 0.88],
        LEVEL_NAMES_3,
    )
    forked = [
        {
            "suffix": ["Animals", "Horses", "Purebred"],
            "suffix_indices": [0, 1, 2],
            "score": -1.2,
            "per_level_probs": [0.40, 0.95, 0.88],
        },
        {
            "suffix": ["Plants", "Trees", "Oak"],
            "suffix_indices": [5, 3, 1],
            "score": -2.5,
            "per_level_probs": [0.35, 0.80, 0.65],
        },
    ]
    routing = _make_routing("HUMAN_REVIEW", None, [])

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=forked,
        routing=routing,
        path_score=-1.2,
        level_names=LEVEL_NAMES_3,
    )

    assert len(summary["candidates"]) == 2
    c0_labels = _path_labels(summary["candidates"][0])
    c1_labels = _path_labels(summary["candidates"][1])
    assert c0_labels == ["Animals", "Horses", "Purebred"]
    assert c1_labels == ["Plants", "Trees", "Oak"]
    # Depth 0 fork: all confidence comes from suffix
    assert summary["candidates"][1]["path"][0]["confidence"] == 0.35
    assert summary["candidates"][1]["path"][2]["confidence"] == 0.65
    assert summary["routing"] == "HUMAN_REVIEW"


# ─── Test: fork at depth > 0 (confident prefix exists) ───


def test_fork_at_depth_1():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.90, 0.60, 0.31],
        LEVEL_NAMES_3,
    )
    forked = [
        {
            "suffix": ["Conveyor belts", "Other"],
            "suffix_indices": [2, 3],
            "score": -2.1,
            "per_level_probs": [0.60, 0.31],
        },
        {
            "suffix": ["Transmission belts", "V-belts"],
            "suffix_indices": [4, 1],
            "score": -2.4,
            "per_level_probs": [0.55, 0.65],
        },
    ]
    routing = _make_routing("HUMAN_REVIEW", 0, ["Rubber"], "Heading confidence below threshold")

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=forked,
        routing=routing,
        path_score=-2.1,
        level_names=LEVEL_NAMES_3,
    )

    assert len(summary["candidates"]) == 2
    c0 = summary["candidates"][0]
    assert _path_labels(c0) == ["Rubber", "Conveyor belts", "Other"]
    assert c0["path"][0]["confidence"] == 0.90  # from predictions prefix
    assert c0["path"][1]["confidence"] == 0.60  # from suffix
    assert c0["path"][2]["confidence"] == 0.31

    c1 = summary["candidates"][1]
    assert _path_labels(c1) == ["Rubber", "Transmission belts", "V-belts"]
    assert c1["path"][0]["confidence"] == 0.90
    assert c1["path"][2]["confidence"] == 0.65


# ─── Test: with code mapping ───


def test_with_code_mapping():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    code_mapping = {
        "code_column": "hts8",
        "level_columns": LEVEL_NAMES_3,
        "mapping": {
            "Rubber|Conveyor belts|Other": ["40103900"],
        },
    }

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("AUTO_ACCEPT"),
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
        code_mapping=code_mapping,
    )

    leaf = summary["candidates"][0]["path"][2]
    assert leaf["codes"] == ["40103900"]
    assert "total_codes" not in leaf


def test_code_prefix_on_non_leaf():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    code_mapping = {
        "code_column": "hts8",
        "level_columns": LEVEL_NAMES_3,
        "mapping": {
            "Rubber|Conveyor belts|Other": ["40103910", "40103950"],
            "Rubber|Conveyor belts|Reinforced": ["40104100"],
            "Rubber|Tires|Pneumatic": ["40210000"],
        },
    }

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("AUTO_ACCEPT"),
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
        code_mapping=code_mapping,
    )

    c = summary["candidates"][0]
    # Chapter "Rubber" → all codes start with "40"
    assert c["path"][0]["code_prefix"] == "40"
    # Heading "Conveyor belts" → codes start with "4010"
    assert c["path"][1]["code_prefix"] == "4010"
    # Leaf gets full codes
    assert c["path"][2]["codes"] == ["40103910", "40103950"]


def test_code_mapping_path_not_found():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Unknown"],
        [0.85, 0.92, 0.30],
        LEVEL_NAMES_3,
    )
    code_mapping = {
        "code_column": "hts8",
        "level_columns": LEVEL_NAMES_3,
        "mapping": {
            "Rubber|Conveyor belts|Other": ["40103900"],
        },
    }

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("HUMAN_REVIEW"),
        path_score=-3.0,
        level_names=LEVEL_NAMES_3,
        code_mapping=code_mapping,
    )

    # Leaf path not in mapping — no codes
    leaf = summary["candidates"][0]["path"][2]
    assert "codes" not in leaf
    # But non-leaf prefix should still work (partial path matches)
    assert summary["candidates"][0]["path"][0].get("code_prefix") == "40103900"
    assert summary["candidates"][0]["path"][1].get("code_prefix") == "40103900"


# ─── Test: code capping ───


def test_codes_capped_at_max():
    predictions = _make_predictions(
        ["Rubber", "Belts", "Other"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    many_codes = [f"CODE{i:04d}" for i in range(20)]
    code_mapping = {
        "code_column": "hts8",
        "level_columns": LEVEL_NAMES_3,
        "mapping": {
            "Rubber|Belts|Other": many_codes,
        },
    }

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("AUTO_ACCEPT"),
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
        code_mapping=code_mapping,
    )

    leaf = summary["candidates"][0]["path"][2]
    assert len(leaf["codes"]) == MAX_CODES
    assert leaf["codes"] == many_codes[:MAX_CODES]
    assert leaf["total_codes"] == 20


def test_codes_under_cap_no_truncation():
    predictions = _make_predictions(
        ["Rubber", "Belts", "Seals"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    code_mapping = {
        "code_column": "hts8",
        "level_columns": LEVEL_NAMES_3,
        "mapping": {
            "Rubber|Belts|Seals": ["40169310", "40169350"],
        },
    }

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("AUTO_ACCEPT"),
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
        code_mapping=code_mapping,
    )

    leaf = summary["candidates"][0]["path"][2]
    assert leaf["codes"] == ["40169310", "40169350"]
    assert "total_codes" not in leaf


# ─── Test: cap at 5 candidates ───


def test_cap_at_5_candidates():
    predictions = _make_predictions(
        ["A", "B", "C"],
        [0.30, 0.50, 0.40],
        LEVEL_NAMES_3,
    )
    forked = [
        {
            "suffix": [f"X{i}", f"Y{i}", f"Z{i}"],
            "suffix_indices": [i, i, i],
            "score": -1.0 - i * 0.5,
            "per_level_probs": [0.30, 0.50, 0.40],
        }
        for i in range(7)
    ]
    routing = _make_routing("HUMAN_REVIEW", None, [])

    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=forked,
        routing=routing,
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
    )

    assert len(summary["candidates"]) == 5


# ─── Test: lookup_codes ───


def test_lookup_codes_none_mapping():
    assert lookup_codes(["A", "B"], None) is None


def test_lookup_codes_found():
    cm = {"mapping": {"A|B": ["001", "002"]}}
    assert lookup_codes(["A", "B"], cm) == ["001", "002"]


def test_lookup_codes_not_found():
    cm = {"mapping": {"A|B": ["001"]}}
    assert lookup_codes(["A", "C"], cm) is None


# ─── Test: code_prefix_for_partial_path ───


def test_code_prefix_single_level():
    cm = {
        "mapping": {
            "Rubber|Belts|Seals": ["40169310", "40169350"],
            "Rubber|Tires|Pneumatic": ["40210000"],
        }
    }
    assert _code_prefix_for_partial_path(["Rubber"], cm) == "40"


def test_code_prefix_two_levels():
    cm = {
        "mapping": {
            "Rubber|Belts|Seals": ["40169310", "40169350"],
            "Rubber|Belts|Other": ["40160000"],
        }
    }
    assert _code_prefix_for_partial_path(["Rubber", "Belts"], cm) == "4016"


def test_code_prefix_no_match():
    cm = {"mapping": {"A|B|C": ["001"]}}
    assert _code_prefix_for_partial_path(["X"], cm) is None


# ─── Test: individual note generators ───


def test_note_root_uncertainty_detected():
    predictions = _make_predictions(
        ["Rubber", "Belts", "Other"],
        [0.35, 0.80, 0.70],
        LEVEL_NAMES_3,
        top_candidates=[
            [
                {"label": "Rubber", "confidence": 0.35},
                {"label": "Plastics", "confidence": 0.30},
                {"label": "Metal", "confidence": 0.20},
            ],
            [{"label": "Belts", "confidence": 0.80}],
            [{"label": "Other", "confidence": 0.70}],
        ],
    )
    note = _note_root_uncertainty(predictions, [], None)
    assert note is not None
    assert "Chapter-level uncertainty" in note
    assert "Rubber" in note
    assert "Plastics" in note


def test_note_root_uncertainty_not_detected():
    predictions = _make_predictions(
        ["Rubber", "Belts", "Other"],
        [0.90, 0.80, 0.70],
        LEVEL_NAMES_3,
        top_candidates=[
            [
                {"label": "Rubber", "confidence": 0.90},
                {"label": "Plastics", "confidence": 0.10},
            ],
            [{"label": "Belts", "confidence": 0.80}],
            [{"label": "Other", "confidence": 0.70}],
        ],
    )
    note = _note_root_uncertainty(predictions, [], None)
    assert note is None


def test_note_heading_strength_across_forks():
    predictions = _make_predictions(
        ["A", "B", "C"],
        [0.35, 0.95, 0.50],
        LEVEL_NAMES_3,
    )
    forked = [
        {
            "suffix": ["A", "B", "C"],
            "suffix_indices": [0, 1, 2],
            "score": -1.0,
            "per_level_probs": [0.35, 0.95, 0.50],
        },
        {
            "suffix": ["D", "E", "F"],
            "suffix_indices": [3, 4, 5],
            "score": -2.0,
            "per_level_probs": [0.30, 0.92, 0.40],
        },
    ]
    note = _note_heading_strength(predictions, [], forked)
    assert note is not None
    assert ">90%" in note


def test_note_minority_leaf_confidence():
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.90, 0.80, 0.31], -1.0),
        _make_structured_candidate(2, ["A", "D", "E"], LEVEL_NAMES_3, [0.90, 0.75, 0.65], -1.5),
    ]
    predictions = _make_predictions(["A", "B", "C"], [0.90, 0.80, 0.31], LEVEL_NAMES_3)
    note = _note_minority_leaf_confidence(predictions, candidates, None)
    assert note is not None
    assert "Candidate 2" in note
    assert "65%" in note


def test_note_convergent_leaves():
    candidates = [
        _make_structured_candidate(1, ["A", "B", "Same"], LEVEL_NAMES_3, [0.90, 0.80, 0.70], -1.0),
        _make_structured_candidate(2, ["C", "D", "Same"], LEVEL_NAMES_3, [0.80, 0.70, 0.60], -1.5),
    ]
    note = _note_convergent_leaves([], candidates, None)
    assert note is not None
    assert "Same" in note


def test_note_confidence_gap_dominant():
    candidates = [
        _make_structured_candidate(1, ["A"], ["L0"], [0.90], -1.0),
        _make_structured_candidate(2, ["B"], ["L0"], [0.10], -4.0),
    ]
    note = _note_confidence_gap([], candidates, None)
    assert note is not None
    assert "dominant" in note


def test_note_confidence_gap_competitive():
    candidates = [
        _make_structured_candidate(1, ["A"], ["L0"], [0.50], -1.0),
        _make_structured_candidate(2, ["B"], ["L0"], [0.45], -1.3),
    ]
    note = _note_confidence_gap([], candidates, None)
    assert note is not None
    assert "competitive" in note


def test_note_confidence_gap_neither():
    candidates = [
        _make_structured_candidate(1, ["A"], ["L0"], [0.70], -1.0),
        _make_structured_candidate(2, ["B"], ["L0"], [0.30], -2.0),
    ]
    note = _note_confidence_gap([], candidates, None)
    assert note is None


# ─── Test: verdict ───


def test_verdict_auto_accept():
    predictions = _make_predictions(["A", "B", "C"], [0.95, 0.90, 0.85], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.95, 0.90, 0.85], -0.5)
    ]
    routing = _make_routing("AUTO_ACCEPT", 2)
    verdict = _build_verdict(predictions, candidates, routing)
    assert "reliable" in verdict


def test_verdict_out_of_taxonomy_strong_root():
    predictions = _make_predictions(["A", "B", "C"], [0.86, 0.52, 0.51], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.86, 0.52, 0.51], -1.5),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.08, 0.50, 0.97], -3.2),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict(predictions, candidates, routing)
    assert "candidate 1" in verdict.lower()
    assert "86%" in verdict


def test_verdict_out_of_taxonomy_no_strong_root():
    """Root 0.30 vs runner 0.25 → gap 0.05 < 0.15 → 'split' verdict."""
    predictions = _make_predictions(["A", "B", "C"], [0.30, 0.52, 0.51], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.30, 0.52, 0.51], -2.5),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.25, 0.50, 0.97], -3.2),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict(predictions, candidates, routing)
    assert "split" in verdict.lower()
    assert "no clear winner" in verdict.lower()


def test_verdict_human_review_dominant_candidate():
    predictions = _make_predictions(["A", "B", "C"], [0.85, 0.60, 0.31], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.85, 0.60, 0.31], -1.5),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.08, 0.50, 0.97], -5.0),
    ]
    routing = _make_routing("HUMAN_REVIEW")
    verdict = _build_verdict(predictions, candidates, routing)
    assert "clear best" in verdict.lower()


def test_verdict_human_review_same_root_diverge():
    predictions = _make_predictions(["A", "B", "C"], [0.90, 0.60, 0.31], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.90, 0.60, 0.31], -2.1),
        _make_structured_candidate(2, ["A", "D", "E"], LEVEL_NAMES_3, [0.90, 0.55, 0.65], -2.4),
    ]
    routing = _make_routing("HUMAN_REVIEW", 0, ["A"])
    verdict = _build_verdict(predictions, candidates, routing)
    assert "share root" in verdict.lower()
    assert "Heading" in verdict


def test_verdict_human_review_different_roots_close():
    predictions = _make_predictions(["A", "B", "C"], [0.40, 0.60, 0.31], LEVEL_NAMES_3)
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.40, 0.60, 0.31], -2.1),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.35, 0.55, 0.65], -2.4),
    ]
    routing = _make_routing("HUMAN_REVIEW", None, [])
    verdict = _build_verdict(predictions, candidates, routing)
    assert "review both" in verdict.lower()


# ─── Test: min_confidence ───


def test_min_confidence_single_candidate():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.85, 0.92, 0.78],
        LEVEL_NAMES_3,
    )
    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=_make_routing("AUTO_ACCEPT"),
        path_score=-1.0,
        level_names=LEVEL_NAMES_3,
    )
    c = summary["candidates"][0]
    assert c["min_confidence"] == 0.78


def test_min_confidence_forked_candidates():
    predictions = _make_predictions(
        ["Rubber", "Conveyor belts", "Other"],
        [0.90, 0.60, 0.31],
        LEVEL_NAMES_3,
    )
    forked = [
        {
            "suffix": ["Conveyor belts", "Other"],
            "suffix_indices": [2, 3],
            "score": -2.1,
            "per_level_probs": [0.60, 0.31],
        },
        {
            "suffix": ["Transmission belts", "V-belts"],
            "suffix_indices": [4, 1],
            "score": -2.4,
            "per_level_probs": [0.55, 0.65],
        },
    ]
    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=forked,
        routing=_make_routing("HUMAN_REVIEW"),
        path_score=-2.1,
        level_names=LEVEL_NAMES_3,
    )
    # Candidate 1: [0.90, 0.60, 0.31] → min is 0.31
    assert summary["candidates"][0]["min_confidence"] == 0.31
    # Candidate 2: [0.90, 0.55, 0.65] → min is 0.55
    assert summary["candidates"][1]["min_confidence"] == 0.55


# ─── Test: _note_root_uncertainty with code_mapping ───


def test_note_root_uncertainty_with_codes():
    code_mapping = {
        "mapping": {
            "Rubber|X|Y": ["40001000"],
            "Rubber|X|Z": ["40991000"],
            "Plastics|X|Y": ["39001000"],
            "Plastics|X|Z": ["39991000"],
        }
    }
    predictions = _make_predictions(
        ["Rubber", "Belts", "Other"],
        [0.35, 0.80, 0.70],
        LEVEL_NAMES_3,
        top_candidates=[
            [
                {"label": "Rubber", "confidence": 0.35},
                {"label": "Plastics", "confidence": 0.30},
            ],
            [{"label": "Belts", "confidence": 0.80}],
            [{"label": "Other", "confidence": 0.70}],
        ],
    )
    note = _note_root_uncertainty(predictions, [], None, code_mapping=code_mapping)
    assert note is not None
    assert "40: Rubber" in note
    assert "39: Plastics" in note


# ─── Test: _note_bottleneck ───


def test_note_bottleneck_fires():
    """IndustryGroup at 14% with strong neighbors → bottleneck note."""
    candidates = [
        _make_structured_candidate(
            1,
            ["A", "B", "C", "D"],
            ["Sector", "Subsector", "IndustryGroup", "Industry"],
            [0.85, 1.00, 0.14, 1.00],
            -2.0,
        ),
    ]
    note = _note_bottleneck([], candidates, None)
    assert note is not None
    assert "bottleneck" in note.lower()
    assert "IndustryGroup" in note
    assert "14%" in note


def test_note_bottleneck_does_not_fire():
    """All levels above 50% → no bottleneck."""
    candidates = [
        _make_structured_candidate(
            1,
            ["A", "B", "C"],
            LEVEL_NAMES_3,
            [0.85, 0.75, 0.60],
            -1.0,
        ),
    ]
    note = _note_bottleneck([], candidates, None)
    assert note is None


# ─── Test: fallback notes ───


def test_fallback_all_confident():
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.90, 0.85, 0.80], -0.5),
    ]
    note = _fallback_confidence_summary(candidates)
    assert note is not None
    assert ">70%" in note


def test_fallback_decreasing_pattern():
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.90, 0.70, 0.55], -1.0),
    ]
    note = _fallback_confidence_summary(candidates)
    assert note is not None
    assert "decreases" in note
    assert "90%" in note
    assert "55%" in note


def test_fallback_increasing_pattern():
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.32, 0.60, 0.96], -2.0),
    ]
    note = _fallback_confidence_summary(candidates)
    assert note is not None
    assert "increases" in note
    assert "32%" in note
    assert "96%" in note


def test_fallback_average():
    """Non-monotonic, not all >70% → average."""
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.60, 0.80, 0.55], -1.5),
    ]
    note = _fallback_confidence_summary(candidates)
    assert note is not None
    assert "Average" in note
    assert "65%" in note


def test_notes_never_empty():
    """Even when no detector fires, fallback ensures non-empty notes."""
    predictions = _make_predictions(
        ["A", "B", "C"],
        [0.90, 0.85, 0.80],
        LEVEL_NAMES_3,
    )
    routing = _make_routing("AUTO_ACCEPT")
    summary = build_prediction_summary(
        predictions=predictions,
        forked_completions=None,
        routing=routing,
        path_score=-0.5,
        level_names=LEVEL_NAMES_3,
    )
    assert len(summary["classification_notes"]) >= 1


# ─── Test: OOT verdict tiers ───


def test_verdict_oot_moderate_root():
    """Root 55% → moderate tier."""
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.55, 0.40, 0.30], -2.0),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.20, 0.50, 0.60], -3.5),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict([], candidates, routing)
    assert "moderate" in verdict.lower()
    assert "55%" in verdict


def test_verdict_oot_weak_root_close_runner():
    """Root 30%, runner 25% → gap 5pp < 15pp → 'split'."""
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.30, 0.50, 0.40], -2.5),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.25, 0.45, 0.35], -3.0),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict([], candidates, routing)
    assert "split" in verdict.lower()


def test_verdict_oot_weak_root_large_gap():
    """Root 30%, runner 10% → gap 20pp >= 15pp → 'no candidate clearly reliable'."""
    candidates = [
        _make_structured_candidate(1, ["A", "B", "C"], LEVEL_NAMES_3, [0.30, 0.50, 0.40], -2.5),
        _make_structured_candidate(2, ["D", "E", "F"], LEVEL_NAMES_3, [0.10, 0.45, 0.35], -4.0),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict([], candidates, routing)
    assert "no candidate" in verdict.lower()


# ─── Test: verdict with codes ───


def test_verdict_dominant_candidate_with_codes():
    """When candidate has code_prefix, verdict should include it."""
    candidates = [
        _make_structured_candidate(
            1,
            ["Rubber", "Belts", "Seals"],
            LEVEL_NAMES_3,
            [0.85, 0.60, 0.31],
            -1.5,
            codes_by_level={0: {"code_prefix": "40"}},
        ),
        _make_structured_candidate(
            2,
            ["Chemicals", "Acids", "Sulfuric"],
            LEVEL_NAMES_3,
            [0.08, 0.50, 0.97],
            -5.0,
            codes_by_level={0: {"code_prefix": "38"}},
        ),
    ]
    routing = _make_routing("HUMAN_REVIEW")
    verdict = _build_verdict([], candidates, routing)
    assert "40: Rubber" in verdict


def test_verdict_oot_strong_root_with_codes():
    """OOT strong root with codes in label."""
    candidates = [
        _make_structured_candidate(
            1,
            ["Rubber", "Belts", "Seals"],
            LEVEL_NAMES_3,
            [0.75, 0.60, 0.50],
            -1.5,
            codes_by_level={0: {"code_prefix": "40"}},
        ),
        _make_structured_candidate(
            2,
            ["Chemicals", "Acids", "X"],
            LEVEL_NAMES_3,
            [0.10, 0.40, 0.30],
            -4.0,
            codes_by_level={0: {"code_prefix": "38"}},
        ),
    ]
    routing = _make_routing("OUT_OF_TAXONOMY")
    verdict = _build_verdict([], candidates, routing)
    assert "40: Rubber" in verdict
    assert "strong root" in verdict.lower()


def test_verdict_different_roots_close_with_codes():
    """Root-level split verdict includes codes."""
    candidates = [
        _make_structured_candidate(
            1,
            ["Rubber", "Belts", "Seals"],
            LEVEL_NAMES_3,
            [0.40, 0.60, 0.31],
            -2.1,
            codes_by_level={0: {"code_prefix": "40"}},
        ),
        _make_structured_candidate(
            2,
            ["Chemicals", "Acids", "Sulfuric"],
            LEVEL_NAMES_3,
            [0.35, 0.55, 0.65],
            -2.4,
            codes_by_level={0: {"code_prefix": "38"}},
        ),
    ]
    routing = _make_routing("HUMAN_REVIEW", None, [])
    verdict = _build_verdict([], candidates, routing)
    assert "40: Rubber" in verdict
    assert "38: Chemicals" in verdict

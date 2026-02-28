"""Tests for beam search and greedy classifiers."""

import numpy as np
import torch

from prometheus.data.taxonomy import LevelSpec, TaxonomyInfo
from prometheus.inference.beam_search import (
    BeamPath,
    BeamSearchClassifier,
    GreedyClassifier,
)
from prometheus.training.cascade import HierarchicalClassifier


def _make_model_and_taxonomy():
    taxonomy = TaxonomyInfo(
        name="test",
        levels=[
            LevelSpec(
                depth=0,
                name="L0",
                classes=["A", "B"],
                class_to_idx={"A": 0, "B": 1},
                parent_dims=[],
            ),
            LevelSpec(
                depth=1,
                name="L1",
                classes=["X", "Y", "Z"],
                class_to_idx={"X": 0, "Y": 1, "Z": 2},
                parent_dims=[2],
            ),
        ],
        text_column="text",
        embedding_model="test",
    )

    model = HierarchicalClassifier(
        embedding_dim=16,
        num_classes_per_level=[2, 3],
        hidden_dims=[8],
        dropout=0.0,
    )
    return model, taxonomy


def test_beam_search_produces_valid_path():
    model, taxonomy = _make_model_and_taxonomy()
    clf = BeamSearchClassifier(model, taxonomy, beam_width=2)

    emb = np.random.randn(16).astype(np.float32)
    result = clf.predict(emb)

    assert len(result.path) == 2
    assert result.path[0] in ["A", "B"]
    assert result.path[1] in ["X", "Y", "Z"]
    assert len(result.level_predictions) == 2
    assert result.path_score < 0  # log probabilities are negative


def test_greedy_produces_valid_path():
    model, taxonomy = _make_model_and_taxonomy()
    clf = GreedyClassifier(model, taxonomy)

    emb = np.random.randn(16).astype(np.float32)
    result = clf.predict(emb)

    assert len(result.path) == 2
    assert result.path[0] in ["A", "B"]
    assert result.path[1] in ["X", "Y", "Z"]
    assert len(result.alternative_paths) == 0  # greedy has no alternatives


def test_beam_search_batch():
    model, taxonomy = _make_model_and_taxonomy()
    clf = BeamSearchClassifier(model, taxonomy, beam_width=2)

    embs = np.random.randn(5, 16).astype(np.float32)
    results = clf.predict_batch(embs)

    assert len(results) == 5
    for r in results:
        assert len(r.path) == 2


# --- Diversity-aware beam pruning tests ---


def test_prune_with_diversity_guarantees_minority():
    """Minority group at diversity depth gets at least 1 slot."""
    model, taxonomy = _make_model_and_taxonomy()
    clf = BeamSearchClassifier(model, taxonomy, beam_width=3)
    clf._diversity_depth = 0  # diversity anchor at depth 0

    # 5 beams: 4 from group 0, 1 from group 1 (minority)
    beams = [
        BeamPath(indices=[0, 0], labels=["A", "X"], probs=[0.8, 0.9], log_score=-0.1),
        BeamPath(indices=[0, 1], labels=["A", "Y"], probs=[0.8, 0.7], log_score=-0.5),
        BeamPath(indices=[0, 2], labels=["A", "Z"], probs=[0.8, 0.5], log_score=-0.9),
        BeamPath(indices=[1, 0], labels=["B", "X"], probs=[0.2, 0.8], log_score=-1.0),
        BeamPath(indices=[0, 1], labels=["A", "Y"], probs=[0.8, 0.3], log_score=-1.5),
    ]

    pruned = clf._prune_with_diversity(beams, width=3)

    assert len(pruned) == 3
    # Minority group (indices[0]==1) must survive
    depth0_classes = {b.indices[0] for b in pruned}
    assert 1 in depth0_classes, "Minority branch (B) must survive pruning"
    assert 0 in depth0_classes, "Majority branch (A) must survive pruning"


def test_prune_with_diversity_no_op_without_diversity_depth():
    """Without diversity_depth, falls back to pure score pruning."""
    model, taxonomy = _make_model_and_taxonomy()
    clf = BeamSearchClassifier(model, taxonomy, beam_width=3)
    clf._diversity_depth = None  # no diversity

    # Pre-sorted by log_score descending (as _expand_beams would provide)
    beams = [
        BeamPath(indices=[0, 0], labels=["A", "X"], probs=[0.9, 0.9], log_score=-0.1),
        BeamPath(indices=[0, 1], labels=["A", "Y"], probs=[0.9, 0.7], log_score=-0.5),
        BeamPath(indices=[0, 2], labels=["A", "Z"], probs=[0.9, 0.3], log_score=-1.0),
        BeamPath(indices=[1, 0], labels=["B", "X"], probs=[0.1, 0.8], log_score=-2.0),
    ]

    pruned = clf._prune_with_diversity(beams, width=3)

    assert len(pruned) == 3
    # Pure score order: top 3 by log_score
    assert pruned[0].log_score == -0.1
    assert pruned[1].log_score == -0.5
    assert pruned[2].log_score == -1.0
    # Minority branch (B, score=-2.0) should NOT survive — no diversity guarantee
    depth0_classes = {b.indices[0] for b in pruned}
    assert 1 not in depth0_classes


def _make_3level_model_and_taxonomy():
    """3-level taxonomy: 2 classes at L0, 3 at L1, 4 at L2."""
    taxonomy = TaxonomyInfo(
        name="test3",
        levels=[
            LevelSpec(
                depth=0,
                name="L0",
                classes=["A", "B"],
                class_to_idx={"A": 0, "B": 1},
                parent_dims=[],
            ),
            LevelSpec(
                depth=1,
                name="L1",
                classes=["X", "Y", "Z"],
                class_to_idx={"X": 0, "Y": 1, "Z": 2},
                parent_dims=[2],
            ),
            LevelSpec(
                depth=2,
                name="L2",
                classes=["P", "Q", "R", "S"],
                class_to_idx={"P": 0, "Q": 1, "R": 2, "S": 3},
                parent_dims=[2, 3],
            ),
        ],
        text_column="text",
        embedding_model="test",
    )
    model = HierarchicalClassifier(
        embedding_dim=16,
        num_classes_per_level=[2, 3, 4],
        hidden_dims=[8],
        dropout=0.0,
    )
    return model, taxonomy


def test_diversity_preserved_across_levels():
    """Integration: with L1 uncertain, both L0 branches appear in final results."""
    model, taxonomy = _make_3level_model_and_taxonomy()

    # Thresholds set so L1 (depth=1) is always "uncertain" (threshold=1.0)
    # This forces widening at depth 1 and sets diversity_depth=1
    clf = BeamSearchClassifier(
        model,
        taxonomy,
        beam_width=3,
        confidence_thresholds={1: 1.0},  # impossible threshold → always widens
        adaptive_factor=2,
    )

    # Run many times with different random embeddings to ensure the mechanism works
    # (With widening at L1, we get 6 beams per parent, diversity should keep both L0 branches)
    torch.manual_seed(42)
    np.random.seed(42)

    emb = np.random.randn(16).astype(np.float32)
    result = clf.predict(emb)

    assert len(result.path) == 3
    assert result.path[0] in ["A", "B"]
    assert result.path[1] in ["X", "Y", "Z"]
    assert result.path[2] in ["P", "Q", "R", "S"]

    # Verify diversity_depth was set
    assert clf._diversity_depth == 1

    # Check that alternative paths exist
    all_paths = [result] + [
        type("FakePred", (), {"path": bp.labels, "path_indices": bp.indices})()
        for bp in result.alternative_paths
    ]
    # With diversity pruning, we should have paths from different L1 classes
    l1_classes = {p.path[1] for p in all_paths}
    assert len(l1_classes) >= 2, f"Expected diversity at L1 but only found classes: {l1_classes}"

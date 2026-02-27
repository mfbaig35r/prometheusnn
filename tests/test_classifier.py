"""Tests for LevelClassifier."""

import torch
from prometheus.training.classifier import LevelClassifier


def test_level_classifier_no_parent():
    clf = LevelClassifier(embedding_dim=32, num_classes=5)
    emb = torch.randn(4, 32)
    logits = clf(emb)
    assert logits.shape == (4, 5)


def test_level_classifier_with_parent():
    clf = LevelClassifier(embedding_dim=32, num_classes=5, parent_dims=[3, 4])
    emb = torch.randn(4, 32)
    parent = torch.randn(4, 7)  # 3 + 4
    logits = clf(emb, parent)
    assert logits.shape == (4, 5)


def test_predict_proba_sums_to_one():
    clf = LevelClassifier(embedding_dim=16, num_classes=3)
    emb = torch.randn(2, 16)
    probs = clf.predict_proba(emb)
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=0)


def test_temperature_scaling():
    clf = LevelClassifier(embedding_dim=16, num_classes=3)
    emb = torch.randn(1, 16)

    logits_before = clf(emb).detach()
    clf.set_temperature(2.0)
    logits_after = clf(emb).detach()

    # Higher temperature should reduce logit magnitudes
    assert logits_after.abs().mean() < logits_before.abs().mean()

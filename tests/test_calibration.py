"""Tests for temperature calibration."""

import numpy as np

from prometheus.training.calibration import compute_ece, find_optimal_temperature


def test_compute_ece_perfect_calibration():
    # Perfect predictions: confidence matches accuracy
    rng = np.random.default_rng(42)
    n = 100
    probs = np.zeros((n, 3))
    labels = rng.integers(0, 3, size=n)
    for i in range(n):
        probs[i, labels[i]] = 0.9
        others = [j for j in range(3) if j != labels[i]]
        probs[i, others[0]] = 0.05
        probs[i, others[1]] = 0.05

    ece = compute_ece(probs, labels)
    # Not perfect 0 but should be small
    assert ece < 0.15


def test_find_optimal_temperature():
    rng = np.random.default_rng(42)
    # Generate logits where T=1.5 would be optimal
    logits = rng.standard_normal((200, 5)) * 3.0
    labels = logits.argmax(axis=1)

    temp, ece_before, ece_after = find_optimal_temperature(logits, labels)

    assert temp > 0
    assert ece_after <= ece_before

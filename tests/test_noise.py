"""Tests for noise injection."""

import numpy as np
import torch
from prometheus.training.noise import ParentNoiseInjector


def test_build_confusion_profile():
    injector = ParentNoiseInjector(noise_rate=0.2)
    # 3-class confusion matrix
    cm = np.array([
        [8, 1, 1],
        [2, 7, 1],
        [0, 1, 9],
    ])
    profile = injector.build_confusion_profile(0, cm)

    assert 0 in profile  # class 0 confused with 1 and 2
    assert profile[0] == {1: 1, 2: 1}
    assert profile[1] == {0: 2, 2: 1}


def test_noisy_context_no_noise_when_not_training():
    injector = ParentNoiseInjector(noise_rate=0.5)
    cm = np.array([[5, 5], [5, 5]])
    injector.build_confusion_profile(0, cm)

    batch = {0: np.array([0, 1, 0, 1])}
    ctx = injector.build_noisy_context(
        batch, parent_depths=[0], num_classes_per_level=[2],
        device=torch.device("cpu"), training=False
    )

    # Should be exact one-hot of true labels
    expected = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    torch.testing.assert_close(ctx, expected)


def test_serialization_roundtrip():
    injector = ParentNoiseInjector(noise_rate=0.15)
    cm = np.array([[8, 2], [1, 9]])
    injector.build_confusion_profile(0, cm)

    serialized = injector.get_serializable_profiles()
    assert "0" in serialized  # depth as string key

    injector2 = ParentNoiseInjector(noise_rate=0.15)
    injector2.load_profiles(serialized)
    assert 0 in injector2.confusion_profiles

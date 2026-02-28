"""Tests for data splits and batch iteration."""

import numpy as np

from prometheus.data.splits import BatchIterator


def test_datasplit_basics(small_datasplit):
    ds = small_datasplit
    assert len(ds) == 20
    assert ds.num_levels == 3
    assert ds.get_labels(0).shape == (20,)
    assert ds.leaf_labels.shape == (20,)


def test_batch_iterator_covers_all_data(small_datasplit):
    it = BatchIterator(small_datasplit, batch_size=8, shuffle=False)
    assert len(it) == 3  # ceil(20/8)

    seen = 0
    for batch in it:
        assert "embeddings" in batch
        assert 0 in batch
        assert 1 in batch
        assert 2 in batch
        seen += len(batch["embeddings"])
    assert seen == 20


def test_batch_iterator_shuffle_deterministic(small_datasplit):
    it1 = BatchIterator(small_datasplit, batch_size=8, shuffle=True, random_state=42)
    it2 = BatchIterator(small_datasplit, batch_size=8, shuffle=True, random_state=42)

    for b1, b2 in zip(it1, it2):
        np.testing.assert_array_equal(b1["embeddings"], b2["embeddings"])


def test_batch_iterator_no_shuffle(small_datasplit):
    it = BatchIterator(small_datasplit, batch_size=5, shuffle=False)
    batches = list(it)
    assert len(batches) == 4
    np.testing.assert_array_equal(batches[0]["embeddings"], small_datasplit.embeddings[:5])

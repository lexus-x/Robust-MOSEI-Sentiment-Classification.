from __future__ import annotations

import numpy as np

from real_sentiment.corruptions import burst_noise, contiguous_drop, progressive_drift, temporal_shift


def test_contiguous_drop_only_hits_valid_tokens():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    mask = np.array([0, 0, 1, 1, 1], dtype=bool)
    out = contiguous_drop(x, mask, fraction=0.34, rng=np.random.default_rng(0))

    assert np.array_equal(out[:2], x[:2])
    assert (out[2:] == 0.0).any()


def test_temporal_shift_preserves_left_padding():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([0, 0, 1, 1, 1], dtype=bool)
    out = temporal_shift(x, mask, shift=1)

    assert np.array_equal(out[:2], x[:2])
    assert np.array_equal(out[2], np.zeros(2, dtype=np.float32))
    assert np.array_equal(out[3], np.array([1.0, 1.0], dtype=np.float32))


def test_progressive_drift_changes_late_valid_tokens_more():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    mask = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    out = progressive_drift(x, mask, max_shift=2)

    assert np.array_equal(out[0], x[0])
    assert not np.array_equal(out[-1], x[-1])


def test_burst_noise_keeps_shape():
    x = np.ones((6, 3), dtype=np.float32)
    mask = np.array([0, 1, 1, 1, 1, 1], dtype=bool)
    out = burst_noise(x, mask, span_fraction=0.4, scale=0.5, rng=np.random.default_rng(1))

    assert out.shape == x.shape
    assert np.array_equal(out[0], x[0])
    assert not np.array_equal(out[1:], x[1:])

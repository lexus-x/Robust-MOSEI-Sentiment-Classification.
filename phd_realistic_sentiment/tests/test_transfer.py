from __future__ import annotations

import numpy as np

from real_sentiment.transfer import temporal_split_indices


def test_temporal_split_indices_correct_sizes():
    split = temporal_split_indices(100, train_fraction=0.70)
    assert len(split["train"]) == 70
    assert len(split["test"]) == 30
    assert split["train"][-1] < split["test"][0]


def test_temporal_split_indices_all_covered():
    split = temporal_split_indices(50, train_fraction=0.60)
    all_indices = np.concatenate([split["train"], split["test"]])
    assert len(all_indices) == 50
    assert set(all_indices.tolist()) == set(range(50))


def test_temporal_split_no_overlap():
    split = temporal_split_indices(200, train_fraction=0.80)
    train_set = set(split["train"].tolist())
    test_set = set(split["test"].tolist())
    assert len(train_set & test_set) == 0

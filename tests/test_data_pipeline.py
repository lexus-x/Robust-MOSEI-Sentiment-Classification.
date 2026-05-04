from __future__ import annotations

import random

import torch

from multimod.data import apply_condition, build_dataloaders, load_mosei

from .helpers import write_toy_mosei_pickle


def test_load_mosei_three_class_labels_and_masks(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    text, audio, vision, mask, labels, sample_ids = load_mosei(data_path, split="train")

    assert text.shape == (4, 6, 4)
    assert audio.shape == (4, 6, 3)
    assert vision.shape == (4, 6, 2)
    assert mask.shape == (4, 6)
    assert labels.tolist() == [0, 1, 2, 2]
    assert sample_ids[0].startswith("clip_")
    assert mask[0].tolist() == [False, False, False, True, True, True]

    loaders = build_dataloaders(data_path, batch_size=2, num_workers=0)
    batch = next(iter(loaders["train"]))
    assert "raw_sentiment" in batch
    assert batch["raw_sentiment"].dtype == torch.float32


def test_apply_condition_missing_and_jitter(tmp_path):
    data_path = write_toy_mosei_pickle(tmp_path / "toy_mosei.pkl")
    loaders = build_dataloaders(data_path, batch_size=2, num_workers=0)
    batch = next(iter(loaders["train"]))

    missing_audio = apply_condition(batch, "missing_audio", rng=random.Random(13))
    assert torch.count_nonzero(missing_audio["audio"]) == 0
    assert torch.equal(missing_audio["vision"], batch["vision"])

    jittered = apply_condition(batch, "mild_jitter", rng=random.Random(13))
    assert jittered["audio"].shape == batch["audio"].shape
    assert jittered["vision"].shape == batch["vision"].shape
    assert not torch.equal(jittered["audio"], batch["audio"])

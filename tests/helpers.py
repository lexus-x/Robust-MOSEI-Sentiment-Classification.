"""Helpers for synthetic MOSEI-style test fixtures."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _left_padded_sequence(valid_len: int, total_len: int, feature_dim: int, base: float) -> np.ndarray:
    seq = np.zeros((total_len, feature_dim), dtype=np.float32)
    values = np.linspace(base, base + valid_len - 1, num=valid_len, dtype=np.float32)
    seq[-valid_len:, :] = values[:, None]
    return seq


def write_toy_mosei_pickle(path: str | Path) -> Path:
    path = Path(path)
    total_len = 6
    text_dim, audio_dim, vision_dim = 4, 3, 2
    labels = np.array([-1.5, -0.1, 0.8, 1.4], dtype=np.float32)
    valid_lengths = [3, 4, 5, 6]

    def build_split(offset: float) -> dict[str, object]:
        text = []
        audio = []
        vision = []
        ids = []
        for index, valid_len in enumerate(valid_lengths):
            base = offset + index + 1.0
            text.append(_left_padded_sequence(valid_len, total_len, text_dim, base))
            audio.append(_left_padded_sequence(valid_len, total_len, audio_dim, base * 0.5))
            vision.append(_left_padded_sequence(valid_len, total_len, vision_dim, base * 0.25))
            ids.append(f"clip_{offset:.0f}_{index}")
        return {
            "text": np.stack(text),
            "audio": np.stack(audio),
            "vision": np.stack(vision),
            "labels": labels.reshape(-1, 1, 1),
            "id": ids,
        }

    packed = {
        "train": build_split(0.0),
        "valid": build_split(10.0),
        "test": build_split(20.0),
    }
    with path.open("wb") as handle:
        pickle.dump(packed, handle)
    return path

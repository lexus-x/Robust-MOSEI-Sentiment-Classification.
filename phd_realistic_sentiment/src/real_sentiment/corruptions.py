"""Feature-level corruption operators for realistic multimodal robustness studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: str


def _valid_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.astype(bool))


def contiguous_drop(
    features: np.ndarray,
    mask: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Zero a contiguous fraction of valid tokens."""
    output = np.array(features, copy=True)
    valid = _valid_indices(mask)
    if valid.size == 0 or fraction <= 0.0:
        return output
    span = max(1, int(round(valid.size * fraction)))
    span = min(span, valid.size)
    start_offset = int(rng.integers(0, valid.size - span + 1))
    chosen = valid[start_offset : start_offset + span]
    output[chosen] = 0.0
    return output


def temporal_shift(
    features: np.ndarray,
    mask: np.ndarray,
    shift: int,
) -> np.ndarray:
    """Shift valid tokens relative to the reference stream while preserving left padding."""
    output = np.array(features, copy=True)
    valid = _valid_indices(mask)
    if valid.size == 0 or shift == 0:
        return output
    values = output[valid].copy()
    shifted = np.zeros_like(values)
    if abs(shift) >= valid.size:
        output[valid] = 0.0
        return output
    if shift > 0:
        shifted[shift:] = values[:-shift]
    else:
        shifted[:shift] = values[-shift:]
    output[valid] = shifted
    return output


def progressive_drift(
    features: np.ndarray,
    mask: np.ndarray,
    max_shift: int,
) -> np.ndarray:
    """Apply a monotonic drift so later valid tokens move more than early ones."""
    output = np.array(features, copy=True)
    valid = _valid_indices(mask)
    if valid.size == 0 or max_shift <= 0:
        return output
    values = output[valid].copy()
    drifted = np.zeros_like(values)
    for idx in range(valid.size):
        shift = int(round(max_shift * idx / max(valid.size - 1, 1)))
        src_idx = max(0, idx - shift)
        drifted[idx] = values[src_idx]
    output[valid] = drifted
    return output


def burst_noise(
    features: np.ndarray,
    mask: np.ndarray,
    span_fraction: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject local additive noise into a valid contiguous span."""
    output = np.array(features, copy=True)
    valid = _valid_indices(mask)
    if valid.size == 0 or span_fraction <= 0.0 or scale <= 0.0:
        return output
    span = max(1, int(round(valid.size * span_fraction)))
    span = min(span, valid.size)
    start_offset = int(rng.integers(0, valid.size - span + 1))
    chosen = valid[start_offset : start_offset + span]
    output[chosen] = output[chosen] + rng.normal(loc=0.0, scale=scale, size=output[chosen].shape)
    return output


def apply_corruption(
    features: np.ndarray,
    mask: np.ndarray,
    spec: CorruptionSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if spec.name == "block_missing_audio" or spec.name == "block_missing_vision":
        fraction = {"mild": 0.20, "moderate": 0.40, "severe": 0.60}[spec.severity]
        return contiguous_drop(features, mask, fraction=fraction, rng=rng)
    if spec.name == "lead_lag_audio" or spec.name == "lead_lag_vision":
        shift = {"2_frames": 2, "4_frames": 4, "8_frames": 8}[spec.severity]
        return temporal_shift(features, mask, shift=shift)
    if spec.name == "drift_audio":
        max_shift = {"mild": 2, "moderate": 4}[spec.severity]
        return progressive_drift(features, mask, max_shift=max_shift)
    if spec.name == "burst_noise_vision":
        scale = {"mild": 0.25, "moderate": 0.50}[spec.severity]
        return burst_noise(features, mask, span_fraction=0.25, scale=scale, rng=rng)
    raise ValueError(f"Unsupported corruption spec: {spec}")

"""Claim and benchmark definitions for the PhD-track reboot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AcceptanceGate:
    name: str
    target: str
    rationale: str


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    family: str
    target_modalities: tuple[str, ...]
    severity_levels: tuple[str, ...]
    description: str
    rationale: str


@dataclass(frozen=True)
class ThesisClaim:
    project_name: str
    abstract: str
    north_star_claim: str
    bootstrap_claim: str
    can_say_now: tuple[str, ...]
    cannot_say_now: tuple[str, ...]
    acceptance_gates: tuple[AcceptanceGate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "abstract": self.abstract,
            "north_star_claim": self.north_star_claim,
            "bootstrap_claim": self.bootstrap_claim,
            "can_say_now": list(self.can_say_now),
            "cannot_say_now": list(self.cannot_say_now),
            "acceptance_gates": [asdict(gate) for gate in self.acceptance_gates],
        }


def build_default_thesis_claim() -> ThesisClaim:
    return ThesisClaim(
        project_name="PhD Realistic Sentiment",
        abstract=(
            "A compact multimodal sentiment system should be judged not just by clean accuracy, "
            "but by whether it stays competitive under realistic modality failure, temporal shift, "
            "and uncertainty-sensitive decision making."
        ),
        north_star_claim=(
            "A compact, calibrated multimodal sentiment model can remain competitive with larger robust baselines "
            "under realistic modality failure and temporal shift while improving uncertainty-aware behavior."
        ),
        bootstrap_claim=(
            "The existing repo only supports an efficiency-under-synthetic-robustness bootstrap result, not the full thesis claim."
        ),
        can_say_now=(
            "the old repo provides a compactness baseline worth building from",
            "there is already matched-seed evidence on a synthetic robustness protocol",
            "parameter efficiency can be part of the thesis story",
        ),
        cannot_say_now=(
            "current MOSEI SOTA",
            "published-standard-metric superiority",
            "real-world robustness",
            "better uncertainty-aware decision making",
        ),
        acceptance_gates=(
            AcceptanceGate(
                name="standard_clean_metrics",
                target="Standard MOSEI metrics within a narrow tolerance of a strong robust baseline.",
                rationale="Avoid thesis collapse into custom metric overfitting.",
            ),
            AcceptanceGate(
                name="mean_robustness_parity",
                target="Mean perturbed weighted-F1 gap vs baseline ≤ 0.02.",
                rationale="On average across all realistic conditions, the compact model must stay close.",
            ),
            AcceptanceGate(
                name="worst_case_robustness",
                target="Worst single-condition weighted-F1 gap vs baseline ≤ 0.03.",
                rationale=(
                    "No single condition should catastrophically break the model, "
                    "but a slightly wider tolerance is realistic for extreme corruptions like 60% vision dropout."
                ),
            ),
            AcceptanceGate(
                name="calibration_advantage",
                target="Compact model ECE < baseline ECE on clean and perturbed evaluation.",
                rationale="The compact model should be better calibrated, not just smaller.",
            ),
            AcceptanceGate(
                name="uncertainty_aware_abstention",
                target=(
                    "When the model abstains on high-uncertainty samples, "
                    "accuracy on retained samples improves vs no-abstention baseline."
                ),
                rationale="Robustness claims without uncertainty discipline are weak.",
            ),
            AcceptanceGate(
                name="efficiency",
                target="Materially smaller parameter or runtime budget than the strongest baseline.",
                rationale="Compactness is the clearest current advantage.",
            ),
            AcceptanceGate(
                name="transfer",
                target="Evidence beyond one synthetic protocol or one benchmark split.",
                rationale="A PhD claim must travel.",
            ),
        ),
    )


def build_benchmark_manifest() -> dict[str, Any]:
    conditions = [
        BenchmarkCondition(
            name="clean",
            family="reference",
            target_modalities=("text", "audio", "vision"),
            severity_levels=("none",),
            description="Unmodified packed feature stream.",
            rationale="Reference anchor for all robustness deltas.",
        ),
        BenchmarkCondition(
            name="block_missing_audio",
            family="missingness",
            target_modalities=("audio",),
            severity_levels=("mild", "moderate", "severe"),
            description="Contiguous valid-token spans are zeroed in audio only.",
            rationale="Real sensors fail in bursts, not only globally.",
        ),
        BenchmarkCondition(
            name="block_missing_vision",
            family="missingness",
            target_modalities=("vision",),
            severity_levels=("mild", "moderate", "severe"),
            description="Contiguous valid-token spans are zeroed in vision only.",
            rationale="Occlusion and dropped frames are typically local.",
        ),
        BenchmarkCondition(
            name="lead_lag_audio",
            family="temporal_shift",
            target_modalities=("audio",),
            severity_levels=("2_frames", "4_frames", "8_frames"),
            description="Audio valid tokens are shifted relative to text.",
            rationale="Speech and transcript alignment is not exact in deployment.",
        ),
        BenchmarkCondition(
            name="lead_lag_vision",
            family="temporal_shift",
            target_modalities=("vision",),
            severity_levels=("2_frames", "4_frames", "8_frames"),
            description="Vision valid tokens are shifted relative to text.",
            rationale="Face tracking and frame extraction induce lag and lead.",
        ),
        BenchmarkCondition(
            name="drift_audio",
            family="temporal_drift",
            target_modalities=("audio",),
            severity_levels=("mild", "moderate"),
            description="Audio indices are progressively displaced across the clip.",
            rationale="Sync drift accumulates over time.",
        ),
        BenchmarkCondition(
            name="burst_noise_vision",
            family="local_corruption",
            target_modalities=("vision",),
            severity_levels=("mild", "moderate"),
            description="Short spans receive additive burst noise.",
            rationale="Compression artifacts and detector failures are transient.",
        ),
        BenchmarkCondition(
            name="compound_audio_vision_failure",
            family="compound",
            target_modalities=("audio", "vision"),
            severity_levels=("moderate", "severe"),
            description="Missing spans and temporal shift co-occur across two modalities.",
            rationale="Hard cases are often correlated, not isolated.",
        ),
    ]
    return {
        "benchmark_name": "Realistic Multimodal Sentiment Robustness Protocol",
        "purpose": "Replace single synthetic jitter stress tests with reusable realistic corruption families.",
        "conditions": [asdict(condition) for condition in conditions],
        "primary_metrics": [
            "clean_weighted_f1",
            "avg_perturbed_weighted_f1",
            "clean_mosei_acc_7",
            "clean_mosei_f1_nonneg",
            "ece",
            "selective_risk",
            "coverage_at_fixed_risk",
            "abstention_accuracy",
        ],
    }

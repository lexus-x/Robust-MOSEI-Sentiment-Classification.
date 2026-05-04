"""PhD-track realistic multimodal sentiment project."""

from .bootstrap import build_bootstrap_evidence
from .benchmark import (
    ProtocolSpec,
    apply_protocol_condition,
    build_protocol_specs,
    compare_roles,
    run_protocol_for_root,
)
from .claim import (
    AcceptanceGate,
    BenchmarkCondition,
    ThesisClaim,
    build_benchmark_manifest,
    build_default_thesis_claim,
)
from .reporting import render_proposal_report, render_realistic_benchmark_report, render_thesis_report
from .transfer import evaluate_temporal_transfer, temporal_split_indices

__all__ = [
    "AcceptanceGate",
    "BenchmarkCondition",
    "ProtocolSpec",
    "ThesisClaim",
    "apply_protocol_condition",
    "build_benchmark_manifest",
    "build_bootstrap_evidence",
    "build_default_thesis_claim",
    "build_protocol_specs",
    "compare_roles",
    "evaluate_temporal_transfer",
    "render_proposal_report",
    "render_realistic_benchmark_report",
    "render_thesis_report",
    "run_protocol_for_root",
    "temporal_split_indices",
]


from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.bounded_cloud_density_chain import (
    BoundedCloudDensityChainError,
    _validate_contract,
    evaluate_bounded_cloud_density_chain,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cb_bounded_cloud_density_chain_v1.json"


def test_contract_rejects_tail_compression() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["candidate"]["pointwise_tail_compression_or_clipping_allowed"] = True
    with pytest.raises(BoundedCloudDensityChainError):
        _validate_contract(contract)


def test_formal_evaluation_is_exact_and_gate_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_bounded_cloud_density_chain(contract, ROOT)
    second = evaluate_bounded_cloud_density_chain(contract, ROOT)
    assert first == second
    assert set(first["gate_results"]) == {
        "fixture_count",
        "visible_effect",
        "bounded_p99",
        "p99_strength_parity",
        "bounded_maximum",
        "maximum_tail_reduction",
        "no_new_boundary",
        "no_isolated_excursions",
        "spatially_correlated_residual",
        "bounded_mean_drift",
        "density_transmittance_roundtrip",
        "identity_scanner_exact",
        "repeat_exact",
        "row_partition_exact",
        "input_immutable",
        "wrong_domain_rejected",
        "finite_bounded",
        "no_scanner_double_counting",
        "no_tail_compression_or_normalization",
    }

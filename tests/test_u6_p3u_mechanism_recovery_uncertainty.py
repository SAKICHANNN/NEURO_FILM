from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3u_mechanism_recovery_uncertainty_v1.json"


def test_p3u_contract_freezes_three_increasing_uncertainty_regimes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    regimes = contract["regimes"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert [row["bits"] for row in regimes] == [16, 12, 10]
    assert [row["source_noise_sigma"] for row in regimes] == sorted(
        row["source_noise_sigma"] for row in regimes
    )
    assert [row["maximum_absolute_source_gain_jitter"] for row in regimes] == sorted(
        row["maximum_absolute_source_gain_jitter"] for row in regimes
    )


def test_p3u_contract_keeps_confirmation_sealed_and_clipping_forbidden() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["roles"]["fit_reads_confirmation"] is False
    assert contract["quantization"]["clipping_allowed"] is False
    assert contract["common_gates"]["maximum_quantization_clipped_scalar_count"] == 0
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "fit confirmation" in forbidden

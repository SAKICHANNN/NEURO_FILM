from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3t_multiexposure_mechanism_recovery_v1.json"


def test_p3t_contract_is_frozen_before_implementation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert len(contract["truth_cases"]) == 8
    assert [row["family"] for row in contract["truth_cases"]].count(
        "mixed-abstain"
    ) == 2
    assert contract["roles"]["fit_reads_confirmation"] is False


def test_p3t_contract_keeps_families_bounded_and_distinct() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidates = contract["candidate_families"]
    assert candidates["physical-backing-return"]["fraction_bounds"] == [0.0, 0.2]
    assert candidates["lens-diffusion"]["total_weight_bounds"] == [0.0, 0.3]
    assert sum(candidates["lens-diffusion"]["normalized_weight_shape"]) == 1.0
    assert "gaussian(source, sigma)" in candidates["physical-backing-return"][
        "equation"
    ]
    assert "gaussian_i(source) - source" in candidates["lens-diffusion"][
        "equation"
    ]


def test_p3t_contract_forbids_same_cohort_rescue() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "confirmation targets" in forbidden
    assert "named-stock profile" in forbidden

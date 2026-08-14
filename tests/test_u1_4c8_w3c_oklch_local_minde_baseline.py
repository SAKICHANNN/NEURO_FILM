from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.gamutmlp_oklch_local_minde_baseline import (
    CONTRACT_SHA256,
    LocalMindeBaselineError,
    _mapping_metrics,
    _validate_inputs,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u1_4c8_w3c_oklch_local_minde_baseline_v1.json"


def test_contract_freezes_normative_algorithm_and_no_product_change() -> None:
    contract = load_contract(CONTRACT)
    challenger = contract["challenger"]
    assert challenger["jnd_delta_e_ok"] == 0.02
    assert challenger["chroma_epsilon"] == 0.0001
    assert challenger["origin_space"] == "prophoto-rgb"
    assert challenger["destination_space"] == "rec2020"
    assert challenger["source_or_operator_fitting_allowed"] is False
    assert contract["production_default_changed"] is False
    assert (
        CONTRACT_SHA256
        == "59ab5b3bbfec534b4f34bdbe04fd8c4c29ed659e2c2131f9dafc3ba07c9eef2a"
    )


def test_exact_c7_inputs_and_report_validate() -> None:
    c7, rows, report = _validate_inputs(load_contract(CONTRACT), ROOT)
    assert len(rows) == 24
    assert len({row["camera"] for row in rows}) == 8
    assert report["automatic_pass"] is True
    assert c7["ingress"]["mapping_id"] == "rec2020-luminance-axis-analytical-interior-v1"


def test_mapping_metrics_identity_is_zero() -> None:
    source = np.asarray([[[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]]], dtype=np.float32)
    facts = _mapping_metrics(
        source, source.copy(), np.asarray([[True, False]], dtype=bool)
    )
    assert facts["median_delta_e_ok"] == 0.0
    assert facts["p95_hue_error_degrees"] == 0.0
    assert facts["p95_lightness_error"] == 0.0


def test_contract_drift_rejects(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["challenger"]["jnd_delta_e_ok"] = 0.01
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalMindeBaselineError, match="contract hash drift"):
        load_contract(mutated)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_characteristic_prior import (
    CharacteristicPriorError,
    compile_prior,
    evaluate_prior,
    load_contract,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicCurve,
    ManufacturerCharacteristicPrior,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2q_kodak_250d_characteristic_prior_v1.json"


def test_curve_rejects_nonmonotone_density_and_extrapolation() -> None:
    with pytest.raises(ValueError, match="monotone"):
        ManufacturerCharacteristicCurve(
            "red", np.asarray([0.0, 1.0, 2.0]), np.asarray([0.1, 0.3, 0.2])
        )
    curve = ManufacturerCharacteristicCurve(
        "red", np.asarray([0.0, 1.0, 2.0]), np.asarray([0.1, 0.3, 0.5])
    )
    with pytest.raises(ValueError, match="outside"):
        curve.apply(np.asarray([-1e-9]))


def test_contract_rejects_shape_gate_relaxation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["generic_normalized_shape_rmse_each_channel_min"] = 0.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CharacteristicPriorError, match="contract drift"):
        load_contract(path)


def test_compiled_prior_is_repeat_exact_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first_report, first_bundle = evaluate_prior(contract, ROOT)
    second_report, second_bundle = evaluate_prior(contract, ROOT)
    assert first_report == second_report
    assert first_bundle == second_bundle
    assert first_report["passed"]
    assert first_report["decision"] == "retain_research_manufacturer_characteristic_prior"
    assert all(first_report["gate_results"].values())
    assert first_report["prior"]["input_domain"] == "relative_layer_log_exposure"
    assert min(
        row["rmse"] for row in first_report["generic_u2_2_normalized_shape"].values()
    ) >= 0.05


def test_prior_wire_replay_and_rgb_layer_order() -> None:
    contract = load_contract(CONTRACT)
    trace = json.loads((ROOT / contract["parents"]["trace"]["path"]).read_text())
    prior, _ = compile_prior(trace, source_evidence_id="a" * 64)
    replay = ManufacturerCharacteristicPrior.from_dict(prior.to_dict())
    exposure = np.asarray(
        [[curve.domain[0] for curve in prior.curves], [curve.domain[1] for curve in prior.curves]]
    )
    assert np.array_equal(prior.apply(exposure), replay.apply(exposure))
    assert tuple(curve.layer for curve in prior.curves) == ("red", "green", "blue")

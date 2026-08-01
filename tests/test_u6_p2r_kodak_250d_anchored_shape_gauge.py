from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_anchored_shape_gauge import (
    AnchoredShapeGaugeError,
    evaluate_gauge,
    load_contract,
)
from src.eval.physical_characteristic_prior import compile_prior
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.manufacturer_shape_gauge import compile_anchored_shape_gauge

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2r_kodak_250d_anchored_shape_gauge_v1.json"


def test_contract_rejects_alternate_anchor_rule(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gauge"]["source_anchor_exposure_rule"] = "midpoint"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnchoredShapeGaugeError, match="contract drift"):
        load_contract(path)


def test_anchored_shape_gauge_is_repeat_exact_and_closes() -> None:
    contract = load_contract(CONTRACT)
    first_report, first_bundle = evaluate_gauge(contract, ROOT)
    second_report, second_bundle = evaluate_gauge(contract, ROOT)
    assert first_report == second_report
    assert first_bundle == second_bundle
    assert not first_report["passed"]
    assert first_report["decision"] == "close_anchored_shape_gauge"
    assert not first_report["gate_results"]["linear_roundtrip"]
    assert not first_report["gate_results"]["generic_difference_population"]
    assert all(
        value
        for key, value in first_report["gate_results"].items()
        if key not in {"linear_roundtrip", "generic_difference_population"}
    )
    assert first_report["operator"]["input_space"] == "relative_layer_exposure_rgb_order"


def test_gauge_preserves_generic_encoder_and_shared_anchor() -> None:
    contract = load_contract(CONTRACT)
    p2q = json.loads((ROOT / contract["parents"]["p2q_contract"]["path"]).read_text())
    trace = json.loads((ROOT / p2q["parents"]["trace"]["path"]).read_text())
    decision = json.loads(
        (ROOT / contract["parents"]["p2q_decision"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    manufacturer, _ = compile_prior(
        trace,
        source_evidence_id=p2q["parents"]["p2p_decision"][
            "required_stable_evidence_id"
        ],
    )
    assert manufacturer.identity() == decision["prior_identity"]
    generic_config = json.loads(
        (ROOT / contract["parents"]["generic_u2_2"]["path"]).read_text()
    )
    generic = build_operator(generic_config)
    operator, metadata = compile_anchored_shape_gauge(
        manufacturer, generic, generic_config
    )
    assert operator.encoder.to_dict() == generic.encoder.to_dict()
    assert np.array_equal(
        operator.apply(np.asarray([[0.18, 0.18, 0.18]])), np.ones((1, 3))
    )
    assert max(row.strictness_repair_max_density for row in metadata) <= 1e-12

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk10_orthogonal_perceptual_residual_v1.json"


def test_bk10_primitive_contract_is_frozen_before_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = config["parent"]
    assert config["status"] == "primitive_contract_frozen"
    assert hashlib.sha256((ROOT / parent["decision"]).read_bytes()).hexdigest() == (
        parent["decision_sha256"]
    )
    operator = config["operator"]
    assert operator["residual_strength"] == 0.8
    assert operator["maximum_orthogonal_delta_e76"] == 12.0
    assert operator["gamut_iterations"] == 24
    mechanism = config["mechanism_basis"]
    assert mechanism["synthetic_only_parameter_selection"]
    assert not mechanism["bk8_or_bk9_pixels_read_for_parameter_selection"]
    assert not mechanism["per_image_fit_or_routing"]
    assert not mechanism["dense_arm_weight_prediction"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["production_default_changed"]


def test_bk10_synthetic_witnesses_exceed_frozen_primitive_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    witness = config["mechanism_basis"]
    gates = config["primitive_gates"]
    assert witness["prototype_17_cube_median_style_delta_e76"] >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert witness[
        "prototype_17_cube_median_increment_vs_base_delta_e76"
    ] >= gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
    assert witness[
        "prototype_maximum_increment_before_gamut_delta_e76"
    ] <= gates["maximum_increment_before_gamut_delta_e76"]
    assert witness["prototype_maximum_absolute_projection_dot"] <= (
        gates["maximum_absolute_projection_dot"]
    )
    assert witness["prototype_new_uint16_boundary_fraction"] == (
        gates["new_uint16_boundary_fraction_on_17_cube"]
    )

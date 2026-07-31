from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk7_smooth_perceptual_hue_density_v1.json"


def test_bk7_primitive_contract_is_frozen_before_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = config["parent"]
    assert config["status"] == "primitive_contract_frozen"
    assert hashlib.sha256((ROOT / parent["decision"]).read_bytes()).hexdigest() == (
        parent["decision_sha256"]
    )
    assert config["operator"]["strength"] == 1.0
    assert config["operator"]["gamut_iterations"] == 24
    assert config["mechanism_basis"]["synthetic_only_parameter_selection"]
    assert not config["mechanism_basis"]["bk6_pixels_read_for_parameter_selection"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["production_default_changed"]
    assert not config["stock_or_authenticity_claim_allowed"]


def test_bk7_synthetic_witnesses_exceed_frozen_primitive_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    witness = config["mechanism_basis"]
    gates = config["primitive_gates"]
    assert witness["prototype_17_cube_median_style_delta_e76"] >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert witness["prototype_zero_to_one_axis_max_delta_e76"] <= (
        gates["maximum_zero_to_one_axis_delta_e76"]
    )
    assert witness["prototype_maximum_neutral_axis_channel_range"] <= (
        gates["maximum_neutral_axis_channel_range"]
    )
    assert witness["prototype_new_uint16_boundary_fraction"] == (
        gates["new_uint16_boundary_fraction_on_17_cube"]
    )

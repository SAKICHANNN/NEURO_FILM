from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.factorized_ao6_perceptual_residual_regression import (
    FactorizedAO6ResidualRegressionError,
    validate_contract,
)
from src.roll2film.factorized_ao6_perceptual_residual import (
    FactorizedAO6PerceptualResidual,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk16_factorized_ao6_perceptual_residual_v1.json"
)


def test_bk16_contract_is_synthetic_only_and_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["operator"] == FactorizedAO6PerceptualResidual()
    assert validated["source_sha256"] == (
        "a81bd65ce6b88f50d26862ae6f38a03109a7ca02d782eb87ffc3368d657afe9a"
    )
    parent = config["parent"]
    assert hashlib.sha256((ROOT / parent["decision"]).read_bytes()).hexdigest() == (
        parent["decision_sha256"]
    )
    basis = config["mechanism_basis"]
    assert basis["synthetic_only_parameter_selection"]
    assert not basis[
        "bk13_bk14_bk15_pixels_or_votes_read_for_parameter_selection"
    ]
    assert not basis["rejected_per_pixel_projection_prototype"]["retained"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["production_default_changed"]


def test_bk16_prototype_evidence_exceeds_frozen_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["mechanism_basis"]
    gates = config["primitive_gates"]
    assert evidence["prototype_17_cube_median_style_delta_e76"] >= (
        gates["minimum_17_cube_median_style_delta_e76"]
    )
    assert evidence[
        "prototype_17_cube_median_increment_vs_base_delta_e76"
    ] >= gates["minimum_17_cube_median_increment_vs_base_delta_e76"]
    assert evidence[
        "prototype_17_cube_maximum_increment_vs_base_delta_e76"
    ] <= gates["maximum_final_increment_vs_base_delta_e76"]
    assert evidence["prototype_17_cube_final_cap_fraction"] <= (
        gates["maximum_17_cube_final_cap_fraction"]
    )
    assert evidence["prototype_new_uint16_boundary_fraction"] == 0.0
    assert evidence["prototype_neutral_ramp_minimum_lightness_step"] >= 0.0
    assert evidence["prototype_neutral_ramp_maximum_chroma"] <= (
        evidence["prototype_safe_base_neutral_ramp_maximum_chroma"]
    )


def test_bk16_contract_mutation_fails_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["operator"]["residual_strength"] = 0.7
    with pytest.raises(FactorizedAO6ResidualRegressionError):
        validate_contract(ROOT, config)

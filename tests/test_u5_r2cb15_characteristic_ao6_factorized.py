from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.characteristic_ao6_factorized import (
    apply_characteristic_ao6_factorized,
    load_contract,
)
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction

ROOT = Path(__file__).resolve().parents[1]


def test_cb15_synthetic_factorization_is_luminance_exact_and_bounded() -> None:
    config = load_contract(
        ROOT / "configs/u5_r2cb15_characteristic_ao6_factorized_v1.json"
    )
    cb11 = _load_exact_json(
        ROOT,
        config["parents"]["cb11_contract_path"],
        config["parents"]["cb11_contract_sha256"],
    )
    curve = _compiled_curve(load_cb6(ROOT / cb11["parents"]["cb6_contract_path"]))
    rng = np.random.default_rng(1515)
    source = rng.uniform(0.01, 0.99, size=(19, 23, 3)).astype(np.float32)
    ao6 = rng.uniform(0.0, 1.0, size=source.shape).astype(np.float32)
    weights = np.asarray(cb11["operator"]["luminance_weights"], dtype=np.float64)
    output, scale, error = apply_characteristic_ao6_factorized(
        source,
        ao6,
        curve,
        weights=weights,
        strength=cb11["operator"]["nominal_strength"],
        boundary_epsilon=cb11["operator"]["boundary_epsilon"],
    )
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert float(scale.min()) >= 0.0
    assert float(scale.max()) <= 1.0
    assert float(np.max(np.abs(error))) <= 1e-6
    assert (
        _new_boundary_fraction(source, output, cb11["operator"]["boundary_epsilon"])
        == 0.0
    )

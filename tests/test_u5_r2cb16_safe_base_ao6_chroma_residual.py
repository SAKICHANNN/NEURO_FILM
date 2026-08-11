from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction
from src.eval.safe_base_ao6_chroma_residual import (
    apply_safe_base_ao6_chroma_residual,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb16_safe_base_residual_is_luminance_exact_and_bounded() -> None:
    config = load_contract(
        ROOT / "configs/u5_r2cb16_safe_base_ao6_chroma_residual_v1.json"
    )
    assert config["experiment_id"] == "U5.R2CB16"
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rng = np.random.default_rng(1616)
    source = rng.uniform(0.01, 0.99, size=(19, 23, 3)).astype(np.float32)
    safe_base = source.copy()
    ao6 = rng.uniform(0.0, 1.0, size=source.shape).astype(np.float32)
    source_before = source.copy()
    base_before = safe_base.copy()
    ao6_before = ao6.copy()
    output, scale, error = apply_safe_base_ao6_chroma_residual(
        source,
        safe_base,
        ao6,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
    )
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert float(scale.min()) >= 0.0
    assert float(scale.max()) <= 1.0
    assert float(np.max(np.abs(error))) <= 1e-6
    assert _new_boundary_fraction(source, output, 1.0 / 65535.0) == 0.0
    assert np.array_equal(source, source_before)
    assert np.array_equal(safe_base, base_before)
    assert np.array_equal(ao6, ao6_before)

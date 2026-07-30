from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_strict_interior_lut_basis_development import (
    EPSILON,
    FiveKStrictInteriorLUTError,
    apply_strict_interior_lut,
    fit_strict_interior_lut,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bj2_fivek_strict_interior_lut_basis_development_v1.json"


def test_contract_and_parent_are_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(validate_contract(ROOT, config)["populations"]) == 3
    drifted = json.loads(CONFIG.read_text(encoding="utf-8"))
    drifted["operator"]["boundary_epsilon"] = 0.0
    with pytest.raises(FiveKStrictInteriorLUTError):
        validate_contract(ROOT, drifted)


def test_extreme_coefficients_preserve_strict_boundary_class() -> None:
    rng = np.random.default_rng(91)
    source = rng.random((257, 263, 3), dtype=np.float64)
    source[0, 0] = [0.0, EPSILON / 2.0, EPSILON]
    source[0, 1] = [1.0, 1.0 - EPSILON / 2.0, 1.0 - EPSILON]
    source_boundary = (source <= EPSILON) | (source >= 1.0 - EPSILON)
    for value in (-1.0, 1.0):
        output = apply_strict_interior_lut(
            source, np.full((4, 4, 4, 3), value)
        )
        output_boundary = (output <= EPSILON) | (
            output >= 1.0 - EPSILON
        )
        assert not np.any(output_boundary & ~source_boundary)
        assert np.array_equal(output[source_boundary], source[source_boundary])


def test_fit_improves_known_headroom_transform() -> None:
    rng = np.random.default_rng(97)
    source = rng.random((96, 80, 3), dtype=np.float64)
    headroom = np.maximum(
        0.0, np.minimum(source - EPSILON, 1.0 - EPSILON - source)
    )
    target = source + headroom * np.asarray([0.4, -0.25, 0.15])
    fitted = fit_strict_interior_lut(
        source,
        target,
        grid_size=4,
        sample_stride=2,
        identity_shrinkage=0.01,
        smoothness=0.1,
        coefficient_minimum=-1.0,
        coefficient_maximum=1.0,
    )
    before = float(np.sqrt(np.mean((source - target) ** 2)))
    after = float(
        np.sqrt(
            np.mean((apply_strict_interior_lut(source, fitted) - target) ** 2)
        )
    )
    assert after < before * 0.05

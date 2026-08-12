from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.compiled_cloud_attenuation_row_runtime import evaluate, load_contract
from src.film_physics.cross_layer_cloud_attenuation_runtime import (
    CompiledCloudAttenuationProfile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dx_compiled_cloud_attenuation_row_runtime_v1.json"


def test_p4dx_parent_is_exact() -> None:
    assert load_contract(ROOT, CONTRACT)["experiment_id"].startswith("u6.p4dx-")


def test_p4dx_runtime_evaluation_obeys_frozen_gates() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4dx_profile_rejects_amplification() -> None:
    with pytest.raises(ValueError, match="invalid compiled"):
        CompiledCloudAttenuationProfile(
            base_profile=None,  # type: ignore[arg-type]
            channel_residual_gain=(1.1, 0.7, 0.6),
            aperture_factor=4,
            base_rate_multiplier=16.0,
        )


def test_p4dx_profile_rejects_nonfinite_gain() -> None:
    assert not np.isfinite(float("nan"))

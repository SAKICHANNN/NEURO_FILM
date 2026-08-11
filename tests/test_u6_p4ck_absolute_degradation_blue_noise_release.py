from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import absolute_degradation_blue_noise_release as p4ck

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "configs/u6_p4ck_absolute_degradation_blue_noise_release_audit_v1.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_is_exact() -> None:
    p4ck._validate_contract(_contract())


def test_signed_zero_summary_preserves_raw_evidence() -> None:
    values = np.asarray([0.0, -0.0, 0.0, -0.0], dtype=np.float16)
    summary = p4ck._summarize_array(values)
    assert summary["numeric_nonzero_count"] == 0
    assert summary["standard_deviation"] == 0.0
    assert summary["raw_float16_bit_counts"] == {"0x0000": 2, "0x8000": 2}


def test_official_roll_resize_branches_preserve_zero() -> None:
    basis = np.zeros((32, 32, 5), dtype=np.float16)
    basis[::2] = -0.0
    for grain_size in (1.0, 1.15, 1.3):
        layer = p4ck._official_grain_layer(
            basis,
            channel=3,
            offset_yx=(17, 31),
            grain_size=grain_size,
            height=17,
            width=19,
        )
        assert layer.shape == (17, 19)
        assert np.count_nonzero(layer) == 0
        assert float(layer.std()) == 0.0


def test_official_roll_resize_retains_real_nonzero_basis() -> None:
    yy, xx = np.indices((32, 32), dtype=np.float32)
    basis = np.stack(
        [np.sin(xx * (channel + 1) * 0.11) + np.cos(yy * 0.17) for channel in range(5)],
        axis=2,
    ).astype(np.float16)
    layer = p4ck._official_grain_layer(
        basis,
        channel=2,
        offset_yx=(7, 11),
        grain_size=1.15,
        height=17,
        width=19,
    )
    assert np.count_nonzero(layer) > 0
    assert float(layer.std()) > 0.0

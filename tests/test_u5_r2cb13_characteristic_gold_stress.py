from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.characteristic_gold_stress import _encode_png, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cb13_contract_is_frozen_partial_gold_scope() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb13_characteristic_gold_stress_v1.json"
    )
    assert contract["population"]["required_unavailable_ids"] == ["FS_FACE_01"]
    assert contract["population"]["expected_available_source_count"] == 40
    assert "not the complete gold set" in contract["claim_ceiling"]


def test_cb13_png_encoding_is_exact_and_bounded() -> None:
    source = np.linspace(0.0, 1.0, 3 * 9 * 11, dtype=np.float32).reshape(9, 11, 3)
    first, pixels = _encode_png(source)
    second, replay = _encode_png(source.copy())
    assert first == second
    assert np.array_equal(pixels, replay)
    assert pixels.dtype == np.uint8

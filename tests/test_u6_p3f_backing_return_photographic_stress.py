from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_backing_return_photographic_stress import (
    _diagnostic_map,
    _isolated_excursions,
    _load_source,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p3f_backing_return_photographic_stress_v1.json"
MANIFEST = (
    ROOT
    / "outputs"
    / "u5_r2ai1s_rawpixls_confirmation_source_preflight_v1"
    / "manifest.json"
)


def test_contract_pins_existing_rights_cleared_population() -> None:
    contract = load_contract(CONTRACT)
    assert contract["input"]["expected_rows"] == 18
    assert contract["input"]["expected_camera_makes"] == 9
    assert contract["production_integration_allowed"] is False
    assert contract["parents"]["p3e_report_sha256"].startswith("59db1174")


def test_exact_source_load_and_hash_rejection(tmp_path: Path) -> None:
    row = json.loads(MANIFEST.read_text(encoding="utf-8"))[0]
    source, digest = _load_source(row, ROOT)
    assert source.dtype == np.float32
    assert source.shape[-1] == 3
    assert digest == row["decoded_sha256"]
    bad = dict(row)
    bad["decoded_path"] = str(tmp_path / "bad.png")
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="hash drift"):
        _load_source(bad, ROOT)


def test_isolated_excursion_detector_distinguishes_speckle_from_field() -> None:
    difference = np.zeros((9, 9, 3), dtype=np.float32)
    difference[4, 4, 0] = 0.01
    assert (
        _isolated_excursions(
            difference, threshold=0.005, radius=2, minimum_support=3
        )
        == 1
    )
    difference[3:6, 3:6, 0] = 0.01
    assert (
        _isolated_excursions(
            difference, threshold=0.005, radius=2, minimum_support=3
        )
        == 0
    )


def test_diagnostic_mapper_is_finite_bounded_and_monotone() -> None:
    ramp = np.asarray([0.0, 0.25, 1.0, 4.0], dtype=np.float32)
    values = np.repeat(ramp.reshape(1, 4, 1), 3, axis=2)
    mapped = _diagnostic_map(values)
    assert np.all(np.isfinite(mapped))
    assert np.all((mapped >= 0.0) & (mapped < 1.0))
    assert np.all(np.diff(mapped[0, :, 0]) > 0.0)

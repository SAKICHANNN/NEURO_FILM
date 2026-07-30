from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.eval.creative_diffusion_photographic_stress import (
    evaluate,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_4b_creative_diffusion_photographic_stress_v1.json"


def _row(root: Path, sample_id: str, make: str, value: int) -> dict[str, object]:
    path = root / f"{sample_id}.png"
    array = np.full((31, 37, 3), value, dtype=np.uint8)
    array[12:19, 15:22] = min(255, value + 80)
    Image.fromarray(array, mode="RGB").save(path)
    return {
        "id": sample_id,
        "make": make,
        "decoded_path": path.name,
        "decoded_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "allowed_use": "test",
        "rights_scope": "test",
        "decoded_color_state": "test",
    }


def _contract() -> dict[str, object]:
    contract = load_contract(CONTRACT)
    contract["input"] = {
        "expected_rows": 2,
        "expected_camera_makes": 2,
        "required_allowed_use": "test",
        "required_rights_scope": "test",
        "required_color_state": "test",
    }
    contract["visual_protocol"]["fixed_ids"] = ["a", "b"]
    contract["automatic_gates"]["minimum_population_median_mean_abs_change"] = 0.0
    contract["automatic_gates"]["isolated_excursion_threshold"] = 0.1
    return contract


def test_small_photographic_report_repeats(tmp_path: Path) -> None:
    manifest = [_row(tmp_path, "a", "A", 40), _row(tmp_path, "b", "B", 90)]
    first = evaluate(
        _contract(),
        manifest,
        root=tmp_path,
        output_root=tmp_path / "first",
    )
    second = evaluate(
        _contract(),
        manifest,
        root=tmp_path,
        output_root=tmp_path / "second",
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["aggregates"]["source_count"] == 2
    assert first["contact_sheet_sha256"] is not None


def test_source_hash_drift_fails_before_render(tmp_path: Path) -> None:
    row = _row(tmp_path, "a", "A", 40)
    row["decoded_sha256"] = "0" * 64
    contract = _contract()
    contract["input"]["expected_rows"] = 1
    contract["input"]["expected_camera_makes"] = 1
    contract["visual_protocol"]["fixed_ids"] = ["a"]
    with pytest.raises(ValueError, match="hash drift"):
        evaluate(
            contract,
            [row],
            root=tmp_path,
            output_root=tmp_path / "out",
        )


def test_parent_and_input_identities_are_frozen() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["operator"]["fit_to_photographs"] is False
    assert contract["parents"]["required_decision"].startswith("pass_generic")
    assert "film halation" in contract["forbidden_fallbacks"][1]

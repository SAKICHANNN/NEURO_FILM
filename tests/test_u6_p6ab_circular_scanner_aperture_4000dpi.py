from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.circular_scanner_aperture_4000dpi import (
    CircularScanner4000DpiError,
    evaluate_circular_scanner_aperture_4000dpi,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ab_circular_scanner_aperture_4000dpi_v1.json"
PARENT = ROOT / "configs/u6_p6aa_circular_scanner_aperture_decision_v1.json"


def test_contract_is_direct_and_reference_free() -> None:
    contract = load_contract(CONTRACT)
    assert contract["compiler"]["reference_pixel_pitch_um"] == 6.35
    assert contract["compiler"]["production_import_allowed"] is False
    assert "1um reference" in contract["forbidden"][1]


def test_contract_rejects_pitch_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["reference_pixel_pitch_um"] = 1.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CircularScanner4000DpiError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AA parent decision unavailable")
def test_target_evaluator_repeats_without_profile_integration() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_circular_scanner_aperture_4000dpi(contract, ROOT)
    second = evaluate_circular_scanner_aperture_4000dpi(contract, ROOT)
    assert first == second
    assert first["metrics"]["kernel_shape"] == [3, 3]
    assert contract["compiler"]["production_import_allowed"] is False

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.circular_scanner_aperture import (
    CircularScannerAuditError,
    evaluate_circular_scanner_aperture,
    load_contract,
)
from src.film_physics.circular_scanner_aperture import (
    CircularApertureError,
    analytic_circular_aperture_mtf,
    compile_circular_aperture_kernel,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6aa_circular_scanner_aperture_v1.json"
SOURCE = (
    ROOT / "outputs/eval/u6_p4at_nasa_joint_mtf_granularity_source_v1/report_run1.json"
)


def test_analytic_identity_and_invalid_request() -> None:
    assert analytic_circular_aperture_mtf(np.array([0.0]), 12.5)[0] == 1.0
    with pytest.raises(CircularApertureError):
        compile_circular_aperture_kernel(
            aperture_diameter_um=0.0,
            pixel_pitch_um=1.0,
            subpixels_per_axis=512,
        )


def test_contract_rejects_runtime_integration_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["production_import_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CircularScannerAuditError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not SOURCE.is_file(), reason="P6AA source report unavailable")
def test_evaluator_is_exact_and_reference_only() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_circular_scanner_aperture(contract, ROOT)
    second = evaluate_circular_scanner_aperture(contract, ROOT)
    assert first == second
    assert first["metrics"]["kernel_shape"] == [13, 13]
    assert first["metrics"]["minimum_weight"] >= 0.0
    assert contract["compiler"]["production_import_allowed"] is False

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.controlled_halation_acquisition_protocol import (
    HalationAcquisitionProtocolError,
    validate_candidate_rows,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/u6_p3y_controlled_halation_acquisition_protocol_v1.json"
RUNNER = ROOT / "scripts/run_u6_p3y_controlled_halation_acquisition_protocol.py"


def _valid_row(protocol: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {
        field: "value"
        for field in protocol["required_row_fields"]  # type: ignore[index]
    }
    values.update(
        {
            "row_id": "row-001",
            "role": "development",
            "structure_id": "vertical-edge",
            "exposure_scale": 1.0,
            "stock_id": "test-stock",
            "roll_id": "roll-1",
            "process_id": "process-1",
            "scanner_id": "scanner-1",
            "source_bits": 10,
            "target_bits": 10,
            "source_clipping_count": 0,
            "target_clipping_count": 0,
            "exposure_anchor_error_ppm": 100,
        }
    )
    for field in (
        "source_file_sha256",
        "target_file_sha256",
        "source_sample_sha256",
        "target_sample_sha256",
        "exposure_anchor_evidence_sha256",
        "alignment_evidence_sha256",
    ):
        values[field] = "a" * 64
    return values


def test_p3y_protocol_is_exact_and_non_renderable() -> None:
    first = validate_protocol(ROOT, PROTOCOL)
    second = validate_protocol(ROOT, PROTOCOL)
    assert first == second
    assert first["passed"] is True
    assert first["render_authority"] is False
    assert first["product_authority"] is False
    assert first["maximum_anchor_error_ppm"] == 100


def test_p3y_candidate_rows_fail_closed_on_anchor_or_clipping() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    row = _valid_row(protocol)
    validate_candidate_rows(protocol, [row])
    weak_anchor = copy.deepcopy(row)
    weak_anchor["exposure_anchor_error_ppm"] = 101
    with pytest.raises(HalationAcquisitionProtocolError, match="anchor too weak"):
        validate_candidate_rows(protocol, [weak_anchor])
    clipped = copy.deepcopy(row)
    clipped["target_clipping_count"] = 1
    with pytest.raises(HalationAcquisitionProtocolError, match="is clipped"):
        validate_candidate_rows(protocol, [clipped])


def test_p3y_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout

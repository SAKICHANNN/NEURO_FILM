from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from src.preprocess import (
    OFFICIAL_ROMM_ICC_SHA256,
    REC2020_SDR_CICP,
    ROMM_REC2020_CAPABILITY_ID,
    ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256,
    ROMMRec2020ConversionError,
    convert_official_romm_rgb16_to_rec2020_png,
    load_working_image,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_official_romm_to_rec2020.py"
_PROFILE = base64.b64decode(
    "AAADYG5vbmUEAAAAc3BhY1JHQiBYWVogB9YACwATAA8AEwA1YWNzcAAAAAAAAAAAbm9uZW5vbmU"
    "AAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1ub25lLJihZpUlfVITkG4EAsDqyQAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAGZGVzYwAAAMwAAABUQTJCMAAAASAAAADUQjJBMAAAAfQAAADUd3Rw"
    "dAAAAsgAAAAUY3BydAAAAtwAAABYY2hhZAAAAzQAAAAsbWx1YwAAAAAAAAABAAAADGVuVVMAAAA4"
    "AAAAHABJAFMATwAgADIAMgAwADIAOAAtADIAIABSAE8ATQBNACAAUgBHAEIAIABwAHIAbwBmAGkA"
    "bABlbUFCIAAAAAADAwAAAAAAIAAAAEQAAAB0AAAAAAAAAABjdXJ2AAAAAAAAAABjdXJ2AAAAAAAA"
    "AABjdXJ2AAAAAAAAAAAAAGYZAAARTgAABAMAACTeAABbHgAAACIAAAAAAAAAAAAAaXwAAABuAAAA"
    "cgAAAF5wYXJhAAAAAAADAAAAAczNAAEAAAAAAAAAABAAAAAIAHBhcmEAAAAAAAMAAAABzM0AAQAA"
    "AAAAAAAAEAAAAAgAcGFyYQAAAAAAAwAAAAHMzQABAAAAAAAAAAAQAAAACABtQkEgAAAAAAMDAAAA"
    "AAAgAAAARAAAAHQAAAAAAAAAAGN1cnYAAAAAAAAAAGN1cnYAAAAAAAAAAGN1cnYAAAAAAAAAAAAC"
    "sSj//30d///l9f/+6SsAAwQzAAAJoQAAAAAAAAAAAAJtSf///zX///81////NXBhcmEAAAAAAAMA"
    "AAAAjjkAAQAAAAAAAAAQAAAAAACAcGFyYQAAAAAAAwAAAACOOQABAAAAAAAAABAAAAAAAIBwYXJh"
    "AAAAAAADAAAAAI45AAEAAAAAAAAAEAAAAAAAgFhZWiAAAAAAAADbrAAA49cAALv1bWx1YwAAAAAA"
    "AAABAAAADGVuVVMAAAA8AAAAHABDAG8AcAB5AHIAaQBnAGgAdAAgADIAMAAwADYAIABIAGUAdwBs"
    "AGUAdAB0AC0AUABhAGMAawBhAHIAZHNmMzIAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA"
    "AAAAAAAAAQAA"
)


def _write_romm(path: Path, profile: bytes = _PROFILE) -> None:
    pixels = np.asarray(
        [
            [[65535, 0, 0], [0, 65535, 0], [0, 0, 65535]],
            [[4096, 32768, 61440], [49152, 8192, 32768], [32768, 32768, 32768]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(34675, "B", len(profile), profile, False)],
    )


def test_official_romm_product_conversion_is_bounded_exact_and_tagged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    output = tmp_path / "output.png"
    _write_romm(source)
    receipt = convert_official_romm_rgb16_to_rec2020_png(source, output)

    assert receipt["capability_id"] == ROMM_REC2020_CAPABILITY_ID
    assert (
        receipt["qualification"]["evidence_sha256"]
        == ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256
    )
    assert receipt["input"]["embedded_icc_sha256"] == OFFICIAL_ROMM_ICC_SHA256
    assert receipt["diagnostics"]["mapped_pixel_count"] > 0
    assert receipt["diagnostics"]["output_minimum"] >= 0.0
    assert receipt["diagnostics"]["output_maximum"] <= 1.0
    assert receipt["output"]["cicp"] == REC2020_SDR_CICP.hex()
    assert receipt["output"]["exact_sample_readback"] is True

    restored = load_working_image(output)
    assert restored.working_space == "linear_rec2020"
    assert restored.transfer_state == "display_linear"
    stored = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert stored is not None and stored.dtype == np.uint16


def test_official_romm_product_qualification_evidence_is_exact() -> None:
    evidence = ROOT / "docs/evidence/U1_4C13_CC0_SYNTHETIC_ROMM_STRESS_RESULT.json"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == (
        ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_SCOPED_CC0_SYNTHETIC_ROMM_STRESS"
    assert payload["automatic_pass"] is True


def test_official_romm_product_cli_writes_bound_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.tiff"
    output = tmp_path / "output.png"
    receipt_path = tmp_path / "receipt.json"
    _write_romm(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
            "--receipt",
            str(receipt_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert completed.stdout.strip() == str(output)
    assert receipt["output"]["sha256"]
    assert receipt["production_default_changed"] is False


def test_official_romm_product_rejects_profile_drift_without_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tiff"
    output = tmp_path / "output.png"
    profile = bytearray(_PROFILE)
    profile[-1] ^= 1
    _write_romm(source, bytes(profile))
    with pytest.raises(ROMMRec2020ConversionError, match="exact official ROMM ICC"):
        convert_official_romm_rgb16_to_rec2020_png(source, output)
    assert not output.exists()

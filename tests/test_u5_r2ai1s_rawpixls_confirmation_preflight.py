from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.rawpixls_confirmation_preflight import (
    ConfirmationSourceError,
    cross_pool_pairs,
    dhash64,
    duplicate_pairs,
    hamming64,
    image_diagnostics,
    validate_contract,
    validate_manifest_row,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ai1s_rawpixls_confirmation_source_preflight_v1.json"
)


def test_frozen_contract_validates() -> None:
    validate_contract(ROOT, json.loads(CONFIG.read_text(encoding="utf-8")))


def test_dhash_and_diagnostics_are_deterministic() -> None:
    values = np.zeros((24, 32, 3), dtype=np.uint8)
    values[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 7
    values[..., 1] = np.arange(24, dtype=np.uint8)[:, None] * 9
    values[..., 2] = 180
    image = Image.fromarray(values, mode="RGB")
    first = dhash64(image)
    assert first == dhash64(image.copy())
    assert hamming64(first, first) == 0
    diagnostics = image_diagnostics(image)
    assert diagnostics["near_empty_or_monochrome"] is False


def test_duplicate_and_cross_pool_pairs_are_explicit() -> None:
    current = [
        {"id": "a", "decoded_sha256": "x", "dhash64": "0" * 16},
        {"id": "b", "decoded_sha256": "y", "dhash64": "0" * 15 + "1"},
    ]
    exact, near = duplicate_pairs(current, threshold=4)
    assert exact == []
    assert near == [{"left": "a", "right": "b", "distance": 1}]
    cross_exact, cross_near = cross_pool_pairs(
        current,
        [{"id": "old", "decoded_sha256": "x", "dhash64": "f" * 16}],
        threshold=4,
    )
    assert cross_exact == [{"left": "a", "right": "old"}]
    assert cross_near == []


def test_manifest_lineage_fails_closed_on_missing_or_invented_metadata() -> None:
    row = {
        "source_id": "rawpixls:x",
        "source_url": "https://raw.pixls.us/getfile.php/x",
        "author": "unknown",
        "license": "CC0/Public Domain",
        "license_snapshot_date": "2026-07-28",
        "rights_scope": "CC0_public_domain_internal_evaluation",
        "scene_group": "unknown",
        "roll_group": "unknown",
        "lab_group": "unknown",
        "scanner_group": "unknown",
        "uploader_group": "unknown",
        "decoded_sha256": "a" * 64,
        "dhash64": "0" * 16,
        "derivation_lineage": {"decoder": "rawpy_libraw"},
        "allowed_use": "internal_independent_digital_ood_confirmation",
    }
    validate_manifest_row(row)
    missing = dict(row)
    del missing["roll_group"]
    with pytest.raises(ConfirmationSourceError, match="lacks required"):
        validate_manifest_row(missing)
    invented = dict(row)
    invented["scanner_group"] = "looks_like_frontier"
    with pytest.raises(ConfirmationSourceError, match="must not invent"):
        validate_manifest_row(invented)


def test_runner_help_loads_from_repo_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/run_u5_r2ai1s_rawpixls_confirmation_preflight.py"
            ),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in completed.stdout

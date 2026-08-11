from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.analytic_y_chromaticity_face_stress import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb56_analytic_y_chromaticity_face_stress_v1.json"


def test_cb56_binds_two_exact_face_arrays_and_original_lineage() -> None:
    config = load_contract(CONTRACT)
    source = config["source"]
    arrays = []
    for manifest_path, manifest_sha, array_path in zip(
        source["manifest_paths"],
        source["manifest_sha256"],
        source["decoded_srgb_npy_paths"],
        strict=True,
    ):
        raw = (ROOT / manifest_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest_sha
        manifest = json.loads(raw)
        row = next(
            row
            for row in manifest["records"]
            if row["sample_id"] == source["sample_id"]
        )
        assert row["source_path"] == source["original_logical_path"]
        assert row["source_sha256"] == source["original_file_sha256"]
        assert row["source_npy_path"] == array_path
        assert row["source_npy_sha256"] == source["decoded_srgb_npy_sha256"]
        payload = (ROOT / array_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["decoded_srgb_npy_sha256"]
        arrays.append(np.load(ROOT / array_path, allow_pickle=False))
    assert np.array_equal(arrays[0], arrays[1])
    assert list(arrays[0].shape) == source["expected_shape"]
    assert str(arrays[0].dtype) == source["expected_dtype"]


def test_cb56_keeps_cb52_automatic_gates() -> None:
    config = load_contract(CONTRACT)
    cb52 = json.loads(
        (ROOT / config["parents"]["cb52_contract_path"]).read_text(encoding="utf-8")
    )
    for key in (
        "maximum_luminance_reconstruction_error",
        "maximum_new_hard_boundary_fraction",
        "maximum_p999_gradient_ratio_vs_source",
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction",
    ):
        assert config["automatic_gates"][key] == cb52["automatic_gates"][key]
    assert config["source"]["claim_exclusion"].startswith(
        "The unavailable original file is not reopened"
    )

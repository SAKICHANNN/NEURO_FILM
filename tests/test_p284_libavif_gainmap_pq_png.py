from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_p284_libavif_gainmap_pq_png import invalid_publication_controls
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.rec2100_pq_transfer import (
    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p284_contract_is_frozen_and_bound() -> None:
    config = json.loads(
        (ROOT / "configs/p284_libavif_gainmap_pq_png_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "FROZEN_AFTER_P283_BEFORE_P284_PUBLICATION"
    assert config["standards"]["cicp_hex"] == "09100001"
    assert config["gates"]["required_row_count"] == 3
    assert config["gates"]["maximum_absolute_light_error_nits"] == 0.07


def test_p284_existing_pq_rail_is_byte_exact_and_strict(tmp_path: Path) -> None:
    values = np.asarray([[[0.0, 203.0, 750.0], [1.0, 10.0, 100.0]]], dtype=np.float32)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_sha, first_samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
        values, first, row_count=1
    )
    second_sha, second_samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
        values, second, row_count=2
    )
    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    assert np.array_equal(first_samples, second_samples)
    assert (
        sha256_rec2100_pq_rgb16_png_samples(first, width=2, height=1)
        == __import__("hashlib").sha256(first_samples.tobytes()).hexdigest()
    )


def test_p284_invalid_publications_are_atomic(tmp_path: Path) -> None:
    assert all(invalid_publication_controls(tmp_path).values())


def test_p284_evidence_binds_one_way_publication_pass() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/P284_LIBAVIF_GAINMAP_PQ_PNG_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "PASS_PRIVATE_LIBAVIF_GAINMAP_PQ_PNG"
    result = evidence["formal_result"]
    assert result["report_sha256"] == (
        "cc4a7e335c777544907291e9cff34eacd5dedac4a67768f5410d4811a0f95b0c"
    )
    assert all(result["gates"].values())
    assert (
        max(row["maximum_absolute_light_error_nits"] for row in result["records"])
        == 0.04371628489445811
    )

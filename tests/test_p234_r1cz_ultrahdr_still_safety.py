from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_p234_r1cz_ultrahdr_still_safety import spatial_vectors, summarize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p234_r1cz_ultrahdr_still_safety_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p234_contract_binds_local_payload_and_consumed_fixtures() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_SELECTED_FIXTURE_DECODE_OR_CANDIDATE_APPLY"
    assert _sha256(ROOT / config["payload"]["capsule_path"]) == config["payload"][
        "capsule_sha256"
    ]
    assert _sha256(ROOT / config["payload"]["p233_config_path"]) == config[
        "payload"
    ]["p233_config_sha256"]
    fixture_root = ROOT / "tests/fixtures/u1_5c_libultrahdr"
    for row in config["source"]["fixtures"]:
        assert _sha256(fixture_root / row["name"]) == row["sha256"]
        assert (row["width"], row["height"]) == (384, 512)


def test_p234_spatial_vectors_detect_horizontal_and_vertical_edges() -> None:
    source = np.ones((3, 4, 3), dtype=np.float32)
    source[:, 2:, :] = 4.0
    luma, chroma = spatial_vectors(
        source,
        luminance_vector=np.array([0.2627, 0.6780, 0.0593]),
        epsilon_nits=1.0,
    )
    assert luma.shape == (17,)
    assert chroma.shape == (17,)
    assert np.count_nonzero(luma) == 3
    assert np.count_nonzero(chroma) == 0


def test_p234_identity_summary_has_no_new_boundaries_or_spikes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = np.linspace(0.25, 1000.0, 6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
    metrics = summarize(source, source.copy(), config)
    assert metrics["new_boundary_fraction"] == 0.0
    assert metrics["median_log_rgb_material_effect"] == 0.0
    assert metrics["luminance_spatial_p95_ratio"] == 1.0
    assert metrics["chroma_spatial_p95_ratio"] == 1.0
    assert metrics["new_luminance_spike_fraction"] == 0.0

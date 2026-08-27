from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.preprocess.dng_profile_huesatmap_audit import (
    parse_profile_huesatmap_exif,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p254_r1dz_dng_huesatmap_overrange_consumer_intake_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f32le_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_stage_a_contract_and_metadata_bindings_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["stage_a_bindings"]
    contract = ROOT / binding["contract_path"]
    exif = ROOT / binding["consumer_dji_exif_path"]

    assert config["status"] == (
        "WAITING_PRODUCER_VERSIONED_CALLABLE_BEFORE_SOURCE_LOCK_OR_EXECUTION"
    )
    assert _sha256(contract) == binding["contract_sha256"]
    assert exif.stat().st_size == binding["consumer_dji_exif_bytes"]
    assert _sha256(exif) == binding["consumer_dji_exif_sha256"]

    profile = parse_profile_huesatmap_exif(exif.read_text(encoding="utf-8"))
    assert profile.dimensions == (6, 6, 3)
    assert profile.encoding == 0
    assert profile.dynamic_range == 0
    assert _f32le_sha256(profile.data1) == binding[
        "consumer_dji_data1_f32le_sha256"
    ]
    assert _f32le_sha256(profile.data2) == binding[
        "consumer_dji_data2_f32le_sha256"
    ]


def test_stage_b_cannot_execute_without_complete_producer_handoff() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = config["stage_b_required_bindings"]

    assert len(required) == len(set(required))
    assert {
        "producer_publication_commit",
        "callable_artifact_path",
        "callable_artifact_git_blob",
        "callable_artifact_sha256",
        "function_signatures",
        "support_overrange_semantics",
        "fixture_input_table_and_output_identities",
        "default_sdr_parent_identities",
    }.issubset(required)
    assert "stage_b_bindings" not in config
    assert config["gates"][
        "require_stage_b_committed_before_expected_output_read"
    ]
    assert config["gates"]["require_no_producer_worktree_import"]
    assert config["gates"]["require_no_source_copy_into_consumer_src"]
    assert "NOT_READY_PRODUCER_HANDOFF_GAP" in config["not_ready_rule"]

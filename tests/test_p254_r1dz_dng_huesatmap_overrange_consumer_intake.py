from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p254_r1ec_dng_profile_huesat_callable_intake import (
    _payload_control,
)
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

    assert config["status"] == "STAGE_B_SOURCE_LOCKED_READY_FOR_FIXTURE_EXECUTION"
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


def test_stage_b_source_lock_is_complete_before_fixture_execution() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = config["stage_b_required_bindings"]
    binding = config["stage_b_bindings"]

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
    source_lock = ROOT / binding["source_lock_path"]
    assert source_lock.stat().st_size == binding["source_lock_bytes"]
    assert _sha256(source_lock) == binding["source_lock_sha256"]
    assert binding["producer_implementation_commit"] == (
        "89d0d71df4d10949b7e5a148ab399021d74dc09c"
    )
    assert set(binding["artifacts"]) == {
        "arithmetic_dependency",
        "callable",
        "contract",
        "evidence",
        "execution_lock",
        "fixture",
        "schema",
    }
    assert binding["invalid_controls"]["block_workspace_capacity"].startswith(
        "not_applicable"
    )
    assert config["gates"][
        "require_stage_b_committed_before_expected_output_read"
    ]
    assert config["gates"]["require_no_producer_worktree_import"]
    assert config["gates"]["require_no_source_copy_into_consumer_src"]
    assert "NOT_READY_PRODUCER_HANDOFF_GAP" in config["not_ready_rule"]


@pytest.mark.parametrize(
    "name",
    [
        "nonfinite-table",
        "wrong-length-table",
        "malformed-dimensions",
        "negative-saturation-scale",
        "negative-value-scale",
        "nonunit-zero-saturation-value-scale",
        "invalid-weight-low",
        "invalid-weight-high",
        "unsupported-encoding",
        "nonboolean-overrange",
        "unexpected-field",
    ],
)
def test_invalid_payload_controls_are_distinct(name: str) -> None:
    payload = {
        "schema": "zhuise.dng-profile-huesat-callable.v1",
        "dimensions": [2, 2, 2],
        "data_1": [0.0, 1.0, 1.0] * 8,
        "data_2": [0.0, 1.0, 1.0] * 8,
        "calibration_1_weight": 0.5,
        "encoding": 0,
        "support_overrange": True,
    }
    candidate = _payload_control(payload, name)
    assert json.dumps(candidate, sort_keys=True) != json.dumps(payload, sort_keys=True)

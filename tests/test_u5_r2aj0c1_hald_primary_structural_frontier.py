from __future__ import annotations

import math
import hashlib
import io
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.haldclut_primary_frontier import (
    EXPECTED_CONFIG_RAW_SHA256,
    _decode_member,
    _representative_key,
    bind_parents,
    build_frontier,
    choose_representatives,
    evaluate_cube,
    load_config_snapshot,
    normalized_safety_margin,
    population_bundle,
    select_diverse,
    source_contract_evidence,
    stream_native_evidence,
    strength_components,
)
from src.eval.haldclut_structural import (
    HaldCube,
    _native_difference_signature,
    array_sha256,
    generate_controls,
    identity_table,
    load_config_snapshot as load_c0_config,
)


ROOT = Path(__file__).resolve().parents[1]
C1_CONFIG = (
    ROOT / "configs/u5_r2aj0c1_hald_primary_structural_frontier_v1.json"
)
C0_CONFIG = (
    ROOT / "configs/u5_r2aj0c0b_hald_structural_controls_v2.json"
)


@pytest.fixture(scope="module")
def c1_config() -> dict:
    config, raw_sha, _canonical_sha = load_config_snapshot(C1_CONFIG)
    assert raw_sha == EXPECTED_CONFIG_RAW_SHA256
    return config


@pytest.fixture(scope="module")
def c0_controls() -> dict[str, np.ndarray]:
    config, _raw, _canonical = load_c0_config(C0_CONFIG)
    return generate_controls(config)


def test_config_parent_runtime_and_universe_bindings(c1_config: dict) -> None:
    parent = bind_parents(ROOT, c1_config)
    assert len(parent["primary_paths"]) == 194
    assert len(parent["primary_records"]) == 194
    assert sum(
        int(row["bytes"]) for row in parent["primary_records"]
    ) == 345233651
    assert sum(
        int(row["compressed_bytes"]) for row in parent["primary_records"]
    ) == 345161729
    assert {int(row["cube_side"]) for row in parent["primary_records"]} == {
        144,
        256,
    }
    assert all(row["mode"] == "RGB" for row in parent["primary_records"])
    evidence = source_contract_evidence(ROOT, c1_config)
    assert all(
        all(row["checks"].values()) for row in evidence.values()
    )


def test_streamed_native_signature_matches_c0_exactly(
    c1_config: dict,
    c0_controls: dict[str, np.ndarray],
) -> None:
    table = c0_controls["warm_strength_100"]
    streamed = stream_native_evidence(
        table,
        interior_margin=c1_config["automatic_gates"][
            "interior_input_margin"
        ],
        endpoint_epsilon=c1_config["automatic_gates"][
            "endpoint_epsilon"
        ],
        slab_rows=3,
    )
    legacy = _native_difference_signature(table)
    assert streamed.signature_sha256 == array_sha256(legacy)
    first_count = sum(
        int(
            np.prod(
                np.diff(np.empty(table.shape), axis=axis).shape
            )
        )
        for axis in range(3)
    )
    assert streamed.maximum_adjacent_code_step == int(
        np.max(np.abs(legacy[:first_count]))
    )
    assert streamed.maximum_second_code_difference == int(
        np.max(np.abs(legacy[first_count:]))
    )


def test_synthetic_zip_member_decode_binds_exact_identity() -> None:
    path = "HaldCLUT/Color/Test/control.png"
    raster = HaldCube(identity_table(4, quantized=True)).to_raster()
    encoded = io.BytesIO()
    Image.fromarray(raster).save(encoded, format="PNG")
    payload = encoded.getvalue()
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as writer:
        writer.writestr(path, payload)
    archive_bytes.seek(0)
    with zipfile.ZipFile(archive_bytes, "r") as archive:
        info = archive.getinfo(path)
        record = {
            "crc32": f"{info.CRC:08x}",
            "bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "width": 8,
            "height": 8,
            "hald_level": 2,
            "cube_side": 4,
        }
        cube, source, byte_count = _decode_member(
            archive, path=path, record=record
        )
    assert byte_count == len(payload)
    assert np.array_equal(cube.table, identity_table(4, quantized=True))
    assert source["archive_member_path"] == path
    assert source["member_bytes_sha256"] == record["sha256"]


def test_evaluator_uses_no_complete_native_float_copy(
    c1_config: dict,
    c0_controls: dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    populations = population_bundle(c1_config)

    def forbidden_normalized(_self):
        raise AssertionError("complete native float copy is forbidden")

    monkeypatch.setattr(HaldCube, "normalized", forbidden_normalized)
    metrics, checks, _signatures, margin = evaluate_cube(
        HaldCube(c0_controls["warm_strength_100"]),
        config=c1_config,
        populations=populations,
    )
    assert all(checks.values())
    assert metrics["style_delta_e76_median"] >= 3.0
    assert metrics["joint_basic_residual_delta_e76_median"] >= 1.5
    assert 0.0 <= margin <= 1.0


def test_native_n144_identity_streaming_is_structurally_safe(
    c1_config: dict,
) -> None:
    table = identity_table(144, quantized=True)
    metrics, checks, _signatures, margin = evaluate_cube(
        HaldCube(table),
        config=c1_config,
        populations=population_bundle(c1_config),
    )
    assert all(checks.values())
    assert metrics["native_maximum_adjacent_code_step"] <= 2
    assert metrics["native_maximum_second_code_difference"] <= 1
    assert 0.0 <= margin <= 1.0


def test_native_n256_streaming_stays_exact_and_bounded(
    c1_config: dict,
) -> None:
    streamed = stream_native_evidence(
        identity_table(256, quantized=True),
        interior_margin=c1_config["automatic_gates"][
            "interior_input_margin"
        ],
        endpoint_epsilon=c1_config["automatic_gates"][
            "endpoint_epsilon"
        ],
    )
    assert streamed.maximum_adjacent_code_step == 1
    assert streamed.maximum_second_code_difference == 0
    assert streamed.endpoint_count == 0
    assert len(streamed.signature_sha256) == 64


def test_structural_negative_remains_vetoed(
    c1_config: dict,
    c0_controls: dict[str, np.ndarray],
) -> None:
    metrics, checks, _signatures, _margin = evaluate_cube(
        HaldCube(c0_controls["negative_hard_clip"]),
        config=c1_config,
        populations=population_bundle(c1_config),
    )
    assert not all(checks.values())
    assert not checks["new_interior_endpoint_fraction"]
    assert (
        metrics["new_interior_endpoint_fraction"]
        > c1_config["automatic_gates"][
            "maximum_new_interior_endpoint_fraction"
        ]
    )


def _minimal_record(
    candidate_id: str,
    native_hash: str,
    *,
    style: float = 5.0,
    residual: float = 3.0,
    safety: float = 0.5,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "hashes": {"native_table_array_sha256": native_hash},
        "metrics": {
            "style_delta_e76_median": style,
            "joint_basic_residual_delta_e76_median": residual,
        },
        "normalized_safety_margin": safety,
        "source": {"path_identity_sha256": candidate_id},
    }


def test_nontransitive_strength_chain_is_not_single_link_collapsed(
    c1_config: dict,
) -> None:
    ids = ["a" * 64, "b" * 64, "c" * 64]
    records = {
        candidate_id: _minimal_record(
            candidate_id, f"{index + 1:064x}"
        )
        for index, candidate_id in enumerate(ids)
    }
    angles = [0.0, 5.0, 10.0]
    effects = {
        candidate_id: np.array(
            [[math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0]]
        )
        for candidate_id, angle in zip(ids, angles, strict=True)
    }
    components, _pairs, ambiguities = strength_components(
        ordered_ids=ids,
        records=records,
        effects=effects,
        config=c1_config,
    )
    reverse, _pairs_reverse, ambiguities_reverse = strength_components(
        ordered_ids=list(reversed(ids)),
        records=records,
        effects=effects,
        config=c1_config,
    )
    assert components == [[value] for value in ids]
    assert reverse == components
    assert len(ambiguities) == 1
    assert ambiguities_reverse == ambiguities
    assert ambiguities[0]["decision"] == "retain_exact_groups"


def test_representative_prefers_style_then_nonbasic_then_safety() -> None:
    ids = ["a" * 64, "b" * 64, "c" * 64]
    records = {
        ids[0]: _minimal_record(ids[0], "1" * 64, style=6, residual=4),
        ids[1]: _minimal_record(ids[1], "2" * 64, style=7, residual=3),
        ids[2]: _minimal_record(ids[2], "3" * 64, style=7, residual=5),
    }
    assert _representative_key(records[ids[2]]) < _representative_key(
        records[ids[1]]
    )
    chosen = choose_representatives([ids], records)
    assert chosen[0]["representative"] == ids[2]


def test_diversity_selection_is_deterministic_and_keeps_singleton(
    c1_config: dict,
) -> None:
    ids = ["a" * 64, "b" * 64, "c" * 64]
    records = {
        ids[0]: _minimal_record(ids[0], "1" * 64, style=9),
        ids[1]: _minimal_record(ids[1], "2" * 64, style=8),
        ids[2]: _minimal_record(ids[2], "3" * 64, style=7),
    }
    representatives = [
        {
            "component_id": value,
            "members": [value],
            "representative": value,
        }
        for value in ids
    ]
    signatures = {
        ids[0]: np.zeros((2, 3)),
        ids[1]: np.full((2, 3), 2.0),
        ids[2]: np.full((2, 3), 0.1),
    }
    selected, distances, trace = select_diverse(
        representatives=representatives,
        records=records,
        residual_signatures=signatures,
        config=c1_config,
    )
    assert selected == ids[:2]
    assert len(distances) == 3
    assert len(trace) == 2
    singleton, _distances, _trace = select_diverse(
        representatives=representatives[:1],
        records=records,
        residual_signatures=signatures,
        config=c1_config,
    )
    assert singleton == ids[:1]


def test_safety_margin_is_bounded_and_deterministic(c1_config: dict) -> None:
    gates = c1_config["automatic_gates"]
    metrics = {
        "new_interior_endpoint_fraction": 0.0,
        "negative_jacobian_fraction": 0.0,
        "minimum_jacobian_determinant": 1.0,
        "maximum_jacobian_spectral_norm": 1.0,
        "neutral_luma_reversal": 0.0,
        "gradient_rgb_gain_p99": 1.0,
        "maximum_gradient_rgb_gain": 1.0,
        "maximum_gradient_channel_second_difference": 0.0,
        "native_maximum_adjacent_code_step": 1,
        "native_maximum_second_code_difference": 1,
    }
    margin = normalized_safety_margin(metrics, gates)
    assert 0.0 < margin <= 1.0
    boundary = deepcopy(metrics)
    boundary["native_maximum_second_code_difference"] = 32
    assert normalized_safety_margin(boundary, gates) == 0.0


def test_runner_rejects_wrong_commit_before_output(
    c1_config: dict,
) -> None:
    output = (
        ROOT / "outputs/_test_u5_r2aj0c1_wrong_commit_must_not_exist"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/run_u5_r2aj0c1_hald_primary_structural_frontier.py"
            ),
            "--single-run",
            "--output",
            str(output),
            "--expected-config-sha256",
            EXPECTED_CONFIG_RAW_SHA256,
            "--software-commit",
            "0" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "software commit must equal" in completed.stderr
    assert not output.exists()


def test_build_frontier_empty_and_singleton_branches(c1_config: dict) -> None:
    empty = build_frontier(
        records=[],
        effects={},
        residual_signatures={},
        config=c1_config,
    )
    assert not empty["frontier_pass"]
    assert empty["frontier_claim"] == "no_structural_challenger"
    assert "photographic_contract_design_allowed" not in empty

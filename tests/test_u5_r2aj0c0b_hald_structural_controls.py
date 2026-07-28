from __future__ import annotations

import inspect
import json
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.eval.haldclut_structural import (
    EXPECTED_CONFIG_CANONICAL_SHA256,
    HaldCube,
    HaldStructuralError,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
    evaluate_controls,
    finalize_repeat_evidence,
    generate_controls,
    identity_table,
    load_config_snapshot,
    run_single_control_process,
    strict_json_loads,
    validate_config,
    validate_manifest,
    validate_repeat_decision,
    validate_report,
)
from src.roll2film.lut import DenseLUT3D


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/u5_r2aj0c0b_hald_structural_controls_v2.json"


@pytest.fixture(scope="module")
def config_bundle() -> tuple[dict, str, str]:
    return load_config_snapshot(CONFIG_PATH)


@pytest.fixture(scope="module")
def evaluated(config_bundle: tuple[dict, str, str]) -> tuple[dict, dict]:
    return evaluate_controls(config_bundle[0], root=ROOT)


def test_frozen_config_and_independent_id_hashes(
    config_bundle: tuple[dict, str, str],
) -> None:
    config, _raw_sha, canonical_sha = config_bundle
    assert canonical_sha == EXPECTED_CONFIG_CANONICAL_SHA256
    ids = [row["id"] for row in config["control_generation"]["controls"]]
    assert canonical_sha256(ids) == config["control_generation"][
        "ordered_control_ids_sha256"
    ]
    assert canonical_sha256(config["control_generation"]["controls"]) == config[
        "independent_expected_hashes"
    ]["control_definitions_canonical_sha256"]
    assert len(ids) == len(set(ids)) == 16


def test_config_mutation_or_unknown_key_fails_closed(
    config_bundle: tuple[dict, str, str],
) -> None:
    mutated = deepcopy(config_bundle[0])
    mutated["unknown"] = True
    with pytest.raises(HaldStructuralError, match="differs"):
        validate_config(mutated)


def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(HaldStructuralError, match="duplicate"):
        strict_json_loads(b'{"a":1,"a":2}')
    with pytest.raises(HaldStructuralError, match="non-finite"):
        strict_json_loads(b'{"a":NaN}')
    with pytest.raises(HaldStructuralError, match="finite"):
        canonical_json_bytes({"a": float("inf")})


def test_hald_red_fastest_raster_roundtrip_and_illegal_geometry() -> None:
    table = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    for red in range(4):
        for green in range(4):
            for blue in range(4):
                table[red, green, blue] = [
                    red * 64 + green * 8 + blue,
                    red,
                    blue,
                ]
    cube = HaldCube(table)
    raster = cube.to_raster()
    flat = raster.reshape(-1, 3)
    assert np.array_equal(flat[1], table[1, 0, 0])
    assert np.array_equal(flat[4], table[0, 1, 0])
    assert np.array_equal(flat[16], table[0, 0, 1])
    assert np.array_equal(HaldCube.from_raster(raster).table, table)
    with pytest.raises(HaldStructuralError, match="integer cube"):
        HaldCube.from_raster(np.zeros((9, 9, 3), dtype=np.uint8))
    with pytest.raises(HaldStructuralError, match="square"):
        HaldCube.from_raster(np.zeros((8, 7, 3), dtype=np.uint8))


def test_streaming_trilinear_matches_small_dense_oracle() -> None:
    rng = np.random.default_rng(2026072801)
    table = rng.integers(0, 256, size=(4, 4, 4, 3), dtype=np.uint8)
    points = rng.random((4096, 3))
    actual = HaldCube(table).apply(points, batch_rows=37)
    expected = DenseLUT3D(
        table.astype(np.float64) / 255.0,
        np.zeros(3),
        np.ones(3),
        "trilinear",
    ).apply(points)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-16)


def test_generated_controls_match_all_independent_table_hashes(
    config_bundle: tuple[dict, str, str],
) -> None:
    config = config_bundle[0]
    controls = generate_controls(config)
    assert {name: array_sha256(value) for name, value in controls.items()} == config[
        "independent_expected_hashes"
    ]["control_source_tables"]
    assert np.array_equal(
        controls["warm_strength_100"],
        controls["warm_strength_100_duplicate"],
    )


def test_control_evaluation_passes_exact_roles_and_components(
    evaluated: tuple[dict, dict],
) -> None:
    _manifest, report = evaluated
    assert report["all_control_expectations_pass"]
    assert all(report["checks"].values())
    assert report["eligible_control_ids"] == [
        "warm_strength_050",
        "warm_strength_075",
        "warm_strength_100",
        "warm_strength_100_duplicate",
        "cool_strength_100",
    ]
    expected = [
        ["cool_strength_100"],
        [
            "warm_strength_050",
            "warm_strength_075",
            "warm_strength_100",
            "warm_strength_100_duplicate",
        ],
    ]
    assert all(
        components == expected
        for components in report["equivalence"]["components_by_order"].values()
    )
    assert report["equivalence"]["novelty"][0]["check"]
    assert (
        report["equivalence"]["novelty"][0][
            "residual_signature_delta_e76_median"
        ]
        > 8.0
    )


def test_negative_controls_fail_only_as_controls_not_as_bank_veto(
    evaluated: tuple[dict, dict],
) -> None:
    _manifest, report = evaluated
    records = {row["control_id"]: row for row in report["records"]}
    expected_failed_checks = {
        "negative_axis_swap_rg": {
            "negative_jacobian_fraction",
            "minimum_jacobian_determinant",
        },
        "negative_hard_clip": {"new_interior_endpoint_fraction"},
        "negative_staircase_7": {"native_maximum_second_code_difference"},
        "negative_single_cell_spike": {
            "minimum_jacobian_determinant",
            "maximum_jacobian_spectral_norm",
            "maximum_gradient_rgb_gain",
            "neutral_luma_reversal",
            "native_maximum_adjacent_code_step",
            "native_maximum_second_code_difference",
        },
    }
    for control_id, intended in expected_failed_checks.items():
        record = records[control_id]
        failed = {key for key, value in record["safety_checks"].items() if not value}
        assert failed.intersection(intended)
        assert not record["structurally_safe"]
        assert not record["eligible"]
        assert record["expectation_pass"]
    assert all(
        row["expectation_pass"]
        for row in report["records"]
        if not row["control_id"].startswith("negative_")
    )


def test_oracles_and_precision_conformance_are_exact(
    evaluated: tuple[dict, dict],
) -> None:
    manifest, _report = evaluated
    assert manifest["hald_conformance"]["all_checks_pass"]
    assert {
        row["cube_side"]
        for row in manifest["hald_conformance"]["records"]
    } == {4, 36, 144, 256}
    assert manifest["jacobian_conformance"]["all_checks_pass"]
    assert (
        manifest["jacobian_conformance"][
            "finite_difference_maximum_absolute_error"
        ]
        < 1e-12
    )
    assert manifest["rgb16_tiff_conformance"]["all_checks_pass"]
    assert (
        manifest["rgb16_tiff_conformance"]["source_array_sha256"]
        == manifest["rgb16_tiff_conformance"]["decoded_array_sha256"]
    )


def test_evaluator_api_has_no_archive_path_and_external_open_is_forbidden(
    config_bundle: tuple[dict, str, str],
) -> None:
    assert set(inspect.signature(evaluate_controls).parameters) == {
        "config",
        "root",
    }
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        normalized = path.as_posix().casefold()
        if "/data/external/" in normalized:
            raise AssertionError("external data root was opened")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", guarded_open), patch.object(
        zipfile, "ZipFile", side_effect=AssertionError("ZIP was opened")
    ):
        manifest, report = evaluate_controls(config_bundle[0], root=ROOT)
    assert manifest["hald_conformance"]["all_checks_pass"]
    assert report["all_control_expectations_pass"]


def test_manifest_and_report_validators_reject_adversarial_fields(
    config_bundle: tuple[dict, str, str],
    evaluated: tuple[dict, dict],
) -> None:
    config = config_bundle[0]
    common = {
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "state": "control-run-only-non-promoting",
        "software_commit": "a" * 40,
        "tracked_worktree_clean": True,
        "config": {
            "path": "configs/u5_r2aj0c0b_hald_structural_controls_v2.json",
            "raw_sha256_before": config_bundle[1],
            "raw_sha256_after": config_bundle[1],
            "canonical_sha256": config_bundle[2],
        },
        "parent_evidence": {
            "aj0b2": {
                "decision_path": config["parent_evidence"]["decision_path"],
                "decision_sha256": config["parent_evidence"]["decision_sha256"],
                "decision": config["parent_evidence"]["required_decision"],
                "result_commit": config["parent_evidence"]["result_commit"],
                "archive_sha256": config["parent_evidence"]["archive_sha256"],
                "primary_paths_sha256": config["parent_evidence"][
                    "primary_paths_sha256"
                ],
            },
            "c0_v1_close": {
                "decision_path": config["supersedes"]["close_decision_path"],
                "decision_sha256": (
                    "a655f1bf6b0707ae3ab55fd9f02b970320598df210c62de7a30576699c3dd98d"
                ),
                "decision": config["supersedes"]["required_close_decision"],
                "zero_access": True,
            },
        },
        "runtime": config["runtime"]["packages"]
        | {"python": config["runtime"]["python"]},
        "ordered_control_ids_sha256": config["control_generation"][
            "ordered_control_ids_sha256"
        ],
        "access_ledger": {
            "archive_open_calls": 0,
            "archive_bytes_read": 0,
            "primary_open_calls": 0,
            "primary_cluts_decoded": 0,
            "primary_candidate_metrics": 0,
            "external_root_control_pixels": 0,
            "photograph_renders": 0,
        },
        "primary_evaluation_performed": False,
        "photograph_rendered": False,
        "visual_review_allowed": False,
        "evidence_promoted": False,
        "c1_contract_design_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    manifest = {
        "schema_version": "u5-r2aj0c0b-hald-structural-control-manifest-v2",
        **common,
        **evaluated[0],
    }
    report = {
        "schema_version": "u5-r2aj0c0b-hald-structural-control-report-v2",
        **common,
        **evaluated[1],
    }
    validate_manifest(manifest, config)
    validate_report(report, config)
    bad_root = deepcopy(report)
    bad_root["automatic_pass"] = True
    with pytest.raises(HaldStructuralError, match="keys differ"):
        validate_report(bad_root, config)
    bad_hash = deepcopy(report)
    bad_hash["records"][0]["hashes"]["unknown"] = "0" * 64
    with pytest.raises(HaldStructuralError, match="control hashes"):
        validate_report(bad_hash, config)
    bad_bool = deepcopy(report)
    bad_bool["access_ledger"]["archive_bytes_read"] = False
    with pytest.raises(HaldStructuralError, match="integer zero"):
        validate_report(bad_bool, config)


def test_two_process_evidence_is_exact_and_only_repeat_can_open_c1(
    tmp_path: Path,
    config_bundle: tuple[dict, str, str],
) -> None:
    config, raw_sha, _canonical_sha = config_bundle
    commit = "b" * 40
    first = tmp_path / "run_a"
    second = tmp_path / "run_b"
    for directory in (first, second):
        run_single_control_process(
            root=ROOT,
            config_path=CONFIG_PATH,
            output_dir=directory,
            expected_config_raw_sha256=raw_sha,
            software_commit=commit,
        )
        child = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        assert "automatic_pass" not in child
        assert not child["c1_contract_design_allowed"]
    decision = finalize_repeat_evidence(
        first_dir=first,
        second_dir=second,
        config=config,
        config_raw_sha256=raw_sha,
        software_commit=commit,
        root=ROOT,
    )
    assert decision["automatic_pass"]
    assert decision["c1_contract_design_allowed"]
    assert not decision["primary_evaluation_allowed"]
    validate_repeat_decision(decision, config)
    wrong_label = deepcopy(decision)
    wrong_label["decision"] = "close_controls_or_repeat_failure"
    with pytest.raises(HaldStructuralError, match="contradicts"):
        validate_repeat_decision(wrong_label, config)
    wrong_config = deepcopy(decision)
    wrong_config["config"]["raw_sha256"] = "0" * 63
    with pytest.raises(HaldStructuralError, match="config binding"):
        validate_repeat_decision(wrong_config, config)


def test_repeat_reconstruction_rejects_same_mutation_in_both_children(
    tmp_path: Path,
    config_bundle: tuple[dict, str, str],
) -> None:
    config, raw_sha, _canonical_sha = config_bundle
    commit = "c" * 40
    directories = [tmp_path / "a", tmp_path / "b"]
    for directory in directories:
        run_single_control_process(
            root=ROOT,
            config_path=CONFIG_PATH,
            output_dir=directory,
            expected_config_raw_sha256=raw_sha,
            software_commit=commit,
        )
        report_path = directory / "report.json"
        report = strict_json_loads(report_path.read_bytes())
        report["records"][0]["metrics"]["output_minimum"] = 0.125
        report_path.write_bytes(canonical_json_bytes(report))
    decision = finalize_repeat_evidence(
        first_dir=directories[0],
        second_dir=directories[1],
        config=config,
        config_raw_sha256=raw_sha,
        software_commit=commit,
        root=ROOT,
    )
    assert not decision["automatic_pass"]
    assert not decision["c1_contract_design_allowed"]


def test_runner_rejects_a_software_commit_that_is_not_head(
    tmp_path: Path,
    config_bundle: tuple[dict, str, str],
) -> None:
    forbidden_output = (
        ROOT / "outputs" / f"_test_wrong_commit_{tmp_path.name}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/run_u5_r2aj0c0b_hald_structural_controls.py"
            ),
            "--single-run",
            "--output",
            str(forbidden_output),
            "--expected-config-sha256",
            config_bundle[1],
            "--software-commit",
            "0" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "software commit must equal" in completed.stderr
    assert not forbidden_output.exists()


def test_large_identity_table_stays_uint8() -> None:
    table = identity_table(256, quantized=True)
    assert table.dtype == np.uint8
    assert table.nbytes == 256**3 * 3

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

from scripts.audit_u7_8c_canon_real_scale_input_batch import (
    _controller_report,
    _manifest_payload,
    _require_relative_posix_path,
    _run_controller,
    _stable_payload,
    _validate_file_binding,
)
from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_8c_canon_real_scale_input_batch_v1.json"
CONTRACT = ROOT / "docs/planning/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_CONTRACT.md"
EVIDENCE = ROOT / "docs/evidence/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_RESULT.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _worker() -> dict:
    return {
        "schema_version": "neuro-film.u7-8c-canon-real-scale-input-batch-result.v1",
        "node_id": "U7.8C",
        "requested_order": "forward",
        "implementation_commit": "1" * 40,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "source_identity": [{"job_id": "a", "sha256": "4" * 64}],
        "source_set_identity": "5" * 64,
        "transaction": {"batch_id": "6" * 64, "rows": []},
        "decode_counts": {"a": 1},
        "per_job_wall_seconds": [{"job_id": "a", "wall_seconds": 8.0}],
        "controls": {
            "injected_failure_decode_counts": {"a": 1},
            "injected_second_child_published_nothing": True,
            "late_foreign_decode_counts": {"a": 1},
            "late_foreign_complete_batch_preserved": True,
        },
        "gate_results": {
            "all_six_sources_exact_and_immutable": True,
            "successful_decode_calls_exactly_once_per_input": True,
            "six_children_eighteen_outputs_and_recipes": True,
            "injected_second_child_publishes_nothing": True,
            "failure_controls_have_no_extra_decodes": True,
            "complete_batch_late_foreign_preserved": True,
            "owned_stage_and_file_residue_count": 0,
            "network_requests": 0,
        },
        "claim_ceiling": "bounded",
    }


def test_frozen_config_binds_six_sources_and_unchanged_product_core() -> None:
    config = _config()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    commit = evidence["bindings"]["implementation_commit"]
    core_bindings = {
        "input_batch_core": evidence["bindings"]["u7_8a_transaction_core"],
        "child_batch_core": evidence["bindings"]["three_look_child_core"],
    }
    for label, row in config["bindings"].items():
        if label in core_bindings:
            archived = subprocess.check_output(
                ["git", "show", f"{commit}:{row['path']}"], cwd=ROOT
            )
            assert len(archived) == row["bytes"]
            assert hashlib.sha256(archived).hexdigest() == row["sha256"]
            archived_blob = subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{row['path']}"],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
            ).strip()
            assert archived_blob == core_bindings[label]["git_blob"]
        else:
            _validate_file_binding(row, label=label)
    for row in config["sources"]:
        _validate_file_binding(row, label=row["job_id"])
    assert config["node_id"] == "U7.8C"
    assert len(config["sources"]) == 6
    assert config["formal"]["expected_output_count"] == 18
    assert config["formal"]["expected_recipe_count"] == 18
    assert config["resource_gates"] == {
        "maximum_process_tree_rss_bytes": 4 * 1024**3,
        "maximum_controller_wall_seconds": 1200.0,
        "per_job_wall_seconds": "report_only",
    }
    assert CONTRACT.is_file()
    for row in config["sources"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256_file(path) == row["sha256"]


def test_manifest_orders_differ_but_bind_the_same_six_jobs() -> None:
    config = _config()
    forward = _manifest_payload(config, "forward")
    reverse = _manifest_payload(config, "reverse")
    assert forward["jobs"] == list(reversed(reverse["jobs"]))
    assert {row["job_id"] for row in forward["jobs"]} == {
        row["job_id"] for row in config["sources"]
    }


def test_stable_payload_excludes_order_timing_and_rss() -> None:
    first = _worker()
    second = _worker()
    second["requested_order"] = "reverse"
    second["per_job_wall_seconds"] = [{"job_id": "a", "wall_seconds": 99.0}]
    second["resource_measurement"] = {
        "controller_wall_seconds": 100.0,
        "process_tree_peak_rss_bytes": 200,
    }
    assert _stable_payload(first) == _stable_payload(second)


def test_aggregate_paths_require_canonical_relative_posix() -> None:
    assert (
        _require_relative_posix_path("canon-eos-50d-mraw/velvia_50.png", label="output")
        == "canon-eos-50d-mraw/velvia_50.png"
    )
    for value in (
        "C:/private/output.png",
        "C:\\private\\output.png",
        "/private/output.png",
        "../escape.png",
        "job/../escape.png",
        "job\\output.png",
    ):
        try:
            _require_relative_posix_path(value, label="output")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"machine-local path accepted: {value}")


def test_controller_applies_frozen_resource_gates_without_mutating_stable_data() -> (
    None
):
    worker = _worker()
    passed = _controller_report(
        worker,
        wall_seconds=1199.0,
        peak_rss_bytes=4 * 1024**3,
        scratch_root_absent=True,
    )
    assert passed["decision"] == "PASS_PRIVATE_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"
    assert all(passed["resource_gate_results"].values())
    expected_stable = _stable_payload(worker)
    expected_stable["gate_results"] = {
        **expected_stable["gate_results"],
        "scratch_root_absent_after_worker": True,
    }
    assert _stable_payload(passed) == expected_stable

    failed = _controller_report(
        worker,
        wall_seconds=1200.01,
        peak_rss_bytes=4 * 1024**3 + 1,
        scratch_root_absent=True,
    )
    assert failed["decision"] == "FAIL_CLOSED_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"
    assert failed["resource_gate_results"] == {
        "controller_wall_seconds": False,
        "process_tree_peak_rss_bytes": False,
    }


def test_false_boolean_or_cleanup_residue_cannot_pass_as_numeric_zero() -> None:
    false_gate = _worker()
    false_gate["gate_results"]["all_six_sources_exact_and_immutable"] = False
    report = _controller_report(
        false_gate,
        wall_seconds=1.0,
        peak_rss_bytes=1,
        scratch_root_absent=True,
    )
    assert report["decision"] == "FAIL_CLOSED_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"

    cleanup_failure = _controller_report(
        _worker(),
        wall_seconds=1.0,
        peak_rss_bytes=1,
        scratch_root_absent=False,
    )
    assert cleanup_failure["decision"] == (
        "FAIL_CLOSED_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"
    )
    assert cleanup_failure["gate_results"]["scratch_root_absent_after_worker"] is False


def test_controller_redirects_worker_json_without_pipe_backpressure() -> None:
    source = inspect.getsource(_run_controller)
    assert "stdout=subprocess.PIPE" not in source
    assert 'stdout_path = controller_temp / "worker.stdout.json"' in source
    assert "stdout=stdout_handle" in source
    assert "stderr=stderr_handle" in source

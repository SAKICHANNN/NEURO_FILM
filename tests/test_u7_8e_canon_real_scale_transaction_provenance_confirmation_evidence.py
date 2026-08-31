from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION_RESULT.json"
)


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git_blob(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def test_evidence_binds_exact_implementation_and_execution_lock() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_U7_8E_PREFLIGHT_RUNTIME_ASSET_MATERIALIZATION_DRIFT"
    )
    implementation_commit = evidence["bindings"]["implementation_commit"]
    execution_lock_commit = evidence["bindings"]["execution_lock_commit"]
    for commit in (implementation_commit, execution_lock_commit):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=True
        )

    for path, row in evidence["bindings"]["implementation"].items():
        payload = _git_bytes(implementation_commit, path)
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert _git_blob(implementation_commit, path) == row["git_blob"]

    lock = evidence["bindings"]["execution_lock"]
    payload = _git_bytes(execution_lock_commit, lock["path"])
    assert len(payload) == lock["bytes"]
    assert hashlib.sha256(payload).hexdigest() == lock["sha256"]
    assert _git_blob(execution_lock_commit, lock["path"]) == lock["git_blob"]


def test_exact_line_ending_mismatch_precedes_primary_render() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    root_cause = evidence["root_cause"]
    payload = subprocess.check_output(
        ["git", "cat-file", "blob", root_cause["git_blob"]], cwd=ROOT
    )
    assert len(payload) == root_cause["git_bytes"]
    assert hashlib.sha256(payload).hexdigest() == root_cause["git_sha256"]
    assert payload.count(b"\n") == root_cause["git_lf_count"]
    assert payload.count(b"\r") == root_cause["git_cr_count"]

    runtime = payload.replace(b"\n", b"\r\n")
    assert len(runtime) == root_cause["required_checkout_runtime_bytes"]
    assert (
        hashlib.sha256(runtime).hexdigest()
        == root_cause["required_checkout_runtime_sha256"]
    )
    assert runtime.count(b"\n") == root_cause["required_checkout_runtime_lf_count"]
    assert runtime.count(b"\r") == root_cause["required_checkout_runtime_cr_count"]

    runner_path = (
        "scripts/audit_u7_8e_canon_real_scale_transaction_provenance_confirmation.py"
    )
    runner = _git_bytes(evidence["bindings"]["implementation_commit"], runner_path)
    worker = runner.split(b"def _run_worker", 1)[1]
    assert worker.index(b"_validate_runtime_root_asset_ledger(") < worker.index(
        b"_run_primary_with_provenance("
    )


def test_failure_is_predecode_atomic_and_parent_results_stay_immutable() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    attempts = evidence["formal_attempts"]
    assert attempts["forward_controller_count"] == 1
    assert attempts["forward_worker_count"] == 1
    assert attempts["reverse_controller_count"] == 0
    assert attempts["reverse_worker_count"] == 0
    assert attempts["formal_report_count"] == 0
    assert attempts["source_pixel_decode_count"] == 0
    assert attempts["render_count"] == 0
    assert attempts["published_transaction_count"] == 0
    assert attempts["owned_scratch_absent_after_failure"] is True
    assert attempts["formal_output_root_absent_after_failure"] is True

    assert not (
        ROOT
        / "outputs/eval/u7_8e_canon_real_scale_transaction_provenance_confirmation_v1"
    ).exists()
    parents = evidence["bindings"]["parents"]
    assert (
        sha256_file(
            ROOT / "docs/evidence/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_RESULT.json"
        )
        == parents["u7_8c_evidence_sha256"]
    )
    assert (
        sha256_file(
            ROOT
            / "docs/evidence/U7_8D_TRANSACTION_SOFTWARE_PROVENANCE_SNAPSHOT_RESULT.json"
        )
        == parents["u7_8d_evidence_sha256"]
    )
    assert evidence["stop_rule"] == {
        "runner_or_core_corrected_after_formal_failure": False,
        "forward_rerun": False,
        "reverse_run": False,
        "line_ending_or_hash_normalization_rescue": False,
        "source_config_path_threshold_or_output_changed": False,
        "u7_8c_or_u7_8d_rewritten": False,
        "promotion_opened": False,
    }

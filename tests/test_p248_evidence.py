from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P248_ACESCG_OPENEXR_SCANLINE_STREAMING_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_payload(report: dict[str, object]) -> bytes:
    scientific = copy.deepcopy(report["scientific"])
    assert isinstance(scientific, dict)
    build = scientific["build"]
    assert isinstance(build, dict)
    build.pop("binary_bytes")
    build.pop("binary_sha256")
    workers = scientific["workers"]
    assert isinstance(workers, list)
    for worker in workers:
        assert isinstance(worker, dict)
        worker.pop("resource")
    return json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()


def test_p248_evidence_binds_reports_and_stable_payload() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = []
    for binding in evidence["formal_reports"]:
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    first = _stable_payload(reports[0])
    second = _stable_payload(reports[1])
    assert first == second
    assert hashlib.sha256(first).hexdigest() == evidence["result"][
        "stable_identity_excluding_resource_measurements_and_native_binary_container"
    ]


def test_p248_pass_is_bounded_to_frozen_resource_and_correctness_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_ACESCG_OPENEXR_24MP_SCANLINE_STREAMING"
    assert all(evidence["gates"].values())
    result = evidence["result"]
    assert result["maximum_worker_peak_process_tree_rss_bytes"] <= result[
        "frozen_maximum_rss_bytes"
    ]
    assert result["decoded_maximum_absolute_error"] == 0.0
    assert result["new_boundary_count"] == 0
    assert result["owned_workspace_residue"] == 0
    assert not result["native_rebuild_binary_sha_exact"]
    assert result["maximum_rss_ratio_vs_p247_maximum"] == (
        result["maximum_worker_peak_process_tree_rss_bytes"]
        / result["p247_maximum_worker_peak_rss_bytes"]
    )
    assert "public package/schema/capability" in evidence["claim_ceiling"]

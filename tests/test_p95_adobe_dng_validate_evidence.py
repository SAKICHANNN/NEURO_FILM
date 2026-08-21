from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p95_evidence_binds_two_distinct_fail_closed_reports() -> None:
    evidence = json.loads(
        Path("docs/evidence/P95_ADOBE_DNG_VALIDATE_RUNTIME_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    forward = evidence["bindings"]["forward_report"]
    reverse = evidence["bindings"]["reverse_report"]
    assert _sha256(Path(forward["path"])) == forward["sha256"]
    assert _sha256(Path(reverse["path"])) == reverse["sha256"]
    assert forward["sha256"] != reverse["sha256"]
    assert evidence["status"] == "FAIL_CLOSED_ADOBE_DNG_VALIDATE_RUNTIME"
    assert not all(evidence["gates"].values())


def test_p95_failure_is_mandatory_row_and_replay_bounded() -> None:
    evidence = json.loads(
        Path("docs/evidence/P95_ADOBE_DNG_VALIDATE_RUNTIME_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = evidence["metrics"]
    assert metrics["mandatory_rows"] == 5
    assert metrics["successful_rows"] == 2
    assert metrics["rejected_rows"] == 3
    assert metrics["blackmagic_returncode"] == 106
    assert metrics["stage2_file_hash_exact_on_all_four_rendered_rows"]
    assert metrics["stage3_file_hash_exact_on_all_four_rendered_rows"]
    assert metrics["final_decoded_pixel_hash_exact_on_all_four_rendered_rows"]
    assert not metrics["final_file_hash_exact_on_all_four_rendered_rows"]

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p94_evidence_binds_exact_report_and_passes_all_gates() -> None:
    evidence = json.loads(
        Path("docs/evidence/P94_DNG_FORWARD_MATRIX_MECHANICS_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    report_binding = evidence["bindings"]["raw_report"]
    report = json.loads(Path(report_binding["path"]).read_text(encoding="utf-8"))
    assert _sha256(Path(report_binding["path"])) == report_binding["sha256"]
    assert report["stable_identity_sha256"] == report_binding["stable_identity_sha256"]
    assert evidence["status"] == "PASS_PRIVATE_DNG_FORWARD_MATRIX_MECHANICS"
    assert all(evidence["gates"].values())
    assert report["metrics"]["raster_sample_rgb_decode_calls"] == 0


def test_p94_four_formal_reports_are_byte_exact() -> None:
    root = Path("outputs/eval/p94_dng_forward_matrix_mechanics")
    paths = [
        root / "outer1_forward.json",
        root / "outer1_reverse.json",
        root / "outer2_forward.json",
        root / "outer2_reverse.json",
    ]
    assert len({_sha256(path) for path in paths}) == 1

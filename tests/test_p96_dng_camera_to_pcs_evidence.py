from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p96_evidence_binds_exact_reports_and_fails_only_build_gate() -> None:
    evidence = json.loads(
        Path("docs/evidence/P96_DNG_CAMERA_TO_PCS_PORTABLE_ABI_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    forward = evidence["bindings"]["forward_report"]
    reverse = evidence["bindings"]["reverse_report"]
    assert _sha256(Path(forward["path"])) == forward["sha256"]
    assert _sha256(Path(reverse["path"])) == reverse["sha256"]
    assert forward["sha256"] == reverse["sha256"]
    assert evidence["status"] == "FAIL_CLOSED_DNG_CAMERA_TO_PCS_PORTABLE_ABI"
    assert [name for name, passed in evidence["gates"].items() if not passed] == [
        "independent_builds"
    ]


def test_p96_arithmetic_and_atomicity_are_exact_component_facts() -> None:
    evidence = json.loads(
        Path("docs/evidence/P96_DNG_CAMERA_TO_PCS_PORTABLE_ABI_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = evidence["metrics"]
    assert evidence["execution"]["transformed_triplet_count"] == 1285
    assert metrics["maximum_python_native_absolute_error"] == 0.0
    assert metrics["maximum_msvc_llvm_absolute_error"] == 0.0
    assert metrics["failure_atomicity_cases_passed"] == 5
    assert metrics["msvc_independent_dll_hash_exact"]
    assert not metrics["llvm_independent_dll_hash_exact"]

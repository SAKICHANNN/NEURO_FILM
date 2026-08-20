from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_3D_DNG_LIBRAW_METADATA_CONFORMANCE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_3d_structural_failure_binds_frozen_files() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_BEFORE_LIBRAW_COMPARISON_DNG_RECEIPT_SCOPE"
    for key in ("contract", "config", "comparison_core", "runner"):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_u1_3d_stops_without_default_or_pixel_rescue() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    execution = evidence["execution"]
    assert execution["same_structural_error_across_opposite_ends"] is True
    assert execution["formal_report_files_written"] == 0
    assert execution["libraw_inspection_calls"] == 0
    assert execution["raw_sample_reads"] == 0
    assert execution["rgb_postprocess_calls"] == 0
    assert evidence["failed_gate"]["post_result_default_semantics_rescue"] is False


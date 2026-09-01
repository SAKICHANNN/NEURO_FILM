from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5F_RAW_HALF_SIZE_PREVIEW_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u4_5f_evidence_binds_exact_failed_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = []
    for role in ("forward", "reverse"):
        binding = evidence["formal_reports"][role]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == binding["status"]
        assert report["scientific_identity"] == binding["scientific_identity"]
        assert report["summary"] == binding["summary"]
        assert [key for key, value in report["gates"].items() if not value] == [
            "candidate_to_full_wall_ratio"
        ]
        reports.append(report)

    assert reports[0]["scientific_identity"] == reports[1]["scientific_identity"]
    assert all(
        left["candidate"]["output_hashes"] == right["candidate"]["output_hashes"]
        and left["full"]["output_hashes"] == right["full"]["output_hashes"]
        for left, right in zip(
            reports[0]["records"], reports[1]["records"], strict=True
        )
    )


def test_u4_5f_retains_memory_and_fidelity_signal_without_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    gates = evidence["terminal_gates"]
    assert gates["candidate_peak_rss_at_most_512_mib"] is True
    assert gates["candidate_to_full_rss_ratio_at_most_0_75_per_source"] is True
    assert gates["rgb_rmse_at_most_0_03"] is True
    assert gates["rgb_p95_at_most_0_08"] is True
    assert gates["median_and_worst_candidate_to_full_wall_ratio"] is False
    assert gates["formal_pass"] is False
    assert evidence["status"] == "FAIL_CLOSED_U4_5F_RAW_HALF_SIZE_PREVIEW_LATENCY_TAIL"
    assert "Do not tune scheduling" in evidence["decision"]["stop"]

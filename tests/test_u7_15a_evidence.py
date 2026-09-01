from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION_RESULT.json"
)


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_15a_evidence_is_exact_and_bounded() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    assert report["status"] == (
        "PASS_PRIVATE_U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION"
    )
    assert report["formal_reports"] == {
        "forward_path": "outputs/eval/u7_15a_desktop_batch_representative_selection/formal_forward.json",
        "reverse_path": "outputs/eval/u7_15a_desktop_batch_representative_selection/formal_reverse.json",
        "bytes_each": 3743,
        "sha256": "2ea8b1499489f24719adbe0b6ff240033f09f0d2d6d9336a644b7d326a90ee60",
        "byte_exact": True,
        "scientific_identity": "a726c15ce5facc1485afcdaf9dcbfb82863918cd2df127a0509472900328521b",
        "order_independent_scientific_payload_exact": True,
        "owned_tmp_residue": 0,
        "owned_publication_residue": 0,
    }
    assert all(report["gates"].values())
    assert report["science"]["current_explicit_preview"] == "z-later.png"
    assert report["science"]["canonical_order"] == ["a-first.png", "z-later.png"]
    assert report["claim"]["explicit_user_authority_only"] is True
    assert report["claim"]["automatic_aesthetic_routing"] is False
    assert report["claim"]["calibrated_stock_response"] is False
    assert report["claim"]["physical_film_reproduction"] is False


def test_u7_15a_evidence_binds_formal_git_objects() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    commit = report["commits"]["formal_execution"]
    for path, binding in report["git_object_bindings"].items():
        blob = _git_blob(commit, path)
        assert hashlib.sha256(blob).hexdigest() == binding["sha256"]
        assert (
            subprocess.run(
                ["git", "rev-parse", f"{commit}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == binding["git_blob"]
        )


def test_u7_15a_evidence_binds_formal_materialized_bytes() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    formal = json.loads(
        (ROOT / report["formal_reports"]["forward_path"]).read_text("utf-8")
    )
    assert formal["bindings"] == report["formal_materialized_bindings"]

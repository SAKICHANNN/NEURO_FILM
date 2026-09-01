from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_18A_DESKTOP_FAST_EXPORT_POLICY_RESULT.json"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def test_u7_18a_evidence_binds_exact_fast_export_result() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    assert payload["status"] == "PASS_PRIVATE_U7_18A_DESKTOP_FAST_EXPORT_POLICY"

    reports = payload["formal_reports"]
    forward_bytes = (ROOT / reports["forward_path"]).read_bytes()
    reverse_bytes = (ROOT / reports["reverse_path"]).read_bytes()
    assert len(forward_bytes) == reports["forward_bytes"]
    assert len(reverse_bytes) == reports["reverse_bytes"]
    assert hashlib.sha256(forward_bytes).hexdigest() == reports["forward_sha256"]
    assert hashlib.sha256(reverse_bytes).hexdigest() == reports["reverse_sha256"]

    forward = json.loads(forward_bytes)
    reverse = json.loads(reverse_bytes)
    assert forward["scientific"] == reverse["scientific"]
    assert forward["scientific_sha256"] == reverse["scientific_sha256"]
    assert forward["scientific_sha256"] == reports["scientific_identity"]
    assert all(forward["scientific"]["gates"].values())
    assert all(row["passed"] for row in forward["scientific"]["case_results"].values())

    for path, identity in payload["bindings"].items():
        value = _git_bytes(path)
        assert len(value) == identity["bytes"]
        assert hashlib.sha256(value).hexdigest() == identity["sha256"]
        assert (
            subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
            ).strip()
            == identity["git_blob"]
        )

    observation = payload["primary_observation"]
    assert observation["preview_policy"] == {"tile_size": 256, "tile_workers": 1}
    assert observation["admitted_export_policy"] == {
        "tile_size": 512,
        "tile_workers": 8,
    }
    for order in ("forward", "reverse"):
        metrics = observation[order]
        assert metrics["candidate_to_baseline_wall_ratio"] <= 0.85
        assert metrics["candidate_rss_increase_bytes"] <= 536_870_912
        assert metrics["candidate_peak_process_tree_rss_bytes"] <= 4_294_967_296
    assert observation["candidate_repeat_exact"] is True
    assert observation["strict_replay_exact"] is True
    assert observation["source_immutable"] is True
    assert observation["owned_residue_zero"] is True

    assert payload["claim"]["mode"] == "film-inspired"
    assert payload["claim"]["evidence_grade"] == "look-approximation"
    assert payload["claim"]["user_tunable_resource_policy"] is False
    assert payload["claim"]["calibrated_stock_response"] is False
    assert payload["claim"]["physical_film_reproduction"] is False

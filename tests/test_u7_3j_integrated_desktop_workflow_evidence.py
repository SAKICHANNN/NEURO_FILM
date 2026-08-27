from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_3J_INTEGRATED_DESKTOP_WORKFLOW_RESULT.json"


def test_u7_3j_formal_evidence_is_exact_and_bounded() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "5e1e5e7499aaea38e276b8e58d8d6c6bc6cb6a2e6be536f14fce5f20bd39bc85"
    )
    value = json.loads(payload)
    scientific = value["scientific"]

    assert scientific["status"] == "PASS_PRIVATE_INTEGRATED_DESKTOP_WORKFLOW"
    assert value["stable_identity"] == (
        "sha256:c9446654b4a5c8e3b4c5b99110c136cb50984a484fea4f2cc35a3d4eb232ea22"
    )
    assert scientific["edge_version"] == "Microsoft Edge 151.0.4129.107"
    assert all(
        passed is True or count == 0
        for passed, count in (
            (scientific["gates"]["all_controls_labelled_and_operable"], 0),
            (scientific["gates"]["all_selected_outputs_exact"], 0),
            (scientific["gates"]["forward_reverse_exact"], 0),
            (scientific["gates"]["repeat_submission_rejected"], 0),
            (True, scientific["gates"]["non_loopback_network_requests"]),
            (True, scientific["gates"]["owned_residue_count"]),
        )
    )
    assert scientific["gates"]["machine_local_paths_exposed"] is False
    assert scientific["gates"]["valid_recipe_count_exact"] == 3
    assert scientific["gates"]["preview_count_exact"] == 3
    assert scientific["gates"]["distinct_style_count_exact"] == 3
    assert scientific["gates"]["distinct_preview_sha256_count_exact"] == 3

    rows = scientific["rows"]
    assert {row["style"] for row in rows} == {
        "ektar_100",
        "portra_400",
        "velvia_50",
    }
    assert all(row["output_sha256"] == row["expected_output_sha256"] for row in rows)
    assert all(row["dom"]["allImagesDecoded"] is True for row in rows)
    assert all(row["dom"]["remoteResources"] == [] for row in rows)
    assert all(row["output_count_after_second_submission"] == 1 for row in rows)
    assert "Look Approximation" in scientific["claim_ceiling"]
    assert "no stock accuracy" in scientific["claim_ceiling"]

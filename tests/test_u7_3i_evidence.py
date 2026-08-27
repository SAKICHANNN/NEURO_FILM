from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_3I_LOCAL_BROWSER_RECIPE_EXPORT_RESULT.json"


def test_u7_3i_formal_evidence_is_exact_and_bounded() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "70519123d91e70068f28811250af62e1bef459bf9608b42d8533f7a9a4288841"
    )
    value = json.loads(payload)
    scientific = value["scientific"]
    assert scientific["status"] == "PASS_PRIVATE_LOCAL_BROWSER_RECIPE_EXPORT"
    assert scientific["edge_version"] == "Microsoft Edge 151.0.4129.107"
    assert scientific["runs_output_exact"] is True
    assert all(scientific["gates"].values())
    assert len(scientific["runs"]) == 2
    assert {row["style"] for row in scientific["runs"]} == {"ektar_100"}
    assert {
        row["output_sha256"] for row in scientific["runs"]
    } == {"6c5c8b9c04fea4f265e1b742ce568b13a34d72462f7b6466f29685dabd03045e"}
    assert "No calibrated stock" in scientific["claim_ceiling"]

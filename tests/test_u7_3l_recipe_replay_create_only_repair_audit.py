from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_3l_recipe_replay_create_only_repair import run_audit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_3L_RECIPE_REPLAY_CREATE_ONLY_REPAIR_RESULT.json"


def test_u7_3l_formal_audit_passes_and_is_order_exact() -> None:
    forward = run_audit("forward")
    reverse = run_audit("reverse")
    assert forward == reverse
    assert forward["status"] == "PASS_PRIVATE_U7_3L_RECIPE_REPLAY_CREATE_ONLY_REPAIR"
    assert all(forward["scientific"]["gates"].values())


def test_u7_3l_tracked_evidence_matches_fresh_audit() -> None:
    expected = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert run_audit("forward") == expected

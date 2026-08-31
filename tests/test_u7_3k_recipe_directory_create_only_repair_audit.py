from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_u7_3k_recipe_directory_create_only_repair import run_audit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_3K_RECIPE_DIRECTORY_CREATE_ONLY_REPAIR_RESULT.json"


def test_u7_3k_formal_audit_passes_both_orders() -> None:
    forward = run_audit("forward")
    reverse = run_audit("reverse")
    assert forward == reverse
    assert forward["status"] == "PASS_PRIVATE_U7_3K_RECIPE_DIRECTORY_CREATE_ONLY_REPAIR"
    assert all(forward["gates"].values())
    assert {row["name"] for row in forward["rows"]} == {"request_set", "workspace"}
    assert all(row["initial_foreign_directory_preserved"] for row in forward["rows"])
    assert all(row["foreign_added_member_preserved"] for row in forward["rows"])
    assert all(row["foreign_replacement_preserved"] for row in forward["rows"])


def test_u7_3k_tracked_evidence_is_exact() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "5a3f9981fcedbee7e172122cd6dc4f5e6444e167e5730a58b5fde621d31a4146"
    )
    evidence = json.loads(payload)
    assert evidence["status"] == "PASS_PRIVATE_U7_3K_RECIPE_DIRECTORY_CREATE_ONLY_REPAIR"
    assert evidence["stable_identity"] == (
        "fcfe3e9e83dfcd5a805be3a3517077bf12f3864366e1225e9f9c13a8ca014af9"
    )
    assert all(evidence["gates"].values())

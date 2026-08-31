from __future__ import annotations

from scripts.audit_u7_3l_recipe_replay_create_only_repair import run_audit


def test_u7_3l_formal_audit_passes_and_is_order_exact() -> None:
    forward = run_audit("forward")
    reverse = run_audit("reverse")
    assert forward == reverse
    assert forward["status"] == "PASS_PRIVATE_U7_3L_RECIPE_REPLAY_CREATE_ONLY_REPAIR"
    assert all(forward["scientific"]["gates"].values())

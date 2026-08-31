from __future__ import annotations

from scripts.audit_u7_3k_recipe_directory_create_only_repair import run_audit


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

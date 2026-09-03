from __future__ import annotations

from pathlib import Path

from scripts.audit_u7_21d_product_recipe_commit_snapshot import audit


def test_audit_passes_and_is_order_invariant(tmp_path: Path) -> None:
    first = audit(tmp_path, "forward")
    second = audit(tmp_path, "reverse")
    assert first == second
    assert first["status"] == "PASS_PRIVATE_U7_21D_PRODUCT_RECIPE_COMMIT_SNAPSHOT"
    assert all(first["gates"].values())

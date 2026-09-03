from __future__ import annotations

from pathlib import Path

from scripts.audit_u7_21e_desktop_batch_aggregate_publication_scope import audit


def test_audit_passes_and_is_order_invariant(tmp_path: Path) -> None:
    first = audit(tmp_path, "forward")
    second = audit(tmp_path, "reverse")
    assert first == second
    assert first["status"] == (
        "PASS_PRIVATE_U7_21E_DESKTOP_BATCH_AGGREGATE_PUBLICATION_SCOPE"
    )
    assert all(first["gates"].values())

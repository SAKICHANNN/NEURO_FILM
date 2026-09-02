from __future__ import annotations

from pathlib import Path

from scripts import audit_u7_9d_private_runtime_source_scope_binding as audit


def test_formal_fixture_controls_are_order_independent(tmp_path: Path) -> None:
    names = (
        "docs_descendant",
        "dirty_tracked_docs",
        "scoped_untracked",
        "runtime_commit_drift",
        "requirements_commit_drift",
        "non_descendant",
    )
    forward = [audit._control(name, tmp_path / f"f-{name}") for name in names]
    reverse = [audit._control(name, tmp_path / f"r-{name}") for name in reversed(names)]
    assert all(row["passed"] for row in forward)
    assert sorted(forward, key=lambda row: row["name"]) == sorted(
        reverse, key=lambda row: row["name"]
    )

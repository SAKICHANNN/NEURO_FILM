from __future__ import annotations

import tempfile
from pathlib import Path

from scripts import audit_u7_9d_private_runtime_source_scope_binding as audit

ROOT = Path(__file__).resolve().parents[1]


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


def test_formal_fixture_runs_on_repo_relative_p_backed_scratch() -> None:
    with tempfile.TemporaryDirectory(prefix="u7_9d_test_", dir=ROOT / "tmp") as raw:
        row = audit._control("docs_descendant", Path(raw) / "control")
    assert row["passed"] is True

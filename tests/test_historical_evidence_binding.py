from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]


def _historical_readme_blob() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^:README.md"], cwd=ROOT, text=True
    ).strip()


def test_recorded_git_blob_can_prove_nonmaterialized_worktree_binding() -> None:
    assert_historical_evidence_binding(
        ROOT,
        {
            "path": "README.md",
            "sha256": "0" * 64,
            "git_blob": _historical_readme_blob(),
        },
    )


def test_unknown_git_blob_does_not_bypass_historical_binding() -> None:
    with pytest.raises(AssertionError, match="recorded git_blob"):
        assert_historical_evidence_binding(
            ROOT,
            {
                "path": "README.md",
                "sha256": "0" * 64,
                "git_blob": "f" * 40,
            },
        )

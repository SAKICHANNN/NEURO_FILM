from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.freeze_u5_r2spcp3_pixel_roles import ROLE_COUNTS, selection_key

ROOT = Path(__file__).resolve().parents[1]


def test_role_counts_consume_exact_eligible_pool() -> None:
    assert sum(count for _, count in ROLE_COUNTS) == 457


def test_role_names_are_unique_and_ordered() -> None:
    names = [name for name, _ in ROLE_COUNTS]
    assert names == [
        "development",
        "confirmation",
        "operator_fit",
        "operator_calibration",
        "operator_sealed",
        "reserve",
    ]
    assert len(names) == len(set(names))


def test_selection_key_is_stable_and_scene_sensitive() -> None:
    assert selection_key("I0001") == selection_key("I0001")
    assert selection_key("I0001") != selection_key("I0002")


def test_direct_cli_entrypoint_imports_repo_scripts() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/freeze_u5_r2spcp3_pixel_roles.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts import audit_u7_14b_desktop_canonical_scratch_boundary as audit

ROOT = Path(__file__).resolve().parents[1]


def test_u7_14b_contract_and_config_are_bound_to_the_frozen_parent() -> None:
    config = json.loads(audit.CONFIG.read_text("utf-8"))
    assert config["node_id"] == "U7.14B"
    assert config["parent_head"] == "9ab788dc4999fea054b0d4e76499fca54818a29a"
    assert config["canonical_boundary"] == {
        "accept_escape_reparse_points": False,
        "accept_missing_paths": False,
        "accept_outside_paths": False,
        "accept_regular_files": False,
        "accept_root": True,
        "accept_existing_descendants": True,
        "logical_root": "tmp",
        "machine_specific_drive_path_in_code": False,
        "resolve_before_compare": True,
    }


def test_u7_14b_formal_controller_has_no_import_or_syntax_side_effect() -> None:
    source = Path(audit.__file__).read_text("utf-8")
    ast.parse(source)
    assert "build_report(args.order)" in source
    assert "tracked worktree must be clean before formal execution" in source
    assert "formal report must be under repository outputs" in source

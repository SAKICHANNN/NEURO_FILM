from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts import audit_u7_15a_desktop_batch_representative_selection as audit


def test_u7_15a_contract_binds_optional_explicit_representative() -> None:
    config = json.loads(audit.CONFIG.read_text("utf-8"))
    assert config["node_id"] == "U7.15A"
    assert config["parent_head"] == "ff1c0d720a9afaf86b440ee5b6b86d240c323b50"
    assert config["interface"] == {
        "method": "ProductDesktopWorkflow.render_batch_previews",
        "new_keyword": "representative_path",
        "default": None,
        "default_semantics": "canonical-first-exact",
        "explicit_semantics": "one-exact-bound-member",
        "canonical_batch_order_unchanged": True,
        "receipt_schema_unchanged": True,
    }


def test_u7_15a_formal_controller_is_create_only_and_source_locked() -> None:
    source = Path(audit.__file__).read_text("utf-8")
    ast.parse(source)
    assert "tracked worktree must be clean before formal execution" in source
    assert "owned formal path must be absent before execution" in source
    assert 'with output.open("x"' in source
    assert "formal report must be under repository outputs" in source

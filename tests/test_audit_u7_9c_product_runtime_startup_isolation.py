from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_u7_9c_product_runtime_startup_isolation as audit

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_implementation_commit_and_blob_resolve() -> None:
    config = json.loads(audit.CONFIG.read_text("utf-8"))
    assert audit._git("cat-file", "-t", config["implementation"]["commit"]) == (
        "commit"
    )
    assert audit._git(
        "rev-parse", f'{config["implementation"]["commit"]}:scripts/install_product_runtime.py'
    ) == config["implementation"]["installer_git_blob"]


def test_source_facts_cover_contract_config_installer_renderer_and_requirements() -> None:
    facts = audit._source_facts()
    assert set(facts) == {
        "config_sha256",
        "contract_sha256",
        "installer_sha256",
        "renderer_sha256",
        "requirements_sha256",
    }
    assert all(len(value) == 64 for value in facts.values())


def test_formal_root_must_be_repo_relative_tmp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo-relative tmp"):
        audit.audit(tmp_path / "root", tmp_path, "forward")


def test_invalid_order_rejects_before_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sideways"):
        audit.audit(ROOT / "tmp" / "u7_9c_invalid", tmp_path, "sideways")


def test_sentinel_repository_uses_process_scoped_safe_directory() -> None:
    source = Path(audit.__file__).read_text("utf-8")
    assert 'f"safe.directory={repository}"' in source
    assert "config --global" not in source

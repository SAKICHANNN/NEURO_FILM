from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import audit_u4_5e_receipt_bound_preview_session as audit_module
from scripts.audit_u4_5e_receipt_bound_preview_session import (
    _implementation_git_objects_exact,
    build_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5e_receipt_bound_preview_session_v1.json"


def test_formal_report_passes_all_frozen_gates() -> None:
    report = build_report(config_path=CONFIG, order="forward")
    assert report["status"] == "PASS_PRIVATE_RECEIPT_BOUND_PREVIEW_SESSION"
    assert all(report["scientific"]["gates"].values())
    assert report["owned_runtime_residue_zero"] is True


def test_implementation_binding_rejects_unknown_cache_core() -> None:
    real_git_object = audit_module._git_object

    def unknown_cache_core(commit: str, path: str) -> str:
        if commit == "HEAD" and path == "src/inference/three_stock_preview_cache.py":
            return "0" * 40
        return real_git_object(commit, path)

    with patch.object(audit_module, "_git_object", unknown_cache_core):
        assert _implementation_git_objects_exact() is False

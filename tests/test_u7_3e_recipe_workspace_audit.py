from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/audit_u7_3e_offline_recipe_workspace.py")
    spec = importlib.util.spec_from_file_location("u7_3e_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u7_3e_audit_passes_in_both_orders() -> None:
    module = _module()
    forward = module.run_audit(reverse=False)
    reverse = module.run_audit(reverse=True)
    assert forward == reverse
    assert forward["scientific"]["status"] == "PASS_PRIVATE_OFFLINE_RECIPE_WORKSPACE"
    assert all(forward["scientific"]["gates"].values())

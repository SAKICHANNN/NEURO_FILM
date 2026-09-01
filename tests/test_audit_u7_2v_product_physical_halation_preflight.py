from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_2v_product_physical_halation_preflight.py"
CONFIG = ROOT / "configs/u7_2v_product_physical_halation_preflight_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_u7_2v", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_is_order_invariant_and_passes() -> None:
    module = _load_module()
    forward = module.build_report(CONFIG, "forward")
    reverse = module.build_report(CONFIG, "reverse")

    assert forward == reverse
    assert forward["status"] == "PASS"
    assert forward["case_count"] == 59
    assert all(forward["gates"].values())
    assert len(forward["parent_rows"]) == 3

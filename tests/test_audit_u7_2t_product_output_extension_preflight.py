from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_2t_product_output_extension_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_u7_2t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_passes_in_forward_and_reverse_order() -> None:
    module = _load_module()
    forward = module.audit(reverse=False)
    reverse = module.audit(reverse=True)

    assert forward == reverse
    assert forward["status"] == "PASS"
    assert forward["case_count"] == 20
    assert all(forward["gates"].values())

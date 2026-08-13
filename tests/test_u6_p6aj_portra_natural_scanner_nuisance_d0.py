from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.portra_natural_scanner_nuisance_d0 import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6aj_portra_natural_scanner_nuisance_d0_v1.json"


def test_contract_freezes_scanner_print_roles() -> None:
    contract = load_contract(CONFIG)
    assert contract["inputs"]["source_role"].endswith("noritsu_scan")
    assert contract["inputs"]["target_role"].endswith("ra4_epson_scan")
    assert contract["gates"]["minimum_improvement_over_identity_fraction"] == 0.1


def test_contract_rejects_domain_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["inputs"]["fit_domain"] = "display_rgb_unspecified"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)

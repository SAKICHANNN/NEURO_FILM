from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import scanner_crossed_design_identifiability as subject

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6ap_scanner_crossed_design_identifiability_v1.json"


def test_real_design_identifies_only_clean_software_contrast() -> None:
    report = subject.evaluate(subject.load_contract(CONFIG), ROOT)
    assert report["design"]["rank"] == 4
    assert report["design"]["nullity"] == 4
    assert report["pure_named_effects_identified"] == ["software_on_same_ls50_device_operator_sampling"]
    assert report["random_repeat_noise"]["maximum_fixed_condition_repeats"] == 1
    assert report["automatic_pass"] is False


def test_contract_rejects_insufficient_repeat_requirement(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["repeat_noise_requirement"]["minimum_same_physical_slide_scans"] = 2
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        subject.load_contract(path)


def test_parent_drift_fails_closed(tmp_path: Path) -> None:
    value = subject.load_contract(CONFIG)
    value["parents"]["source_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent drift"):
        subject.evaluate(value, ROOT)

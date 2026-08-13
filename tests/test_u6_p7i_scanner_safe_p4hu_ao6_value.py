from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import scanner_safe_p4hu_ao6_value as subject

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p7i_scanner_safe_p4hu_ao6_value_v1.json"


def test_contract_is_frozen_and_parents_are_hash_bound() -> None:
    contract = subject.load_contract(CONFIG)
    subject._validate(contract)
    assert contract["candidate"]["p4hu_or_ao6_refit_allowed"] is False
    for binding in contract["parents"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert subject.sha256_file(path) == binding["sha256"]


def test_contract_rejects_direction_change() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract["candidate"]["direction_change_allowed"] = True
    with pytest.raises(ValueError, match="candidate policy"):
        subject._validate(contract)


def test_wrapper_requires_scanner_safe_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = subject.load_contract(CONFIG)
    monkeypatch.setattr(subject, "_bound_json", lambda *_args: {})
    monkeypatch.setattr(subject, "evaluate_p7h", lambda *_args, **_kwargs: {"automatic_pass": True, "aggregates": {}})
    with pytest.raises(TypeError, match="receipt missing"):
        subject.evaluate(contract, root=ROOT, output_dir=tmp_path / "out", build_dir=tmp_path / "build")


def test_wrapper_applies_additional_gates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = subject.load_contract(CONFIG)
    monkeypatch.setattr(subject, "_bound_json", lambda *_args: {})
    result = {
        "schema": "parent",
        "automatic_pass": True,
        "aggregates": {
            "scanner_safe_residual": {
                "maximum_limited_pixel_fraction": 0.01,
                "minimum_median_scale": 1.0,
                "minimum_scale": 0.5,
                "maximum_collinearity_error": 0.0,
            }
        },
        "blind_review_allowed": True,
        "decision": "parent-pass",
    }
    monkeypatch.setattr(subject, "evaluate_p7h", lambda *_args, **kwargs: result)
    actual = subject.evaluate(contract, root=ROOT, output_dir=tmp_path / "out", build_dir=tmp_path / "build")
    assert actual["automatic_pass"] is True
    assert actual["blind_review_allowed"] is True
    assert all(actual["scanner_safe_residual_gates"].values())
    assert actual["stable_evidence_id"]

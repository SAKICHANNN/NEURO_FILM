from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.rawpixls_confirmation_preflight import (
    ConfirmationSourceError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ao7s_fresh_rawpixls_source_preflight_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_ao7s_contract_has_fresh_balanced_exact_rows() -> None:
    config = _config()
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert len(rows) == 18
    assert len({row["sha256"] for row in rows}) == 18
    assert len({(row["make"], row["model"]) for row in rows}) == 18
    assert {row["make"] for row in rows} == {
        "Canon",
        "Fujifilm",
        "Kodak",
        "Leica",
        "Nikon",
        "Olympus",
        "Panasonic",
        "Pentax",
        "Sony",
    }
    assert len(config["preflight"]["comparison_manifests"]) == 2
    assert config["operator_fitting_allowed"] is False


def test_ao7s_rejects_parent_or_comparison_drift() -> None:
    config = _config()
    config["parent_decision"]["required_decision"] = "wrong"
    with pytest.raises(ConfirmationSourceError, match="state"):
        validate_contract(ROOT, config)

    config = _config()
    config["preflight"]["comparison_manifests"][1]["sha256"] = "0" * 64
    with pytest.raises(ConfirmationSourceError, match="comparison"):
        validate_contract(ROOT, config)

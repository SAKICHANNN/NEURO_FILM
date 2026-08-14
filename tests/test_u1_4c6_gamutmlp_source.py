from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.gamutmlp_prophoto_source_audit import (
    GamutMLPSourceError,
    _member_is_safe,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c6_gamutmlp_nus_prophoto_source_v1.json"


def test_contract_binds_official_archive_and_deterministic_selection() -> None:
    contract = load_contract(CONTRACT)
    assert contract["archive"]["bytes"] == 2703445642
    assert contract["archive"]["expected_png_members"] == 2000
    assert contract["archive"]["expected_style_counts"] == {
        "color": 500,
        "landscape": 500,
        "standard": 500,
        "vivid": 500,
    }
    assert contract["selection"]["expected_rows"] == 24
    assert len(contract["selection"]["camera_models"]) == 8
    assert contract["rights"]["commercial_product_use_allowed"] is False


def test_contract_drift_and_unsafe_members_reject(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["archive"]["bytes"] += 1
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GamutMLPSourceError, match="contract hash drift"):
        load_contract(mutated)
    assert _member_is_safe("train_prop_512_16b/a.png", "train_prop_512_16b")
    assert not _member_is_safe("../a.png", "train_prop_512_16b")
    assert not _member_is_safe("train_prop_512_16b\\a.png", "train_prop_512_16b")


def test_bound_archive_identity_when_available() -> None:
    contract = load_contract(CONTRACT)
    archive = ROOT / contract["archive"]["logical_path"]
    if not archive.is_file():
        pytest.skip("P-backed GamutMLP archive unavailable")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert archive.stat().st_size == contract["archive"]["bytes"]
    assert digest.hexdigest() == contract["archive"]["sha256"]

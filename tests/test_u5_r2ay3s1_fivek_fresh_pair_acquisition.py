from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.fivek_fresh_pair_acquisition import (
    FiveKFreshPairAcquisitionError,
    ensure_owned_root,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay3s1_fivek_fresh_pair_acquisition_v1.json"
)


def test_contract_binds_exact_owned_inventory() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["preflight_manifest"]["rows"]) == 64
    assert config["preflight"]["expected_bytes"] == 3648683762
    assert config["ownership"]["external_root"].startswith("D:/nf-019f4b76-")


def test_owned_root_rejects_foreign_nonempty_directory(
    tmp_path: Path,
) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unowned.txt").write_text("x", encoding="utf-8")
    ownership = {
        "owner": "thread",
        "purpose": "test",
    }
    with pytest.raises(FiveKFreshPairAcquisitionError):
        ensure_owned_root(foreign, ownership)


def test_owned_root_marker_is_repeat_stable(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    ownership = {
        "owner": "thread",
        "purpose": "test",
    }
    first = ensure_owned_root(owned, ownership)
    marker = (first / ".neuro_film_owner.json").read_bytes()
    second = ensure_owned_root(owned, ownership)
    assert first == second
    assert (second / ".neuro_film_owner.json").read_bytes() == marker

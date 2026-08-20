from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_u1_4c18_prophoto_24mp_product import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c18_prophoto_24mp_product_resources_v1.json"


def test_c18_contract_binds_exact_24mp_fixture_and_frozen_goals() -> None:
    payload, digest = load_contract(CONTRACT)
    assert digest == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert payload["fixture"]["pixels"] == 24_000_000
    assert payload["measurement"]["fresh_processes"] == 2
    assert payload["gates"]["maximum_peak_process_tree_rss_bytes"] == 2**30
    assert payload["gates"]["maximum_worker_wall_seconds"] == 15.0


def test_c18_contract_rejects_fixture_identity_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixture"]["sha256"] = "0" * 64
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound file drift"):
        load_contract(changed)


def test_c18_contract_rejects_measurement_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["measurement"]["fresh_processes"] = 1
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement drift"):
        load_contract(changed)

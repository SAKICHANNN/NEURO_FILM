from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.base_return_topology_photographic import load_contract, validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3r_base_return_topology_photographic_v1.json"
MANIFEST = ROOT / "outputs/u5_r2bh1s_global_policy_source_preflight_v1/manifest.json"


def test_contract_rejects_posthoc_diagnostic_strength_change(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixed_arms"]["diagnostic_return_fraction_scale"] = 5.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_contract(path)


@pytest.mark.skipif(not MANIFEST.is_file(), reason="exact BH1 source manifest unavailable")
def test_population_is_exact_and_disjoint_from_p3o() -> None:
    rows, geometry, _ = validate_contract(ROOT, load_contract(CONTRACT))
    assert len(rows) == 12
    assert len({row["make"] for row in rows}) == 12
    assert geometry["experiment_id"] == "U6.P3Q1"

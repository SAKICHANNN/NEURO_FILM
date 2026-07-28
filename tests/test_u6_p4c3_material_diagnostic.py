from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.benchmark_u6_p4c3_material import load_contract
from scripts.render_u6_p4c3_material_diagnostic import render_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p4c3_streamed_material_benchmark_v1.json"


def test_small_material_diagnostic_is_replay_exact(tmp_path: Path) -> None:
    contract = deepcopy(load_contract(CONTRACT))
    contract["diagnostic"]["shape"] = [48, 64]
    first = render_diagnostics(contract, tmp_path / "a")
    second = render_diagnostics(contract, tmp_path / "b")
    assert first["files"] == second["files"]
    assert first["colour_density_min"] > 0.0
    assert 0.0 < first["bw_transmittance_min"]
    assert first["bw_transmittance_max"] <= 1.0

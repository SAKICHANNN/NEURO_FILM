from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_vision3_granularity_diversity import (
    GranularityDiversityError,
    audit_diversity,
    canonical_json,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu2_kodak_vision3_granularity_diversity_v1.json"


def test_contract_rejects_gate_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["minimum_pair_log_shape_rmse_per_channel"] = 0.079
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GranularityDiversityError, match="contract drift"):
        load_contract(changed)


def test_granularity_diversity_is_exact_and_source_bound(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    first = audit_diversity(config, ROOT, overlay_dir=tmp_path / "a")
    second = audit_diversity(config, ROOT, overlay_dir=tmp_path / "b")
    assert canonical_json(first) == canonical_json(second)
    assert first["trace_sha256"] == config["trace"]["sha256"]
    assert first["gate_results"]["source_integrity"] is True
    assert first["gate_results"]["trace_integrity"] is True
    assert first["gate_results"]["measurement_context"] is True
    assert first["gate_results"]["parent_one_gaussian_compiler_remains_closed"] is True
    assert set(first["overlay_sha256"]) == set(config["comparison"]["stocks"])


def test_all_traces_pass_geometry_ink_range_and_coverage(tmp_path: Path) -> None:
    report = audit_diversity(
        load_contract(CONFIG), ROOT, overlay_dir=tmp_path / "overlays"
    )
    for gate in (
        "axis_geometry",
        "source_ink_proximity",
        "sigma_range",
        "common_exposure_coverage",
        "digitization_uncertainty",
    ):
        assert report["gate_results"][gate] is True

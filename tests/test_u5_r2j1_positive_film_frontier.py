from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.positive_film_frontier import (
    PositiveFilmFrontierError,
    _resolve_comparator_output,
    candidate_bank,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2j1_positive_film_frontier_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_candidate_bank_is_the_frozen_cross_product() -> None:
    config = _config()
    candidates = candidate_bank(config)
    assert len(candidates) == 25
    assert candidates[0] == {
        "candidate_id": "neutral_positive_reference__s10",
        "witness_id": "neutral_positive_reference",
        "strength": 0.1,
    }
    assert candidates[-1] == {
        "candidate_id": "cross_bias_like__s35",
        "witness_id": "cross_bias_like",
        "strength": 0.35,
    }


def test_contract_validates_two_complete_comparator_manifests() -> None:
    validated = validate_contract(ROOT, _config())
    assert len(validated["samples"]) == 41
    assert len(validated["comparator_paths"]) == 82
    assert {candidate for candidate, _ in validated["comparator_paths"]} == {
        "bland_safe_rich_control",
        "cyan_shadow_warm_highlight_like__s50",
    }


def test_operator_hash_drift_fails_closed() -> None:
    config = _config()
    config["operator_config_sha256"] = "0" * 64
    with pytest.raises(PositiveFilmFrontierError, match="hash mismatch"):
        validate_contract(ROOT, config)


def test_comparator_output_supports_existing_lineage_conventions() -> None:
    global_manifest = (
        ROOT
        / "outputs/u5_r2b_global_frontier_v1/render_pass1/manifest.json"
    )
    density_manifest = (
        ROOT
        / "outputs/u5_r2e1_density_witness_frontier_v1/render_pass1/manifest.json"
    )
    global_output = json.loads(global_manifest.read_text(encoding="utf-8"))[
        "records"
    ][0]["output"]
    density_output = json.loads(density_manifest.read_text(encoding="utf-8"))[
        "records"
    ][0]["output"]
    assert _resolve_comparator_output(ROOT, global_manifest, global_output).is_file()
    assert _resolve_comparator_output(ROOT, density_manifest, density_output).is_file()
    with pytest.raises(PositiveFilmFrontierError, match="relative"):
        _resolve_comparator_output(ROOT, global_manifest, str(ROOT / "bad.png"))

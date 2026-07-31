from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.canonicalizer_consensus_appearance import (
    CanonicalizerConsensusError,
    canonical_json_bytes,
    load_config,
    make_scenario_distributions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk21_canonicalizer_consensus_appearance_v1.json"


def test_contract_binds_bk20_and_hard_policy() -> None:
    config = load_config(ROOT, CONFIG)
    assert (
        config["parent"]["required_decision"]
        == "unpaired_operator_identification_remains_unidentified_require_film_inspired_claim_ceiling"
    )
    assert len(config["canonicalizers"]) == 2
    assert "never average" in config["policy"]["selection"]
    assert config["policy"]["fallback"] == "identity"


def test_invalid_contract_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["status"] = "draft"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CanonicalizerConsensusError, match="not frozen"):
        load_config(ROOT, path)


def test_scenarios_are_deterministic_and_unpaired() -> None:
    config = load_config(ROOT, CONFIG)
    for index, scenario in enumerate(config["scenarios"]):
        first = make_scenario_distributions(
            config, scenario, scenario_index=index
        )
        second = make_scenario_distributions(
            config, scenario, scenario_index=index
        )
        assert all(np.array_equal(left, right) for left, right in zip(first, second))
        source_a, target_a, source_b, target_b = first
        assert source_a.shape == target_a.shape == source_b.shape == target_b.shape
        assert source_a.shape[1] == 3
        assert np.all((target_a >= 0.0) & (target_a <= 1.0))
        assert not np.array_equal(source_a, target_a)


def test_content_confounded_scenario_differs_from_material() -> None:
    config = load_config(ROOT, CONFIG)
    material = make_scenario_distributions(
        config, config["scenarios"][0], scenario_index=0
    )
    confounded = make_scenario_distributions(
        config, config["scenarios"][2], scenario_index=2
    )
    assert np.linalg.norm(np.mean(material[1], axis=0) - np.mean(material[0], axis=0))
    assert (
        np.linalg.norm(np.mean(confounded[1], axis=0) - np.mean(confounded[0], axis=0))
        > 0.1
    )


def test_report_encoding_rejects_nonfinite() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})

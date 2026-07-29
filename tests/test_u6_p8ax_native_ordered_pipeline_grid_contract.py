from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8ax_native_ordered_pipeline_grid import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u6_p8ax_native_ordered_pipeline_grid_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ax_contract_freezes_ordered_concurrency_grid() -> None:
    config = json.loads(CONFIG.read_text())
    parent = validate_contract(config)
    assert parent["next_leaf"].startswith("U6.P8AX")
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["p8aw_contract"]) == config[
        "p8aw_contract_sha256"
    ]
    assert config["pipeline_workers_candidates"] == [1, 2, 4, 8]
    assert config["execution_order"] == [1, 2, 4, 8, 8, 4, 2, 1]


def test_p8ax_contract_rejects_grid_drift() -> None:
    config = json.loads(CONFIG.read_text())
    for key, value in (
        ("pipeline_workers_candidates", [1, 2, 4]),
        ("execution_order", [1, 2, 4, 8]),
        ("max_in_flight_per_worker", 2),
    ):
        mutated = json.loads(json.dumps(config))
        mutated[key] = value
        try:
            validate_contract(mutated)
        except ValueError:
            pass
        else:
            raise AssertionError(f"P8AX accepted drifted {key}")

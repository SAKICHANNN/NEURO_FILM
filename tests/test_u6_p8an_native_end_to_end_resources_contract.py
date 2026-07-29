from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_u6_p8an_native_end_to_end_resources import (
    _source_rows,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8an_native_end_to_end_resources_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8an_contract_binds_parent_and_freezes_resource_gates() -> None:
    config = json.loads(CONFIG.read_text())
    parent = validate_contract(config)
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert parent["result"]["status"] == "pass"
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == 384 * (
        1024**2
    )
    assert config["gates"]["maximum_worker_elapsed_seconds"] == 15.0
    assert config["safety"]["worker_kill_rss_bytes"] == 1024**3


def test_p8an_analytic_source_rows_are_partition_exact() -> None:
    full = _source_rows(y0=0, y1=17, height=17, width=19)
    partitioned = np.concatenate(
        [
            _source_rows(y0=0, y1=5, height=17, width=19),
            _source_rows(y0=5, y1=11, height=17, width=19),
            _source_rows(y0=11, y1=17, height=17, width=19),
        ],
        axis=0,
    )
    assert partitioned.tobytes() == full.tobytes()

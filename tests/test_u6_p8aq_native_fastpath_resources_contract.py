from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_u6_p8aq_native_fastpath_resources import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u6_p8aq_native_fastpath_resources_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8aq_contract_reuses_frozen_p8an_gates() -> None:
    config = json.loads(CONFIG.read_text())
    validate_contract(config)
    frozen = json.loads(
        (ROOT / config["frozen_resource_contract"]).read_text()
    )
    assert config["scenario"] == frozen["scenario"]
    assert config["tile_rows"] == frozen["tile_rows"]
    assert config["safety"] == frozen["safety"]
    assert config["gates"] == frozen["gates"]
    assert _sha256(ROOT / config["parent_decision"]) == config[
        "parent_decision_sha256"
    ]
    assert _sha256(ROOT / config["frozen_resource_contract"]) == config[
        "frozen_resource_contract_sha256"
    ]
    assert config["execution"]["ao6_context_abi"] == "v2"
    assert config["execution"]["ao6_display_abi"] == "v2"

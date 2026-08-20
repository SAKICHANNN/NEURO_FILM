from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8bo1_android_native_standard_runtime_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_binds_exact_p8bo_parent_and_portable_sources() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_before_android_execution"
    for key in ("parent", "portable_contract"):
        bound = contract[key]
        assert _sha256(ROOT / bound["path"]) == bound["sha256"]
    parent = json.loads((ROOT / contract["parent"]["path"]).read_text())
    assert parent["result"]["status"] == contract["parent"]["required_result_status"]
    assert (
        parent["result"]["oracle_output_sha256"]
        == contract["parent"]["required_oracle_output_sha256"]
    )


def test_runtime_scope_and_numeric_gates_are_frozen() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["runtime"]["abi"] == "x86_64"
    assert contract["runtime"]["wipe_data_cold_boots"] == 2
    assert contract["runtime"]["fresh_processes_per_boot"] == 2
    assert contract["fixture"]["maximum_absolute_error"] == 2e-6
    assert contract["fixture"]["maximum_rmse"] == 5e-7
    assert len(contract["forbidden_claims"]) == 6

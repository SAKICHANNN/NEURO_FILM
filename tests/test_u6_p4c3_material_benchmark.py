from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.benchmark_u6_p4c3_material import load_contract, worker


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p4c3_streamed_material_benchmark_v1.json"


def test_small_streaming_worker_is_chunk_exact() -> None:
    contract = deepcopy(load_contract(CONTRACT))
    contract["shape"] = [48, 64]
    first = worker(contract, 7)
    second = worker(contract, 19)
    assert [
        row["output_sha256"] for row in first["layers"]
    ] == [row["output_sha256"] for row in second["layers"]]
    assert all(row["finite_domain_valid"] for row in first["layers"])
    assert sum(
        row["brightening_violation_count"] for row in first["layers"]
    ) == 0

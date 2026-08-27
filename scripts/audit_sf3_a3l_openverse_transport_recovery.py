"""Run the frozen SF3.A3L Openverse transport-recovery audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.openverse_three_stock_refresh import audit_fresh_snapshot
from src.real_film.openverse_transport_recovery import (
    fetch_snapshot_urllib,
    load_effective_config,
    scientific_snapshot,
)
from src.real_film.yfcc_stock_source import atomic_json


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sf3_a3l_openverse_transport_recovery_v1.json",
    )
    args = parser.parse_args()
    contract_bytes = args.config.read_bytes()
    contract = json.loads(contract_bytes)
    base = load_effective_config(ROOT, contract)
    order = [str(stock["film_stock_id"]) for stock in base["stocks"]]
    snapshots = [
        fetch_snapshot_urllib(base, contract, stock_order=order),
        fetch_snapshot_urllib(base, contract, stock_order=list(reversed(order))),
    ]
    audited = [audit_fresh_snapshot(value, base, root=ROOT) for value in snapshots]
    scientific = [scientific_snapshot(value) for value in snapshots]
    scientific_exact = scientific[0] == scientific[1]
    reports = []
    for index, ((report, decision), snapshot) in enumerate(
        zip(audited, snapshots, strict=True)
    ):
        reports.append(
            {
                "schema": "neuro-film.sf3-a3l-openverse-transport-recovery-report.v1",
                "run_index": index,
                "stock_order": order if index == 0 else list(reversed(order)),
                "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                "snapshot": snapshot,
                "audit": report,
                "decision": decision,
                "scientific_payload_sha256": _digest(scientific[index]),
                "scientific_payload_exact": scientific_exact,
            }
        )
    paths = contract["outputs"]
    atomic_json(ROOT / paths["snapshot"], snapshots[0])
    atomic_json(ROOT / paths["report_a"], reports[0])
    atomic_json(ROOT / paths["report_b"], reports[1])
    print(
        json.dumps(
            {
                "decision_a": audited[0][1]["decision"],
                "decision_b": audited[1][1]["decision"],
                "scientific_payload_exact": scientific_exact,
                "scientific_payload_sha256": reports[0]["scientific_payload_sha256"],
                "freshness": audited[0][1].get("freshness"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if scientific_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())

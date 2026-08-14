#!/usr/bin/env python3
"""Replay all frozen U1.4C16 natural ProPhoto render receipts once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json
from src.inference.prophoto_rec2020_replay import (
    replay_supported_prophoto_velvia_rec2020,
)

CONTRACT_SCHEMA = "neuro-film.u1-4c17-prophoto-rec2020-receipt-replay-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c17-prophoto-rec2020-receipt-replay-report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(root: Path, relative_value: str, expected_sha256: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("U1.4C17 paths must be repository-relative")
    path = root / relative
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError(f"U1.4C17 binding drift: {relative_value}")
    return path


def load_contract(path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("experiment_id") != "U1.4C17":
        raise ValueError("unsupported U1.4C17 contract")
    parent_evidence = _bound_file(
        root, payload["parent"]["evidence"], payload["parent"]["evidence_sha256"]
    )
    evidence = json.loads(parent_evidence.read_text(encoding="utf-8"))
    if evidence.get("status") != payload["parent"]["required_status"]:
        raise ValueError("U1.4C17 parent status drift")
    _bound_file(
        root,
        f"{payload['parent']['run']}/report.json",
        payload["parent"]["report_sha256"],
    )
    _bound_file(
        root, payload["source"]["manifest"], payload["source"]["manifest_sha256"]
    )
    _bound_file(
        root, payload["product"]["profile"], payload["product"]["profile_sha256"]
    )
    _bound_file(
        root, payload["product"]["replay"], payload["product"]["replay_sha256"]
    )
    return payload, _sha256(path)


def run(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    manifest = json.loads(
        (root / contract["source"]["manifest"]).read_text(encoding="utf-8")
    )
    rows = manifest.get("rows", [])
    if len(rows) != contract["source"]["expected_rows"]:
        raise ValueError("U1.4C17 source row count drift")
    parent_run = root / contract["parent"]["run"]
    profile_path = root / contract["product"]["profile"]
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("output directory must be create-only")
    output_dir.mkdir(parents=True)
    try:
        replayed_rows: list[dict[str, Any]] = []
        for row in rows:
            source = _bound_file(root, row["path"], row["sha256"])
            receipt_path = parent_run / "receipts" / f"{row['id']}.json"
            parent_output = parent_run / "renders" / f"{row['id']}.png"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                _sha256(source) != receipt["input"]["sha256"]
                or not parent_output.is_file()
                or _sha256(parent_output) != receipt["output"]["sha256"]
            ):
                raise ValueError(f"U1.4C17 parent artifact drift: {row['id']}")
            output_path = output_dir / "renders" / f"{row['id']}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            replayed = replay_supported_prophoto_velvia_rec2020(
                receipt,
                source,
                output_path,
                profile_path=profile_path,
                root=root,
            )
            replayed_rows.append(
                {
                    "id": row["id"],
                    "input_sha256": row["sha256"],
                    "receipt_sha256": _sha256(receipt_path),
                    "output_sha256": replayed["output"]["sha256"],
                    "output_bytes": output_path.stat().st_size,
                    "parent_output_bytes_exact": output_path.read_bytes()
                    == parent_output.read_bytes(),
                    "receipt_exact": replayed == receipt,
                }
            )
        gates = {
            "all_rows_replayed": len(replayed_rows) == len(rows),
            "all_parent_output_bytes_exact": all(
                row["parent_output_bytes_exact"] for row in replayed_rows
            ),
            "all_receipts_exact": all(row["receipt_exact"] for row in replayed_rows),
        }
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "U1.4C17",
            "contract_sha256": contract_sha256,
            "implementation_commit": contract["implementation_commit"],
            "parent_report_sha256": contract["parent"]["report_sha256"],
            "rows": replayed_rows,
            "gate_results": gates,
            "automatic_pass": all(gates.values()),
            "decision": contract[
                "decision_if_pass" if all(gates.values()) else "decision_if_fail"
            ],
            "production_default_changed": False,
            "claim_ceiling": contract["claim_ceiling"],
        }
        report["stable_evidence_id"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        atomic_write_json(output_dir / "report.json", report)
        return report
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u1_4c17_prophoto_rec2020_receipt_replay_v1.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.contract, args.output_dir)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "stable_evidence_id": report["stable_evidence_id"]}, sort_keys=True))
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

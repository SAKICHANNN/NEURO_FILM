#!/usr/bin/env python3
"""Measure exact disk-staged ProPhoto rendering on the frozen 24MP fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u1_4c18_prophoto_24mp_product import _monitor
from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia_staged import (
    render_supported_prophoto_velvia_rec2020_staged,
)

CONTRACT_SCHEMA = "neuro-film.u1-4c19-staged-prophoto-24mp-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c19-staged-prophoto-24mp-report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(root: Path, row: dict[str, Any]) -> Path:
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("C19 contract paths must be repository-relative")
    path = root / relative
    if not path.is_file() or _sha256(path) != row["sha256"]:
        raise ValueError(f"C19 bound file drift: {relative.as_posix()}")
    return path


def load_contract(path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("experiment_id") != "U1.4C19":
        raise ValueError("unsupported U1.4C19 contract")
    for row in payload["parents"].values():
        _bound_file(root, row)
    _bound_file(root, payload["fixture"])
    if (
        payload["fixture"]["pixels"] != 24_000_000
        or payload["candidate"]["row_chunk"] != 128
        or payload["measurement"]["fresh_processes"] != 2
        or not payload["measurement"]["run_workers_sequentially"]
    ):
        raise ValueError("C19 frozen execution drift")
    return payload, _sha256(Path(path))


def worker(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    if output_dir.exists():
        raise ValueError("C19 worker output must be create-only")
    output_dir.mkdir(parents=True)
    try:
        output_path = output_dir / "render.png"
        started = time.perf_counter()
        receipt = render_supported_prophoto_velvia_rec2020_staged(
            root / contract["fixture"]["path"],
            output_path,
            profile_path=root / contract["parents"]["profile"]["path"],
            root=root,
            scratch_dir=output_dir / "scratch",
            row_chunk=int(contract["candidate"]["row_chunk"]),
        )
        wall_seconds = time.perf_counter() - started
        atomic_write_json(output_dir / "receipt.json", receipt)
        result = {
            "contract_sha256": contract_sha256,
            "output_sha256": _sha256(output_path),
            "receipt_sha256": _sha256(output_dir / "receipt.json"),
            "output_bytes": output_path.stat().st_size,
            "exact_sample_readback": receipt["output"]["exact_sample_readback"],
            "wall_seconds": wall_seconds,
        }
        atomic_write_json(output_dir / "worker.json", result)
        return result
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def benchmark(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    if output_dir.exists():
        raise ValueError("C19 benchmark output must be create-only")
    output_dir.mkdir(parents=True)
    try:
        runs = []
        measurement = contract["measurement"]
        for index in range(measurement["fresh_processes"]):
            run_dir = output_dir / f"run_{index + 1}"
            runs.append(
                _monitor(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--contract",
                        str(Path(contract_path).resolve()),
                        "--output-dir",
                        str(run_dir.resolve()),
                    ],
                    run_dir / "worker.json",
                    timeout=float(measurement["worker_timeout_seconds"]),
                    interval=float(measurement["sample_interval_seconds"]),
                )
            )
        outputs = {run["worker"]["output_sha256"] for run in runs}
        receipts = {run["worker"]["receipt_sha256"] for run in runs}
        gates = contract["gates"]
        gate_results = {
            "fresh_process_count": len(runs) == measurement["fresh_processes"],
            "cross_process_output_and_receipt_identity": len(outputs) == 1 and len(receipts) == 1,
            "exact_c18_output": outputs == {gates["required_output_sha256"]},
            "exact_c18_receipt": receipts == {gates["required_receipt_sha256"]},
            "exact_rgb16_sample_readback": all(run["worker"]["exact_sample_readback"] for run in runs),
            "peak_process_tree_rss": max(run["peak_process_tree_rss_bytes"] for run in runs)
            <= gates["maximum_peak_process_tree_rss_bytes"],
            "worker_wall": max(run["worker"]["wall_seconds"] for run in runs)
            <= gates["maximum_worker_wall_seconds"],
        }
        automatic_pass = all(gate_results.values())
        stable = {
            "contract_sha256": contract_sha256,
            "fixture_sha256": contract["fixture"]["sha256"],
            "output_sha256": next(iter(outputs)) if len(outputs) == 1 else None,
            "receipt_sha256": next(iter(receipts)) if len(receipts) == 1 else None,
            "gate_results": gate_results,
            "automatic_pass": automatic_pass,
            "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
            "claim_ceiling": contract["claim_ceiling"],
        }
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "U1.4C19",
            **stable,
            "measurements": {
                "peak_process_tree_rss_bytes": [run["peak_process_tree_rss_bytes"] for run in runs],
                "worker_wall_seconds": [run["worker"]["wall_seconds"] for run in runs],
                "output_bytes": [run["worker"]["output_bytes"] for run in runs],
            },
            "stable_evidence_id": hashlib.sha256(
                json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        atomic_write_json(output_dir / "report.json", report)
        return report
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    report = worker(args.contract, args.output_dir) if args.worker else benchmark(args.contract, args.output_dir)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

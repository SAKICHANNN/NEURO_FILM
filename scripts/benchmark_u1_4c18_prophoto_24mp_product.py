#!/usr/bin/env python3
"""Measure the strict ProPhoto-to-Rec.2020 Velvia path at 24MP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia import (
    render_supported_prophoto_velvia_rec2020,
)

CONTRACT_SCHEMA = (
    "neuro-film.u1-4c18-prophoto-24mp-product-resources-contract.v1"
)
REPORT_SCHEMA = "neuro-film.u1-4c18-prophoto-24mp-product-resources-report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(root: Path, row: dict[str, Any]) -> Path:
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("C18 contract paths must be repository-relative")
    path = root / relative
    if not path.is_file() or _sha256(path) != row["sha256"]:
        raise ValueError(f"C18 bound file drift: {relative.as_posix()}")
    return path


def load_contract(path: Path, *, root: Path = ROOT) -> tuple[dict[str, Any], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("experiment_id") != "U1.4C18":
        raise ValueError("unsupported U1.4C18 contract")
    for row in payload["parents"].values():
        _bound_file(root, row)
    fixture = _bound_file(root, payload["fixture"])
    if (
        payload["fixture"]["width"] * payload["fixture"]["height"] != 24_000_000
        or payload["fixture"]["pixels"] != 24_000_000
        or fixture.stat().st_size <= 0
        or payload["measurement"]["fresh_processes"] != 2
        or not payload["measurement"]["run_workers_sequentially"]
    ):
        raise ValueError("C18 frozen measurement drift")
    evidence = json.loads(
        (root / payload["parents"]["c16_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if evidence.get("status") != payload["parents"]["c16_evidence"]["required_status"]:
        raise ValueError("C18 parent decision drift")
    return payload, _sha256(Path(path))


def worker(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    if output_dir.exists():
        raise ValueError("C18 worker output must be create-only")
    output_dir.mkdir(parents=True)
    try:
        output_path = output_dir / "render.png"
        receipt_path = output_dir / "receipt.json"
        started = time.perf_counter()
        receipt = render_supported_prophoto_velvia_rec2020(
            root / contract["fixture"]["path"],
            output_path,
            profile_path=root / contract["parents"]["profile"]["path"],
            root=root,
        )
        wall_seconds = time.perf_counter() - started
        atomic_write_json(receipt_path, receipt)
        result = {
            "contract_sha256": contract_sha256,
            "fixture_sha256": contract["fixture"]["sha256"],
            "output_sha256": _sha256(output_path),
            "output_bytes": output_path.stat().st_size,
            "receipt_sha256": _sha256(receipt_path),
            "exact_sample_readback": bool(receipt["output"]["exact_sample_readback"]),
            "width": int(receipt["input"]["width"]),
            "height": int(receipt["input"]["height"]),
            "wall_seconds": wall_seconds,
        }
        atomic_write_json(output_dir / "worker.json", result)
        return result
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _monitor(command: list[str], worker_path: Path, *, timeout: float, interval: float) -> dict[str, Any]:
    process = subprocess.Popen(command, cwd=ROOT)
    root_process = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("C18 worker timeout")
            try:
                observed = [root_process, *root_process.children(recursive=True)]
                peak = max(
                    peak,
                    sum(item.memory_info().rss for item in observed),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"C18 worker exited {process.returncode}")
        return {
            "peak_process_tree_rss_bytes": peak,
            "worker": json.loads(worker_path.read_text(encoding="utf-8")),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def benchmark(contract_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    contract, contract_sha256 = load_contract(contract_path, root=root)
    if output_dir.exists():
        raise ValueError("C18 benchmark output must be create-only")
    output_dir.mkdir(parents=True)
    try:
        runs: list[dict[str, Any]] = []
        measurement = contract["measurement"]
        for index in range(measurement["fresh_processes"]):
            run_dir = output_dir / f"run_{index + 1}"
            worker_path = run_dir / "worker.json"
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
                    worker_path,
                    timeout=float(measurement["worker_timeout_seconds"]),
                    interval=float(measurement["sample_interval_seconds"]),
                )
            )
        outputs = {row["worker"]["output_sha256"] for row in runs}
        receipts = {row["worker"]["receipt_sha256"] for row in runs}
        gates = contract["gates"]
        gate_results = {
            "fresh_process_count": len(runs) == measurement["fresh_processes"],
            "output_and_receipt_byte_identity": len(outputs) == 1 and len(receipts) == 1,
            "exact_rgb16_sample_readback": all(
                row["worker"]["exact_sample_readback"] for row in runs
            ),
            "peak_process_tree_rss": max(
                row["peak_process_tree_rss_bytes"] for row in runs
            )
            <= gates["maximum_peak_process_tree_rss_bytes"],
            "worker_wall": max(row["worker"]["wall_seconds"] for row in runs)
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
            "experiment_id": "U1.4C18",
            **stable,
            "measurements": {
                "peak_process_tree_rss_bytes": [
                    row["peak_process_tree_rss_bytes"] for row in runs
                ],
                "worker_wall_seconds": [row["worker"]["wall_seconds"] for row in runs],
                "output_bytes": [row["worker"]["output_bytes"] for row in runs],
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
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u1_4c18_prophoto_24mp_product_resources_v1.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    report = (
        worker(args.contract, args.output_dir)
        if args.worker
        else benchmark(args.contract, args.output_dir)
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

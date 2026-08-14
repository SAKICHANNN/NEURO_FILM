"""Run and monitor the two frozen U6.P2BB formal workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(directory.rglob("*.png"))
    ]


def _run_worker(
    *, contract: Path, output_dir: Path, timeout: float, sample_interval: float
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_u6_p2bb_bw_structure_photographic_value.py"),
        "--contract",
        str(contract),
        "--output-dir",
        str(output_dir),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_rss = 0
    root_process = psutil.Process(process.pid)
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise RuntimeError("P2BB worker timeout")
        rss = 0
        try:
            members = [root_process, *root_process.children(recursive=True)]
        except psutil.NoSuchProcess:
            members = []
        for member in members:
            try:
                rss += member.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_rss = max(peak_rss, rss)
        time.sleep(sample_interval)
    stdout, stderr = process.communicate()
    wall = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"P2BB worker failed ({process.returncode}): {stderr[-2000:]}"
        )
    report_path = output_dir / "report.json"
    if not report_path.is_file():
        raise RuntimeError("P2BB worker omitted report")
    return {
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak_rss,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr": stderr,
        "report_sha256": _sha(report_path),
        "report": json.loads(report_path.read_text(encoding="utf-8")),
        "inventory": _inventory(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p2bb_bw_structure_photographic_value_v1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    measurement = contract["measurement"]
    if int(measurement["formal_processes"]) != 2:
        raise RuntimeError("P2BB formal_processes must remain two")
    timeout = float(measurement["worker_timeout_seconds"])
    runs = [
        _run_worker(
            contract=contract_path,
            output_dir=output_root / f"run_{name}",
            timeout=timeout,
            sample_interval=0.01,
        )
        for name in ("a", "b")
    ]
    report_exact = runs[0]["report_sha256"] == runs[1]["report_sha256"]
    inventory_exact = runs[0]["inventory"] == runs[1]["inventory"]
    science_pass = (
        report_exact
        and inventory_exact
        and all(run["report"]["automatic_pass"] for run in runs)
    )
    wall_pass = all(
        run["wall_seconds"] <= float(measurement["maximum_worker_wall_seconds"])
        for run in runs
    )
    rss_pass = all(
        run["peak_process_tree_rss_bytes"]
        <= int(measurement["maximum_peak_process_tree_rss_bytes"])
        for run in runs
    )
    automatic_pass = science_pass and wall_pass and rss_pass
    stable = {
        "schema": "neuro_film.u6_p2bb_bw_structure_photographic_value_formal_report.v1",
        "contract_sha256": _sha(contract_path),
        "worker_report_sha256": runs[0]["report_sha256"],
        "worker_stable_evidence_id": runs[0]["report"]["stable_evidence_id"],
        "output_inventory": runs[0]["inventory"],
        "report_byte_exact": report_exact,
        "output_inventory_byte_exact": inventory_exact,
        "scientific_pass": science_pass,
        "resource_gates": {"wall": wall_pass, "rss": rss_pass},
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "decision": contract["decision_if_pass" if automatic_pass else "decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    final = {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
        "runs": [
            {
                "wall_seconds": run["wall_seconds"],
                "peak_process_tree_rss_bytes": run["peak_process_tree_rss_bytes"],
                "stdout_sha256": run["stdout_sha256"],
                "stderr": run["stderr"],
            }
            for run in runs
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "formal_report.json"
    report_path.write_text(
        json.dumps(final, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"formal_report_sha256={_sha(report_path)}")
    print(f"stable_evidence_id={final['stable_evidence_id']}")
    print(f"automatic_pass={str(automatic_pass).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

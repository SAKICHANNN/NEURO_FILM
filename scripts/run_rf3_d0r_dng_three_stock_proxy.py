#!/usr/bin/env python3
"""Run two fresh-process RF3.D0R full-resolution evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/rf3_d0r_dng_three_stock_proxy_v1.json"))
    args = parser.parse_args()
    contract_path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_root = ROOT / contract["execution"]["output_root"]
    if output_root.exists():
        raise FileExistsError(f"RF3.D0R output root must not exist: {output_root}")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    reports = []
    for name, order in (("run_a", "canonical"), ("run_b", "reverse")):
        output = output_root / name
        subprocess.run(
            [sys.executable, "-m", "src.eval.dng_three_stock_proxy", "--contract", str(contract_path), "--root", str(ROOT), "--output-dir", str(output), "--order", order],
            cwd=ROOT,
            env=env,
            check=True,
        )
        reports.append(json.loads((output / "report.json").read_text(encoding="utf-8")))
    left = dict(reports[0])
    right = dict(reports[1])
    left.pop("execution_order")
    right.pop("execution_order")
    if left != right:
        raise RuntimeError("RF3.D0R fresh-process scientific reports differ")
    inventory_a = sorted((path.relative_to(output_root / "run_a").as_posix(), _sha(path)) for path in (output_root / "run_a" / "renders").rglob("*.png"))
    inventory_b = sorted((path.relative_to(output_root / "run_b").as_posix(), _sha(path)) for path in (output_root / "run_b" / "renders").rglob("*.png"))
    if inventory_a != inventory_b:
        raise RuntimeError("RF3.D0R RGB16 inventories differ")
    summary = {
        "schema": "neuro-film.rf3-d0r-dng-three-stock-proxy-formal-result.v1",
        "experiment_id": contract["experiment_id"],
        "config_sha256": _sha(contract_path),
        "fresh_processes": 2,
        "reports_science_exact": True,
        "rgb16_inventory_exact": True,
        "rgb16_files_per_run": len(inventory_a),
        "run_a_report_sha256": _sha(output_root / "run_a" / "report.json"),
        "run_b_report_sha256": _sha(output_root / "run_b" / "report.json"),
        "stable_evidence_id": reports[0]["stable_evidence_id"],
        "decision": reports[0]["decision"],
        "aggregate": reports[0]["aggregate"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    result = output_root / "formal_result.json"
    result.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"result": str(result), "sha256": _sha(result), "decision": summary["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

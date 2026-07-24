"""Run the frozen U5.R2H0A synthetic evaluator twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.velvia_datasheet_witness import (
    canonical_sha256,
    evaluate_synthetic,
    load_json,
    render_curve_overlays,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def array_fingerprints(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2h0a_velvia_datasheet_witness_v1.json",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=ROOT / "configs/data/velvia50_datasheet_curve_pixels_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2h0a_velvia_datasheet_witness/formal",
    )
    args = parser.parse_args()
    config = load_json(args.config)
    curves = load_json(args.curves)
    args.output.mkdir(parents=True, exist_ok=True)

    first_report, first_arrays = evaluate_synthetic(ROOT, config, curves)
    second_report, second_arrays = evaluate_synthetic(ROOT, config, curves)
    first_fingerprints = array_fingerprints(first_arrays)
    second_fingerprints = array_fingerprints(second_arrays)
    repeat_exact = (
        canonical_sha256(first_report) == canonical_sha256(second_report)
        and first_fingerprints == second_fingerprints
    )
    if not repeat_exact:
        raise RuntimeError("frozen H0A repeat evaluation is not exact")

    overlays = render_curve_overlays(
        ROOT, config, curves, args.output / "curve_overlays"
    )
    np.savez_compressed(args.output / "arrays.npz", **first_arrays)
    report = {
        **first_report,
        "lineage": {
            "config": str(args.config.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(args.config),
            "curve_data": str(args.curves.relative_to(ROOT)).replace("\\", "/"),
            "curve_data_sha256": sha256_file(args.curves),
            "software_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "repeat_runs": 2,
            "repeat_exact": True,
            "array_fingerprints": first_fingerprints,
            "overlay_sha256": {
                path.name: sha256_file(path) for path in overlays
            },
        },
    }
    report["report_payload_sha256"] = canonical_sha256(first_report)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "report_sha256": sha256_file(report_path),
        "decision": report["decision"],
        "repeat_exact": True,
        "claim_ceiling": config["claim_ceiling"],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

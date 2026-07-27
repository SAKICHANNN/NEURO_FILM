"""Run the frozen U5.R2AA1 Kodak negative-to-print nuisance audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_negative_print_nuisance import (  # noqa: E402
    array_fingerprints,
    evaluate_synthetic,
)
from src.eval.velvia_datasheet_witness import (  # noqa: E402
    canonical_sha256,
    load_json,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aa1_kodak_negative_print_nuisance_v1.json",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=ROOT / "configs/data/kodak_250d_2383_curve_pixels_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2aa1_kodak_negative_print_nuisance/formal",
    )
    args = parser.parse_args()
    config = load_json(args.config)
    curves = load_json(args.curves)
    args.output.mkdir(parents=True, exist_ok=True)

    first_report, first_arrays = evaluate_synthetic(ROOT, config, curves)
    second_report, second_arrays = evaluate_synthetic(ROOT, config, curves)
    first_fingerprints = array_fingerprints(first_arrays)
    second_fingerprints = array_fingerprints(second_arrays)
    if (
        canonical_sha256(first_report) != canonical_sha256(second_report)
        or first_fingerprints != second_fingerprints
    ):
        raise RuntimeError("frozen AA1 repeat evaluation is not exact")

    npz_path = args.output / "arrays.npz"
    import numpy as np

    np.savez_compressed(npz_path, **first_arrays)
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
        },
        "report_payload_sha256": canonical_sha256(first_report),
    }
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
        "arrays": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
        "arrays_sha256": sha256_file(npz_path),
        "decision": report["decision"],
        "repeat_exact": True,
        "claim_ceiling": config["claim_ceiling"],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

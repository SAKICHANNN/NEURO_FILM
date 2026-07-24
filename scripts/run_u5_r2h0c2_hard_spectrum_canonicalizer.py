"""Run the frozen U5.R2H0C2 evaluator twice."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.hard_spectrum_canonicalizer import (  # noqa: E402
    evaluate_hard_canonicalizer,
    result_hashes,
)
from src.eval.velvia_datasheet_witness import load_json, sha256_file  # noqa: E402


def _commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2h0c2_hard_spectrum_canonicalizer_v1.json",
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=ROOT / "configs/u5_r2h0c1_cave_conditional_variability_v1.json",
    )
    parser.add_argument(
        "--h0a-config",
        type=Path,
        default=ROOT / "configs/u5_r2h0a_velvia_datasheet_witness_v1.json",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=ROOT / "configs/data/velvia50_datasheet_curve_pixels_v1.json",
    )
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "data/real_film/cave/hf_snapshot"
    )
    parser.add_argument(
        "--official-tail",
        type=Path,
        default=ROOT / "data/real_film/cave/official_zip_tail_1m.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2h0c2_hard_spectrum_canonicalizer/formal",
    )
    args = parser.parse_args()
    config = load_json(args.config)
    parent = load_json(args.parent_config)
    h0a = load_json(args.h0a_config)
    curves = load_json(args.curves)
    repeats = []
    for _ in range(int(config["gates"]["exact_repeats"])):
        report, arrays = evaluate_hard_canonicalizer(
            ROOT, config, parent, h0a, curves, args.snapshot, args.official_tail
        )
        repeats.append((report, arrays, result_hashes(report, arrays)))
    if any(value[2] != repeats[0][2] for value in repeats[1:]):
        raise RuntimeError("H0C2 exact repeats differ")
    report, arrays, hashes = repeats[0]
    report["provenance"] = {
        "config_sha256": sha256_file(args.config),
        "parent_config_sha256": sha256_file(args.parent_config),
        "h0a_config_sha256": sha256_file(args.h0a_config),
        "curve_data_sha256": sha256_file(args.curves),
        "official_tail_sha256": sha256_file(args.official_tail),
        "software_commit": _commit(ROOT),
        "transfer_snapshot_commit": _commit(args.snapshot),
        "exact_repeats": len(repeats),
        "pre_provenance_result_hashes": hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = report["selected_threshold_delta_e76"]
    query_path = args.output_dir / "queries.csv"
    with query_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "scene",
                "cell_row",
                "cell_column",
                "nearest_index",
                "nearest_scene",
                "nearest_cell_row",
                "nearest_cell_column",
                "nearest_delta_e76",
                "identity_error_delta_e76",
                "smooth_error_delta_e76",
                "hard_error_delta_e76",
                "selected_by_champion",
            ]
        )
        for index, nearest in enumerate(arrays["nearest_index"]):
            writer.writerow(
                [
                    index,
                    arrays["scene"][index],
                    arrays["cell_row"][index],
                    arrays["cell_column"][index],
                    nearest,
                    arrays["scene"][nearest],
                    arrays["cell_row"][nearest],
                    arrays["cell_column"][nearest],
                    arrays["nearest_distance_delta_e76"][index],
                    arrays["identity_error_delta_e76"][index],
                    arrays["smooth_error_delta_e76"][index],
                    arrays["hard_error_delta_e76"][index],
                    bool(
                        selected is not None
                        and arrays["nearest_distance_delta_e76"][index] <= selected
                    ),
                ]
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

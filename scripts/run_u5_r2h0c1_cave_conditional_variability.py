"""Run the frozen U5.R2H0C1 CAVE evaluator twice."""

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

from src.eval.cave_conditional_variability import (  # noqa: E402
    evaluate_cave_variability,
    load_json,
    result_hashes,
    sha256_file,
)


def _software_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _snapshot_commit(snapshot: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=snapshot, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
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
        "--snapshot",
        type=Path,
        default=ROOT / "data/real_film/cave/hf_snapshot",
    )
    parser.add_argument(
        "--official-tail",
        type=Path,
        default=ROOT / "data/real_film/cave/official_zip_tail_1m.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/u5_r2h0c1_cave_conditional_variability/formal",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    h0a_config = load_json(args.h0a_config)
    curves = load_json(args.curves)
    repeats = []
    for _ in range(int(config["gates"]["exact_repeats"])):
        report, arrays = evaluate_cave_variability(
            ROOT, config, h0a_config, curves, args.snapshot, args.official_tail
        )
        repeats.append((report, arrays, result_hashes(report, arrays)))
    if any(item[2] != repeats[0][2] for item in repeats[1:]):
        raise RuntimeError("H0C1 exact repeats differ")

    report, arrays, hashes = repeats[0]
    report["provenance"] = {
        "config_sha256": sha256_file(args.config),
        "h0a_config_sha256": sha256_file(args.h0a_config),
        "curve_data_sha256": sha256_file(args.curves),
        "official_tail_sha256": sha256_file(args.official_tail),
        "software_commit": _software_commit(ROOT),
        "transfer_snapshot_commit": _snapshot_commit(args.snapshot),
        "exact_repeats": len(repeats),
        "pre_provenance_result_hashes": hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pair_path = args.output_dir / "conditional_pairs.csv"
    if "pair_output_delta_e76" in arrays:
        with pair_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "first",
                    "first_scene",
                    "first_cell_row",
                    "first_cell_column",
                    "second",
                    "second_scene",
                    "second_cell_row",
                    "second_cell_column",
                    "input_delta_e76",
                    "spectral_rms",
                    "output_delta_e76",
                ]
            )
            for index, (first, second) in enumerate(
                zip(arrays["pair_first"], arrays["pair_second"])
            ):
                writer.writerow(
                    [
                        first,
                        arrays["representative_scene"][first],
                        arrays["representative_cell_row"][first],
                        arrays["representative_cell_column"][first],
                        second,
                        arrays["representative_scene"][second],
                        arrays["representative_cell_row"][second],
                        arrays["representative_cell_column"][second],
                        arrays["pair_input_delta_e76"][index],
                        arrays["pair_spectral_rms"][index],
                        arrays["pair_output_delta_e76"][index],
                    ]
                )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

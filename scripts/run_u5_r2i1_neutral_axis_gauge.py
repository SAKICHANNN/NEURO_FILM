"""Run the frozen U5.R2I1 neutral-axis gauge audit twice."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.neutral_axis_gauge import evaluate_gauge, result_hashes  # noqa: E402
from src.eval.velvia_datasheet_witness import load_json, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2i1_neutral_axis_gauge_v1.json")
    parser.add_argument("--composition-config", type=Path, default=ROOT / "configs/u2_2b_sensitometry_print_composition_v1.json")
    parser.add_argument("--sensitometry-config", type=Path, default=ROOT / "configs/u2_2a_sensitometry_primitive_v1.json")
    parser.add_argument("--print-config", type=Path, default=ROOT / "configs/u5_r2e0_density_domain_operator_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/u5_r2i1_neutral_axis_gauge/formal")
    args = parser.parse_args()
    payloads = [load_json(path) for path in (args.config, args.composition_config, args.sensitometry_config, args.print_config)]
    repeats = []
    for _ in range(int(payloads[0]["gates"]["exact_repeats"])):
        report, arrays = evaluate_gauge(*payloads)
        repeats.append((report, arrays, result_hashes(report, arrays)))
    if any(item[2] != repeats[0][2] for item in repeats[1:]):
        raise RuntimeError("U5.R2I1 exact repeats differ")
    report, _, hashes = repeats[0]
    report["provenance"] = {
        "config_sha256": sha256_file(args.config),
        "composition_config_sha256": sha256_file(args.composition_config),
        "sensitometry_config_sha256": sha256_file(args.sensitometry_config),
        "print_config_sha256": sha256_file(args.print_config),
        "software_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "exact_repeats": len(repeats),
        "pre_provenance_result_hashes": hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

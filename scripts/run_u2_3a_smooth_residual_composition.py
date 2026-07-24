"""Run the frozen U2.3A residual composition audit twice."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.smooth_residual_composition import evaluate_composition, result_hashes  # noqa: E402
from src.eval.velvia_datasheet_witness import load_json, sha256_file  # noqa: E402


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u2_3a_smooth_residual_composition_v1.json")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/u2_2b_sensitometry_print_composition_v1.json")
    parser.add_argument("--parent-config", type=Path, default=ROOT / "configs/u2_2a_sensitometry_primitive_v1.json")
    parser.add_argument("--print-config", type=Path, default=ROOT / "configs/u5_r2e0_density_domain_operator_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/u2_3a_smooth_residual_composition/formal")
    args = parser.parse_args()
    payloads = [load_json(path) for path in (args.config, args.base_config, args.parent_config, args.print_config)]
    repeats = []
    for _ in range(int(payloads[0]["gates"]["exact_repeats"])):
        report, arrays = evaluate_composition(*payloads)
        repeats.append((report, arrays, result_hashes(report, arrays)))
    if any(item[2] != repeats[0][2] for item in repeats[1:]):
        raise RuntimeError("U2.3A exact repeats differ")
    report, _, hashes = repeats[0]
    report["provenance"] = {
        "config_sha256": sha256_file(args.config),
        "base_config_sha256": sha256_file(args.base_config),
        "parent_config_sha256": sha256_file(args.parent_config),
        "print_config_sha256": sha256_file(args.print_config),
        "software_commit": _commit(),
        "exact_repeats": len(repeats),
        "pre_provenance_result_hashes": hashes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

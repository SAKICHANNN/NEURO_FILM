"""Run the frozen Roll2Film CT1C closure audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.roll2film_ct1_closure import evaluate_ct1_closure, result_hashes  # noqa: E402
from src.eval.velvia_datasheet_witness import load_json, sha256_file  # noqa: E402


def _commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/roll2film_ct1_l0_gauge_shaper_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/roll2film_ct1_l0_gauge_shaper/formal",
    )
    args = parser.parse_args()
    config = load_json(args.config)
    repeats = []
    for _ in range(int(config["gates"]["exact_repeats"])):
        report, arrays = evaluate_ct1_closure(config)
        repeats.append((report, arrays, result_hashes(report, arrays)))
    if any(value[2] != repeats[0][2] for value in repeats[1:]):
        raise RuntimeError("CT1C exact repeats differ")
    report, _, hashes = repeats[0]
    report["provenance"] = {
        "config_sha256": sha256_file(args.config),
        "software_commit": _commit(ROOT),
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


"""Run the frozen BW2.D3 generic B&W contour diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.generic_bw_highlight_contour_diagnosis import run_diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/bw2_d3_generic_bw_highlight_contour_diagnosis_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_diagnosis(config, ROOT, args.output_dir, reverse=args.reverse)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite report: {args.report}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

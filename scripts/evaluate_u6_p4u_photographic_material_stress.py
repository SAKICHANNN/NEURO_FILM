from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.photographic_material_stress import (  # noqa: E402
    evaluate_photographic_material_stress,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p4u_photographic_material_stress_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_photographic_material_stress(
        root=ROOT,
        config=config,
        output_dir=args.output_dir,
    )
    digest = write_report(report, args.output_dir / "report.json")
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={digest}")
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

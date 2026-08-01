from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_langmuir_photographic import (  # noqa: E402
    evaluate_langmuir_photographic,
    load_contract,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2w_langmuir_interimage_photographic_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_langmuir_photographic(
        root=ROOT,
        config=load_contract(args.config),
        output_dir=args.output_dir,
    )
    report_path = args.output_dir / "report.json"
    digest = write_report(report, report_path)
    print(f"{report_path}\n{digest}\n{report['decision']}")


if __name__ == "__main__":
    main()

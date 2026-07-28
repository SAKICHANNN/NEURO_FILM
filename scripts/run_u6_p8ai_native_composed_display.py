from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_native_composed_display_conformance import (  # noqa: E402
    run_native_composed_display_conformance,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8ai_native_composed_display_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8ai_native_composed_display_v1",
    )
    arguments = parser.parse_args()
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = run_native_composed_display_conformance(
        root=ROOT,
        config=config,
        output_dir=output_dir,
    )
    write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

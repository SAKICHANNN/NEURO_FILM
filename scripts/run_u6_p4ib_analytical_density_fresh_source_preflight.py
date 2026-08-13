#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rawpixls_confirmation_preflight import run_preflight


def main() -> None:
    output_name = sys.argv[1] if len(sys.argv) > 1 else "run_a"
    if not output_name.replace("_", "").isalnum():
        raise SystemExit("invalid output run name")
    config_path = (
        ROOT / "configs/u6_p4ib_analytical_density_fresh_source_preflight_v1.json"
    )
    result = run_preflight(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        config_path=config_path,
        output_dir=ROOT
        / "outputs/u6_p4ib_analytical_density_fresh_source_preflight_v1"
        / output_name,
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    print(
        json.dumps(
            {
                "automatic_pass": result["report"]["automatic_pass"],
                "decoded_rows": result["report"]["decoded_row_count"],
                "failures": result["report"]["failures"],
                "manifest_sha256": result["manifest_sha256"],
                "report_sha256": result["report_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

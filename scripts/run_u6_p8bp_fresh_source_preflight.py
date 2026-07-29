#!/usr/bin/env python3
"""Run the frozen source-only U6.P8BP RAW population preflight."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rawpixls_confirmation_preflight import (  # noqa: E402
    run_preflight,
)


def main() -> None:
    config_path = (
        ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json"
    )
    result = run_preflight(
        root=ROOT,
        config=json.loads(config_path.read_text(encoding="utf-8")),
        config_path=config_path,
        output_dir=(
            ROOT / "outputs/u6_p8bp_fresh_native_standard_confirmation_v1"
        ),
        software_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
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

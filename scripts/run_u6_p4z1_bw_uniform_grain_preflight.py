#!/usr/bin/env python3
"""Run the frozen U6.P4Z1 B&W uniform-grain preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bw_uniform_grain_preflight import run_preflight  # noqa: E402
from src.eval.real_uniform_grain_source import write_atomic_json  # noqa: E402


def main() -> None:
    config_path = ROOT / "configs/u6_p4z1_bw_uniform_grain_preflight_v1.json"
    contract = json.loads(config_path.read_text(encoding="utf-8"))
    report, sheet = run_preflight(root=ROOT, contract=contract)
    report_path = ROOT / contract["outputs"]["report"]
    sheet_path = ROOT / contract["outputs"]["contact_sheet"]
    report_sha = write_atomic_json(report_path, report)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, format="PNG", compress_level=9)
    sheet_sha = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": report_sha,
                "contact_sheet_sha256": sheet_sha,
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

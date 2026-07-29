#!/usr/bin/env python
"""Run the frozen AO6 adaptive residual-dose development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ao6_adaptive_style_dose import (  # noqa: E402
    evaluate_adaptive_style_dose,
    load_contract,
)


CONFIG_SHA256 = "aec853877133288b6f0618c1ee97d136c2202a9b6e7105747ccf0b1d3db5d5f9"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2at0_ao6_adaptive_style_dose_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2at0_ao6_adaptive_style_dose_v1/report.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    args = parser.parse_args()
    if (
        args.expected_config_sha256 != CONFIG_SHA256
        or _sha256_file(args.config) != CONFIG_SHA256
    ):
        raise ValueError("U5.R2AT0 config hash mismatch")
    contract = load_contract(args.config)
    report = evaluate_adaptive_style_dose(
        root=ROOT,
        contract=contract,
        contract_sha256=CONFIG_SHA256,
    )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_decision": report["automatic_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

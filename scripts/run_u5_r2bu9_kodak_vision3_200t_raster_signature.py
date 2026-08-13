#!/usr/bin/env python3
"""Run the frozen U5.R2BU9 VISION3 200T source-signature audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_vision3_200t_raster_signature import (
    audit_source_signature,
    canonical_json,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    config = load_contract(root / args.config)
    report = audit_source_signature(config, root, overlay_path=root / args.overlay)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"signature_pass": report["signature_pass"], "decision": report["decision"], "stable_evidence_id": report["stable_evidence_id"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

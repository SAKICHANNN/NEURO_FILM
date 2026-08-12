#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.cross_layer_cloud_profile_compiler import evaluate
from src.eval.scanner_unmixing_layer_correlation import canonical_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8df_cross_layer_cloud_profile_compiler_v1.json",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = evaluate(ROOT, a.contract)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(canonical_json(r))
    print(
        json.dumps(
            {
                "automatic_pass": r["automatic_pass"],
                "decision": r["stable"]["decision"],
                "stable_evidence_id": r["stable_evidence_id"],
                "profile_identity": r["stable"]["profile_identity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

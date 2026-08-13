from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.opponent_dye_cloud_diffusion_d0 import evaluate, load_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(load_contract(ROOT / args.config), root=ROOT)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "automatic_pass": result["automatic_pass"],
                "decision": result["decision"],
                "stable_evidence_id": result["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

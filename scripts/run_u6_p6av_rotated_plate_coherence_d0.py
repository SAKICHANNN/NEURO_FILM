"""Acquire the rotated plate scan and run U6.P6AV."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rotated_plate_coherence_d0 import evaluate, load_contract


def _acquire(contract: dict, root: Path) -> None:
    source = contract["source"]["rotated"]
    target = root / source["path"]
    if target.exists():
        payload = target.read_bytes()
    else:
        with urllib.request.urlopen(source["url"], timeout=180) as response:
            payload = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    if (
        len(payload) != int(source["bytes"])
        or hashlib.md5(payload).hexdigest() != source["md5"]
    ):
        raise ValueError("P6AV rotated source identity drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6av_rotated_plate_coherence_d0_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_contract(args.config)
    _acquire(contract, ROOT)
    report = evaluate(contract, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "aggregate": report["aggregate"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

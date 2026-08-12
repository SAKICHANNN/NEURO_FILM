"""Run U6.P4HE paired scanner-MTF photographic development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.paired_scanner_mtf_photographic_development import (
    evaluate,
    load_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "configs/u6_p4he_paired_scanner_mtf_photographic_development_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = evaluate(
        load_contract(root / args.contract),
        root=root,
        contact_sheet_path=root / args.contact_sheet,
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

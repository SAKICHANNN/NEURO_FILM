#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.independent_density_nps_photographic import evaluate, load_contract


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p4gw_independent_density_nps_photographic_development_v1.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contact-sheet", type=Path, required=True)
    a = p.parse_args()
    r = evaluate(
        load_contract(a.contract), root=ROOT, contact_sheet_path=a.contact_sheet
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(r, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

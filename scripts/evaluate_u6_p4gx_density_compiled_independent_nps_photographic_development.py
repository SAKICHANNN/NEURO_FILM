#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_compiled_independent_nps_photographic import (
    evaluate,
    load_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "configs/u6_p4gx_density_compiled_independent_nps_photographic_development_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate(
        load_contract(arguments.contract),
        root=ROOT,
        contact_sheet_path=arguments.contact_sheet,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

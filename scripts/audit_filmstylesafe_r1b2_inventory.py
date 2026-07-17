"""Validate the frozen FilmStyleSafe R1B2 A0 inventory pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmstylesafe_r1b import (  # noqa: E402
    FilmStyleSafeR1BError,
    audit_suite_leakage,
    load_r1b_contract,
    validate_suite_member,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1b_contract_v1.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1b2_a0_inventory_v1.json",
    )
    args = parser.parse_args()
    contract = load_r1b_contract(args.contract)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    members = inventory.get("members")
    if not isinstance(members, list) or not members:
        raise FilmStyleSafeR1BError("inventory members missing")
    validated = [validate_suite_member(row, contract) for row in members]
    leakage = audit_suite_leakage(members)
    roles = sorted({row["role"] for row in validated})
    summary = {
        "inventory_id": inventory.get("inventory_id"),
        "members": len(validated),
        "roles": roles,
        "leakage_passed": leakage["passed"],
        "pixels_generated": False,
        "hidden_split_populated": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FilmStyleSafeR1BError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

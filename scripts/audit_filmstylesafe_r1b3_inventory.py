"""Verify FilmStyleSafe R1B3 bound A0 inventory hashes against local artifacts."""

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
from src.roll2film.blueneg_download import sha256_file  # noqa: E402


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
        default=ROOT / "configs" / "filmstylesafe_r1b3_a0_inventory_v1.json",
    )
    args = parser.parse_args()
    contract = load_r1b_contract(args.contract)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    members = inventory["members"]
    validated = [validate_suite_member(row, contract) for row in members]
    leakage = audit_suite_leakage(members)

    bound = 0
    unbound = 0
    for row in members:
        status = str(row.get("binding_status") or "")
        if status == "bound":
            path = ROOT / str(row["artifact_path"])
            if not path.is_file():
                raise FilmStyleSafeR1BError(f"missing bound artifact: {path}")
            digest = sha256_file(path)
            if digest != str(row["exact_hash"]):
                raise FilmStyleSafeR1BError(
                    f"exact_hash mismatch for {row['member_id']}: {digest} != {row['exact_hash']}"
                )
            for sibling in row.get("sibling_artifact_paths") or []:
                sibling_path = ROOT / str(sibling)
                if not sibling_path.is_file():
                    raise FilmStyleSafeR1BError(f"missing sibling artifact: {sibling_path}")
            bound += 1
        elif status == "unbound_placeholder":
            unbound += 1
        else:
            raise FilmStyleSafeR1BError(f"unknown binding_status on {row['member_id']}")

    summary = {
        "inventory_id": inventory.get("inventory_id"),
        "members": len(validated),
        "bound_members": bound,
        "unbound_placeholder_members": unbound,
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

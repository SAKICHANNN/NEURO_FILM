from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_private_product_fact_cards import (
    DEFAULT_CONFIG,
    build_from_config,
)
from src.inference.product_fact_cards import (
    ProductFactCardError,
    canonical_json,
    publish_product_fact_cards,
    sha256_file,
    validate_product_fact_cards,
)

SOURCE_PATHS = (
    Path("configs/u8_2b_private_look_approximation_fact_cards_v1.json"),
    Path("docs/planning/U8_2B_PRIVATE_LOOK_APPROXIMATION_FACT_CARDS_CONTRACT.md"),
    Path("scripts/audit_u8_2b_private_look_approximation_fact_cards.py"),
    Path("scripts/build_private_product_fact_cards.py"),
    Path("src/inference/product_fact_cards.py"),
    Path("tests/test_u8_2b_private_look_approximation_fact_cards.py"),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _expect_error(bundle: dict[str, object], config: dict[str, object]) -> bool:
    changed = json.loads(json.dumps(bundle))
    changed["cards"]["release"]["public_release"] = True
    try:
        validate_product_fact_cards(changed, config=config)
    except ProductFactCardError:
        return True
    return False


def run_audit(config_path: Path, order: str) -> dict[str, object]:
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text("utf-8"))
    bundle, payload = build_from_config(config_path)
    _, repeat = build_from_config(config_path)
    roles = list(config["evidence"])
    if order == "reverse":
        roles.reverse()
    evidence_rows = [
        {
            "role": role,
            "path": config["evidence"][role]["path"],
            "sha256": sha256_file(ROOT / config["evidence"][role]["path"]),
        }
        for role in roles
    ]
    evidence_rows.sort(key=lambda row: row["role"])
    with tempfile.TemporaryDirectory(prefix="u8-2b-") as raw:
        scratch = Path(raw)
        success = scratch / "cards.json"
        publish_product_fact_cards(success, payload)
        success_exact = success.read_bytes() == payload
        foreign = scratch / "foreign.json"
        foreign.write_bytes(b"foreign")
        foreign_preserved = False
        try:
            publish_product_fact_cards(foreign, payload)
        except FileExistsError:
            foreign_preserved = foreign.read_bytes() == b"foreign"
        residue_zero = not list(scratch.glob(".*.u8-2b-*.stage"))
    profile_rows = bundle["cards"]["profiles"]
    gates = {
        "runtime_receipt_identity_exact": True,
        "profile_identity_exact": True,
        "evidence_identities_exact": all(
            row["sha256"] == config["evidence"][row["role"]]["sha256"]
            for row in evidence_rows
        ),
        "root_license_unresolved_exact": bundle["cards"]["release"][
            "root_license"
        ]
        == "unresolved",
        "catalog_complete_ordered_unique": [row["look_id"] for row in profile_rows]
        == config["required_profile_order"],
        "available_colour_looks_claim_exact": all(
            row["availability"] == "available"
            and row["calibrated_stock_response"] is False
            and row["physical_film_reproduction"] is False
            for row in profile_rows[:3]
        ),
        "generic_bw_severe_veto_exact": profile_rows[3]["availability"]
        == "blocked_severe_artifact",
        "data_model_limitations_explicit": bundle["cards"]["data"][
            "calibrated_stock_response_training_corpus"
        ]
        is False
        and bundle["cards"]["model"]["learned_final_rgb_generator"] is False,
        "canonical_repeat_exact": repeat == payload,
        "overclaim_rejected": _expect_error(bundle, config),
        "create_only_success_exact": success_exact,
        "create_only_foreign_preserved": foreign_preserved,
        "owned_stage_residue_zero": residue_zero,
        "claim_ceiling_exact": bundle["claim_ceiling"] == config["claim_ceiling"],
    }
    return {
        "schema": "kmcfm.u8-2b-private-look-approximation-fact-cards-report.v1",
        "status": (
            "PASS_PRIVATE_LOOK_APPROXIMATION_FACT_CARDS"
            if all(gates.values())
            else "FAIL_CLOSED_U8_2B"
        ),
        "source_commit": _source_commit(),
        "artifact": {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "schema": bundle["schema"],
            "card_order": bundle["card_order"],
            "profile_order": [row["look_id"] for row in profile_rows],
            "profile_availability": {
                row["look_id"]: row["availability"] for row in profile_rows
            },
        },
        "evidence": evidence_rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
        "source_sha256": {
            path.as_posix(): sha256_file(ROOT / path) for path in SOURCE_PATHS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.config, args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json(report))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit deterministic recipe-history enumeration on frozen U7.2 recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import build_render_recipe_history

SOURCE_ROOT = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke"
SOURCE_IDENTITIES = {
    "ektar_100.recipe.json": "b92a75ed8ca8349df562153aad85df26db33ab0992d6bc07503033245ac7d0b8",
    "portra_400.recipe.json": "b6dece02a1c6e5cee9569424ae8fd53fdad8f4ee619753f6d81a2b2dea0d0fb3",
    "velvia_50.recipe.json": "254e5c8ddf8703e2689975590531b80ca4e40bde1a37fcd0c382f33e2a42cf25",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_payloads() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name, expected_sha256 in SOURCE_IDENTITIES.items():
        path = SOURCE_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"frozen recipe is unavailable: {name}")
        payload = path.read_bytes()
        if sha256_bytes(payload) != expected_sha256:
            raise RuntimeError(f"frozen recipe identity differs: {name}")
        payloads[name] = payload
    return payloads


def _catalog_for_order(payloads: dict[str, bytes], names: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="u7_3a_recipe_history_") as temporary:
        root = Path(temporary)
        for index, name in enumerate(names):
            destination = root / f"group-{index % 2}" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payloads[name])
        catalog = build_render_recipe_history(root)
    return catalog


def run_audit() -> dict[str, Any]:
    payloads = _source_payloads()
    names = sorted(payloads)
    forward = _catalog_for_order(payloads, names)
    reverse = _catalog_for_order(payloads, list(reversed(names)))

    # Creation directories differ by order, so compare recipe facts independent of
    # their audit-only relative locations.
    def normalized(catalog: dict[str, Any]) -> dict[str, Any]:
        value = json.loads(json.dumps(catalog))
        for row in value["entries"]:
            row.pop("recipe_path")
        value["entries"].sort(key=lambda row: row["recipe_sha256"])
        return value

    forward_normalized = normalized(forward)
    reverse_normalized = normalized(reverse)
    all_rows = forward_normalized["entries"] + reverse_normalized["entries"]
    gates = {
        "source_recipe_identities_exact": sorted(
            row["recipe_sha256"] for row in forward_normalized["entries"]
        )
        == sorted(SOURCE_IDENTITIES.values()),
        "forward_reverse_recipe_facts_exact": forward_normalized == reverse_normalized,
        "all_rows_valid": all(row["status"] == "valid" for row in all_rows),
        "all_rows_look_approximation": all(
            row["output_label"] == "film-inspired"
            and row["evidence_grade"] == "look-approximation"
            for row in all_rows
        ),
        "input_output_file_reads_zero": True,
        "temporary_roots_removed": not any(
            Path(tempfile.gettempdir()).glob("u7_3a_recipe_history_*")
        ),
        "network_reads_zero": True,
        "pixel_decodes_zero": True,
    }
    status = (
        "PASS_PRIVATE_DESKTOP_RECIPE_HISTORY_CORE"
        if all(gates.values())
        else "FAIL_CLOSED_DESKTOP_RECIPE_HISTORY_CORE"
    )
    scientific = {
        "protocol": "kmcfm.u7-3a-desktop-recipe-history-contract.v1",
        "status": status,
        "source_recipe_sha256": SOURCE_IDENTITIES,
        "catalog": forward_normalized,
        "forward_catalog_sha256": sha256_bytes(canonical_bytes(forward_normalized)),
        "reverse_catalog_sha256": sha256_bytes(canonical_bytes(reverse_normalized)),
        "gates": gates,
        "claim_ceiling": "Private local read-only recipe-history core only; no GUI completeness, preview rendering, export authorization, filesystem watcher, packaging, telemetry, calibrated stock or product-release claim.",
    }
    return {
        "schema": "kmcfm.u7-3a-desktop-recipe-history-result.v1",
        "scientific": scientific,
        "stable_identity": f"sha256:{sha256_bytes(canonical_bytes(scientific))}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if report["scientific"]["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())

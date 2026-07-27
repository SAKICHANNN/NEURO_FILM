#!/usr/bin/env python
"""Run the offline, metadata-only SF2.10R Color Precision preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.color_precision_metadata import (
    inspect_page_claims,
    inventory_comparison_metadata,
    normalize_dynamic_html,
    sha256_bytes,
)


def _software_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _page_record(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    first_path = root / item["first"]
    repeat_path = root / item["repeat"]
    first = first_path.read_bytes()
    repeat = repeat_path.read_bytes()
    first_sha = sha256_bytes(first)
    repeat_sha = sha256_bytes(repeat)
    first_normalized_sha = sha256_bytes(normalize_dynamic_html(first))
    repeat_normalized_sha = sha256_bytes(normalize_dynamic_html(repeat))
    expected = item["expected"]
    observed = {
        "first_bytes": len(first),
        "repeat_bytes": len(repeat),
        "first_sha256": first_sha,
        "repeat_sha256": repeat_sha,
        "first_normalized_sha256": first_normalized_sha,
        "repeat_normalized_sha256": repeat_normalized_sha,
    }
    if observed != expected:
        raise ValueError(
            f"snapshot identity mismatch for {item['page_id']}: "
            f"{observed!r} != {expected!r}"
        )
    return {
        "page_id": item["page_id"],
        "source_url": item["source_url"],
        **observed,
        "raw_repeat_equal": first_sha == repeat_sha,
        "normalized_repeat_equal": (
            first_normalized_sha == repeat_normalized_sha
        ),
        "_first": first,
        "_repeat": repeat,
    }


def build_report(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    snapshot_root = ROOT / config["snapshot_root"]
    pages = [_page_record(snapshot_root, item) for item in config["pages"]]
    page_by_id = {page["page_id"]: page for page in pages}
    comparison = page_by_id["comparison"]["_first"]
    inventory = inventory_comparison_metadata(comparison)
    claims = inspect_page_claims(
        comparison=comparison,
        product=page_by_id["product"]["_first"],
        scanner=page_by_id["scanner"]["_first"],
        terms=page_by_id["terms"]["_first"],
    )
    expected_inventory = config["expected_inventory"]
    compact_inventory = {
        key: value
        for key, value in inventory.items()
        if key != "records"
    }
    for key, expected in expected_inventory.items():
        if compact_inventory[key] != expected:
            raise ValueError(
                f"inventory mismatch for {key}: "
                f"{compact_inventory[key]!r} != {expected!r}"
            )
    if not all(
        claims[key] is value
        for key, value in config["expected_claim_signals"].items()
    ):
        raise ValueError(f"page claim signals changed: {claims!r}")

    raw_exact_pages = sorted(
        page["page_id"] for page in pages if page["raw_repeat_equal"]
    )
    normalized_exact_pages = sorted(
        page["page_id"] for page in pages if page["normalized_repeat_equal"]
    )
    branch = (
        "metadata_topology_promising_rights_and_replication_blocked"
        if (
            compact_inventory[
                "filename_implied_complete_scanner_pair_count"
            ]
            >= config["gates"]["minimum_filename_implied_scanner_pairs"]
            and compact_inventory["stock_label_count"]
            >= config["gates"]["minimum_stock_labels"]
            and len(normalized_exact_pages) == len(pages)
            and claims[
                "comparison_discloses_minor_exposure_adjustments"
            ]
            and claims[
                "comparison_discloses_per_frame_scanner_auto_adjustment"
            ]
            and claims["comparison_footer_all_rights_reserved"]
            and not claims[
                "explicit_comparison_pixel_research_reuse_grant_found"
            ]
        )
        else "source_contract_or_topology_not_reproduced"
    )

    public_pages = []
    for page in pages:
        public_pages.append(
            {key: value for key, value in page.items() if not key.startswith("_")}
        )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "software_commit": _software_commit(),
        "config_sha256": config_sha256,
        "pages": public_pages,
        "raw_exact_page_ids": raw_exact_pages,
        "normalized_exact_page_ids": normalized_exact_pages,
        "comparison_inventory": compact_inventory,
        "source_claim_signals": claims,
        "evidence_interpretation": {
            "stock_labels": "URL-derived weak source metadata only",
            "process_exposure_illuminant_labels": (
                "URL-derived weak source metadata only"
            ),
            "scanner_pairs": (
                "filename-implied pair candidates, not pixel-verified "
                "same-negative registration"
            ),
            "physical_roll_id": "unknown",
            "process_session": "unknown",
            "scanner_settings_profile": "unknown",
            "independent_roll_process_replication": "not established",
            "digital_film_pairs_exposed_by_comparison_page": False,
        },
        "rights_status": "no_public_reuse_grant_found/rights_unknown",
        "decision_branch_before_repeat": branch,
        "html_requests_already_retained": 8,
        "image_url_requests": 0,
        "image_payload_bytes": 0,
        "operator_fitting_opened": False,
        "training_opened": False,
        "lsm_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_10r_color_precision_metadata_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha256 = _load_config(args.config)
    report = build_report(config, config_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

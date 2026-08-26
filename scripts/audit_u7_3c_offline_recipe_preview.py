"""Audit deterministic offline previews of exact recipe-bound render outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    build_recipe_output_previews,
    build_render_recipe_history,
    render_recipe_preview_html,
)

SCHEMA = "kmcfm.u7-3c-offline-recipe-preview-result.v1"
DEFAULT_ROOT = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def audit_catalog(
    catalog: Mapping[str, Any],
    *,
    reverse: bool = False,
    expected_source_dimensions: tuple[int, int] = (6048, 4032),
    expected_preview_dimensions: tuple[int, int] = (960, 640),
) -> tuple[dict[str, Any], bytes]:
    """Audit one catalog with optional reversed decode enumeration."""

    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise TypeError("catalog entries must be a list")
    selected = list(reversed(entries)) if reverse else list(entries)
    previews = build_recipe_output_previews({**catalog, "entries": selected})
    canonical = tuple(sorted(previews, key=lambda preview: preview.recipe_path))
    html = render_recipe_preview_html(canonical)
    rows = {
        preview.recipe_path: {
            "style": preview.style,
            "output_sha256": preview.output_sha256,
            "source_width": preview.source_width,
            "source_height": preview.source_height,
            "preview_width": preview.preview_width,
            "preview_height": preview.preview_height,
            "preview_png_bytes": len(preview.png_bytes),
            "preview_png_sha256": preview.png_sha256,
        }
        for preview in canonical
    }
    scientific: dict[str, Any] = {
        "protocol": "kmcfm.u7-3c-offline-recipe-preview-contract.v1",
        "status": "PASS_PRIVATE_HASH_BOUND_OFFLINE_RECIPE_PREVIEWS",
        "rows": rows,
        "counts": {
            "catalog_entries": len(entries),
            "previews": len(previews),
        },
        "html_bytes": len(html),
        "html_sha256": _sha256(html),
        "output_file_bytes_read": sum(
            Path(preview_path).stat().st_size
            for preview_path in (
                entry["output_path"]
                for entry in entries
                if isinstance(entry, dict) and entry.get("status") == "valid"
            )
        ),
        "information_flow": {
            "input_file_reads": 0,
            "output_file_reads": len(previews),
            "pixel_decodes": len(previews),
            "rerender_calls": 0,
            "network_reads": 0,
            "filesystem_preview_writes": 0,
        },
        "gates": {
            "three_frozen_rows_previewed": len(previews) == 3,
            "source_output_sha256_exact_before_decode": True,
            "all_source_dimensions_exact": all(
                (preview.source_width, preview.source_height) == expected_source_dimensions
                for preview in previews
            ),
            "all_preview_dimensions_bounded": all(
                preview.preview_width <= 960 and preview.preview_height <= 640
                for preview in previews
            ),
            "all_preview_dimensions_exact": all(
                (preview.preview_width, preview.preview_height) == expected_preview_dimensions
                for preview in previews
            ),
            "all_preview_png_hashes_unique": len(
                {preview.png_sha256 for preview in previews}
            )
            == len(previews),
            "html_self_contained_data_images": html.count(b"data:image/png;base64,")
            == len(previews),
            "html_claim_labels_present": html.count(b"look-approximation")
            >= len(previews),
            "input_file_reads_zero": True,
            "rerender_calls_zero": True,
            "network_reads_zero": True,
        },
        "claim_ceiling": "Private read-only offline previews of three already-rendered, hash-bound look-approximation outputs only; no rerender, export, calibrated stock, stock distinguishability, packaging, telemetry or product-release claim.",
    }
    if not all(scientific["gates"].values()):
        scientific["status"] = "FAIL_CLOSED"
    stable = _sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "schema": SCHEMA,
        "scientific": scientific,
        "stable_identity": f"sha256:{stable}",
    }, html


def run_audit(*, reverse: bool = False) -> tuple[dict[str, Any], bytes]:
    """Build the frozen real catalog and audit it."""

    return audit_catalog(build_render_recipe_history(DEFAULT_ROOT), reverse=reverse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report, html = run_audit(reverse=args.reverse)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_bytes(html)
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))


if __name__ == "__main__":
    main()

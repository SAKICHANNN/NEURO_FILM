#!/usr/bin/env python
"""Audit stock/exposure/scanner connectivity from the two chart PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.color_precision_chart_source import (
    canonical_sha256,
    parse_pdfimages_list,
    parse_pdf_text_pages,
    sha256_file,
    summarize_connectivity,
)


def _pdf_text(tool: str, path: Path) -> tuple[str, str]:
    completed = subprocess.run(
        [tool, "-layout", str(path), "-"],
        check=True,
        capture_output=True,
    )
    text = completed.stdout.decode("utf-8", "replace")
    return text, hashlib.sha256(completed.stdout).hexdigest()


def _pdfimages_list(tool: str, path: Path) -> tuple[str, str]:
    completed = subprocess.run(
        [tool, "-f", "1", "-l", "12", "-list", str(path)],
        check=True,
        capture_output=True,
    )
    text = completed.stdout.decode("utf-8", "replace")
    return text, hashlib.sha256(completed.stdout).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bd0_color_precision_chart_source_v1.json",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=Path(
            "D:/neuro_film_external/"
            "u5_r2bd0_color_precision_chart_source_v1/pdfs"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2bd0_color_precision_chart_source_v1/"
            "connectivity_report.json"
        ),
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--pdfimages", default="pdfimages")
    args = parser.parse_args()
    text_tool = shutil.which(args.pdftotext)
    image_tool = shutil.which(args.pdfimages)
    if text_tool is None:
        raise RuntimeError("pdftotext is unavailable")
    if image_tool is None:
        raise RuntimeError("pdfimages is unavailable")
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = (
        ("frontier", args.pdf_root / "Charts Frontier Scans.pdf"),
        ("noritsu", args.pdf_root / "Charts Norritsu Scans.pdf"),
    )
    rows = []
    object_rows = []
    source_bindings = []
    for scanner_id, path in bindings:
        text, text_sha256 = _pdf_text(text_tool, path)
        object_text, object_text_sha256 = _pdfimages_list(image_tool, path)
        rows.extend(parse_pdf_text_pages(text, scanner_id=scanner_id))
        object_rows.extend(
            parse_pdfimages_list(object_text, scanner_id=scanner_id)
        )
        source_bindings.append(
            {
                "scanner_id": scanner_id,
                "pdf_name": path.name,
                "pdf_bytes": path.stat().st_size,
                "pdf_sha256": sha256_file(path),
                "pdftotext_sha256": text_sha256,
                "pdfimages_list_sha256": object_text_sha256,
            }
        )
    report = summarize_connectivity(rows, config)
    advertised_by_key = {
        (str(row["scanner_id"]), int(row["page_index"])): len(
            row["exposure_offsets_ev"]
        )
        for row in rows
    }
    if set(advertised_by_key) != {
        (str(row["scanner_id"]), int(row["page_index"]))
        for row in object_rows
    }:
        raise RuntimeError("PDF text/object page identity mismatch")
    matching_pages = sum(
        int(row["photo_placement_count"])
        == advertised_by_key[(str(row["scanner_id"]), int(row["page_index"]))]
        and int(row["unique_photo_object_count"])
        == int(row["photo_placement_count"])
        for row in object_rows
    )
    layout_diagnostics = {
        "advertised_exposure_label_count": sum(advertised_by_key.values()),
        "embedded_photo_placement_count": sum(
            int(row["photo_placement_count"]) for row in object_rows
        ),
        "unique_embedded_photo_object_count": sum(
            int(row["unique_photo_object_count"]) for row in object_rows
        ),
        "page_count": len(object_rows),
        "pages_with_exact_label_object_count": matching_pages,
        "pages_with_duplicate_object_placements": sum(
            int(row["duplicate_photo_placement_count"]) > 0
            for row in object_rows
        ),
        "exposure_assignment_allowed": matching_pages == len(object_rows),
        "per_page": object_rows,
    }
    report["source_bindings"] = source_bindings
    report["layout_diagnostics"] = layout_diagnostics
    report["exposure_resolved_pixel_analysis_allowed"] = False
    report["pixel_analysis_allowed"] = (
        "internal_page_level_stock_scanner_nuisance_audit_only"
    )
    report["evidence_bundle_id"] = canonical_sha256(
        {
            "stable_evidence_id": report["stable_evidence_id"],
            "source_bindings": source_bindings,
            "layout_diagnostics": layout_diagnostics,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "metrics": report["metrics"],
                "evidence_bundle_id": report["evidence_bundle_id"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

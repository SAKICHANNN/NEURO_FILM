#!/usr/bin/env python
"""Extract and audit Color Precision pages as exposure-unassigned image bags."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.color_precision_chart_source import parse_pdf_text_pages
from src.eval.color_precision_page_identifiability import (
    aggregate_page_bags,
    bind_extracted_images,
    evaluate_page_identifiability,
    extraction_inventory_id,
    parse_pdfimages_objects,
    validate_contract,
)


def _run_text(tool: str, path: Path) -> str:
    completed = subprocess.run(
        [tool, "-layout", str(path), "-"],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", "replace")


def _run_list(tool: str, path: Path) -> str:
    completed = subprocess.run(
        [tool, "-f", "1", "-l", "12", "-list", str(path)],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", "replace")


def _extract(
    tool: str, path: Path, prefix: Path, *, reuse_existing: bool
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and any(prefix.parent.glob(f"{prefix.name}-*.png")):
        return
    subprocess.run(
        [tool, "-f", "1", "-l", "12", "-png", str(path), str(prefix)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2bd1_color_precision_page_identifiability_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2bd1_color_precision_page_identifiability_v1/"
            "report.json"
        ),
    )
    parser.add_argument("--pdftotext", default="pdftotext")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(config, root=ROOT)
    extractor = config["extractor"]
    tool = str(extractor["executable"])
    destination_root = Path(str(extractor["destination_root"]))
    rows = []
    labels = []
    extracted_files: set[Path] = set()
    for pdf in config["source"]["pdfs"]:
        scanner_id = str(pdf["scanner_id"])
        path = Path(str(pdf["path"]))
        prefix = destination_root / scanner_id / scanner_id
        object_rows = parse_pdfimages_objects(
            _run_list(tool, path),
            scanner_id=scanner_id,
            minimum_width=int(extractor["minimum_width"]),
            minimum_height=int(extractor["minimum_height"]),
        )
        _extract(
            tool,
            path,
            prefix,
            reuse_existing=bool(args.reuse_existing),
        )
        bound = bind_extracted_images(object_rows, prefix=prefix)
        rows.extend(bound)
        labels.extend(
            parse_pdf_text_pages(
                _run_text(args.pdftotext, path), scanner_id=scanner_id
            )
        )
        extracted_files.update(prefix.parent.glob(f"{prefix.name}-*.png"))
    total_bytes = sum(path.stat().st_size for path in extracted_files)
    if (
        len(extracted_files) > int(extractor["maximum_extracted_files"])
        or total_bytes > int(extractor["maximum_extracted_bytes"])
    ):
        raise RuntimeError("extraction exceeds the frozen resource bounds")
    resize = tuple(int(value) for value in config["features"]["image_resize"])
    if len(resize) != 2:
        raise RuntimeError("invalid descriptor resize")
    page_rows = aggregate_page_bags(rows, labels, resize=resize)
    report = evaluate_page_identifiability(page_rows, config)
    report["extraction"] = {
        "file_count": len(extracted_files),
        "selected_unique_photo_count": len(rows),
        "bytes": total_bytes,
        "inventory_id": extraction_inventory_id(rows),
        "embedded_profile_semantics": extractor["embedded_profile_semantics"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    output_path.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gates": report["gates"],
                "metrics": report["metrics"],
                "stable_evidence_id": report["stable_evidence_id"],
                "report_sha256": __import__("hashlib")
                .sha256(payload.encode("utf-8"))
                .hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

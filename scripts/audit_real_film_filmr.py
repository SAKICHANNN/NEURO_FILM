"""Audit FILM-R decode, pairing, duplication, metadata, and visual integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.filmr import canonical_json_bytes, hash_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "real_film_filmr_audit.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "filmr_v2" / "evidence" / "pairs.jsonl",
    )
    parser.add_argument(
        "--acquisition-report",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "filmr_v2" / "evidence" / "report.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "real_film" / "filmr_v2" / "files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "filmr_v2" / "integrity_audit",
    )
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _dhash(image: Image.Image, width: int = 16, height: int = 16) -> int:
    gray = image.convert("L").resize((width + 1, height), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for value in bits.ravel():
        result = (result << 1) | int(value)
    return result


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _make_sheet(rows: list[dict[str, Any]], path: Path, tile_size: int) -> None:
    header = 24
    sheet = Image.new("RGB", (tile_size * 2, header + tile_size * len(rows)), "#151515")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), "authentic scan", fill="white")
    draw.text((tile_size + 5, 5), "expert artifact restoration", fill="white")
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("original", "restored")):
            with Image.open(row[key]) as source:
                tile = source.convert("RGB")
                tile.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (tile_size, tile_size), "black")
                canvas.paste(tile, ((tile_size - tile.width) // 2, (tile_size - tile.height) // 2))
            sheet.paste(canvas, (column * tile_size, header + row_index * tile_size))
        draw.text((3, header + row_index * tile_size + 3), row["pair_id"], fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=92, subsampling=0)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if hash_file(args.acquisition_report, "sha256") != config["acquisition_report_sha256"]:
        raise ValueError("FILM-R acquisition report hash mismatch")
    if hash_file(args.manifest, "sha256") != config["pairs_manifest_sha256"]:
        raise ValueError("FILM-R pair manifest hash mismatch")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(config["expected_pairs"]):
        raise ValueError("FILM-R audit pair count mismatch")
    data_dir = args.data_dir.resolve()
    audited: list[dict[str, Any]] = []
    exact_sha_to_pairs: dict[str, list[str]] = {}
    dhashes: dict[str, int] = {}
    family_counts: Counter[str] = Counter()
    for row in rows:
        pair_id = str(row["pair_id"])
        family_counts[str(row["filename_family_claim"])] += 1
        paths = {
            "original": data_dir / row["original_scan"]["filename"],
            "restored": data_dir / row["expert_restoration"]["filename"],
        }
        decoded: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
        for role, path in paths.items():
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                arrays[role] = np.asarray(rgb, dtype=np.int16)
                decoded[role] = {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "icc_profile_bytes": len(image.info.get("icc_profile", b"")),
                    "exif_bytes": len(image.info.get("exif", b"")),
                }
                dhashes[f"{pair_id}:{role}"] = _dhash(rgb)
            observed_sha = hash_file(path, "sha256")
            expected_sha = row[f"{'original_scan' if role == 'original' else 'expert_restoration'}"]["sha256"]
            if observed_sha != expected_sha:
                raise ValueError(f"FILM-R audit SHA drift: {pair_id}/{role}")
            exact_sha_to_pairs.setdefault(observed_sha, []).append(f"{pair_id}:{role}")
        if arrays["original"].shape != arrays["restored"].shape:
            raise ValueError(f"FILM-R paired dimensions differ: {pair_id}")
        delta = np.abs(arrays["original"] - arrays["restored"])
        audited.append(
            {
                "pair_id": pair_id,
                "filename_family_claim": row["filename_family_claim"],
                "original": str(paths["original"]),
                "restored": str(paths["restored"]),
                "decoded": decoded,
                "mean_absolute_u8_change": float(np.mean(delta)),
                "changed_pixel_fraction": float(np.mean(np.any(delta > 0, axis=-1))),
                "changed_pixel_gt8_fraction": float(np.mean(np.any(delta > 8, axis=-1))),
            }
        )
    exact_duplicates = [values for values in exact_sha_to_pairs.values() if len(values) > 1]
    if len(exact_duplicates) > int(config["exact_cross_pair_duplicate_maximum"]):
        raise ValueError(f"FILM-R exact duplicate gate failed: {exact_duplicates}")
    keys = sorted(dhashes)
    near_duplicates = []
    threshold = int(config["perceptual_hamming_threshold"])
    for left_index, left in enumerate(keys):
        left_pair = left.rsplit(":", 1)[0]
        for right in keys[left_index + 1 :]:
            right_pair = right.rsplit(":", 1)[0]
            if left_pair == right_pair:
                continue
            distance = _hamming(dhashes[left], dhashes[right])
            if distance <= threshold:
                near_duplicates.append({"left": left, "right": right, "hamming": distance})
    output_dir = args.output_dir.resolve()
    visual_rows = [
        {"pair_id": row["pair_id"], "original": row["original"], "restored": row["restored"]}
        for row in audited
    ]
    pages: list[Path] = []
    rows_per_sheet = int(config["rows_per_sheet"])
    for page_index, start in enumerate(range(0, len(visual_rows), rows_per_sheet), 1):
        page = output_dir / "sheets" / f"filmr_pairs_page_{page_index:02d}.jpg"
        _make_sheet(visual_rows[start : start + rows_per_sheet], page, int(config["tile_size"]))
        pages.append(page)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": hash_file(args.config, "sha256"),
        "acquisition_report_sha256": hash_file(args.acquisition_report, "sha256"),
        "pairs_manifest_sha256": hash_file(args.manifest, "sha256"),
        "software_commit": _commit(),
        "pairs": len(audited),
        "decoded_files": len(audited) * 2,
        "family_counts_from_filenames": dict(sorted(family_counts.items())),
        "exact_cross_pair_duplicates": exact_duplicates,
        "perceptual_cross_pair_near_duplicates": near_duplicates,
        "icc_profile_file_count": sum(
            row["decoded"][role]["icc_profile_bytes"] > 0 for row in audited for role in ("original", "restored")
        ),
        "exif_file_count": sum(
            row["decoded"][role]["exif_bytes"] > 0 for row in audited for role in ("original", "restored")
        ),
        "dimensions": sorted(
            {f"{row['decoded'][role]['width']}x{row['decoded'][role]['height']}" for row in audited for role in ("original", "restored")}
        ),
        "per_pair": audited,
        "review_sheets": [str(path) for path in pages],
        "review_sheet_sha256": {str(path): hash_file(path, "sha256") for path in pages},
        "visual_decision": "pending",
        "group_policy": config["group_policy"],
    }
    report_path = output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "report": str(report_path),
        "pairs": len(audited),
        "near_duplicates": len(near_duplicates),
        "sheets": len(pages),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Decode/integrity and support audits for frozen BlueNeg stock pilots."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.real_film.fsa_owi_pilot import dhash64, hamming64
from src.roll2film.blueneg_download import sha256_file


class StockPilotIntegrityError(ValueError):
    """Raised when the frozen stock-pilot integrity contract fails closed."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _content_cell(scene_property: Mapping[str, Any] | None) -> str:
    if not isinstance(scene_property, Mapping):
        return "day=unknown|indoor=unknown"
    daytime = str(scene_property.get("is_daytime") or "unknown")
    indoor = str(scene_property.get("is_indoor") or "unknown")
    return f"day={daytime}|indoor={indoor}"


def _near_duplicate_pairs(
    records: Sequence[Mapping[str, Any]], *, threshold: int
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            distance = hamming64(str(left["dhash64"]), str(right["dhash64"]))
            if distance <= threshold:
                pairs.append(
                    {
                        "left_path": left["path"],
                        "right_path": right["path"],
                        "left_frame_id": left["frame_id"],
                        "right_frame_id": right["frame_id"],
                        "left_film_stock_id": left["film_stock_id"],
                        "right_film_stock_id": right["film_stock_id"],
                        "left_lane": left["lane"],
                        "right_lane": right["lane"],
                        "hamming": distance,
                        "same_frame": left["frame_id"] == right["frame_id"],
                        "same_roll": left["roll_id"] == right["roll_id"],
                        "same_stock": left["film_stock_id"] == right["film_stock_id"],
                    }
                )
    pairs.sort(key=lambda row: (row["hamming"], row["left_path"], row["right_path"]))
    return pairs


def audit_stock_pilot_integrity(
    *,
    download_root: Path,
    acquisition: Mapping[str, Any],
    download_report: Mapping[str, Any],
    metadata_report: Mapping[str, Any],
    frame_rows: Sequence[Mapping[str, Any]],
    roll_rows: Sequence[Mapping[str, Any]],
    software_commit: str,
    dhash_threshold: int = 4,
) -> dict[str, Any]:
    """Decode and audit the exact frozen stock-pilot payloads without colour fitting."""
    files = acquisition.get("files")
    if not isinstance(files, list) or not files:
        raise StockPilotIntegrityError("acquisition files missing")
    if int(acquisition.get("file_count", -1)) != len(files):
        raise StockPilotIntegrityError("acquisition file_count mismatch")
    if int(download_report.get("files", -1)) != len(files):
        raise StockPilotIntegrityError("download report file count mismatch")
    if int(download_report.get("bytes", -1)) != int(acquisition.get("bytes", -2)):
        raise StockPilotIntegrityError("download/acquisition byte mismatch")
    if not download_report.get("all_sizes_and_lfs_sha256_verified"):
        raise StockPilotIntegrityError("download report is not fully verified")
    if int(download_report.get("manifest_external_lane_files", -1)) != 0:
        raise StockPilotIntegrityError("download report reports external lane files")

    if sum(int(row.get("size", -1)) for row in files) != int(
        acquisition.get("bytes", -2)
    ):
        raise StockPilotIntegrityError("acquisition byte total mismatch")

    sealed_rolls = {
        str(row["roll_id"])
        for row in roll_rows
        if bool(row.get("contains_official_test_frame"))
    }
    frames_by_id = {str(row["filename"]): row for row in frame_rows}
    if len(frames_by_id) != len(frame_rows):
        raise StockPilotIntegrityError("duplicate BlueNeg frame metadata ids")
    rolls_by_id = {str(row["roll_id"]): row for row in roll_rows}
    if len(rolls_by_id) != len(roll_rows):
        raise StockPilotIntegrityError("duplicate BlueNeg roll metadata ids")
    inventory = {str(row["path"]): row for row in files}
    if len(inventory) != len(files):
        raise StockPilotIntegrityError("duplicate acquisition paths")

    decoded: list[dict[str, Any]] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    decode_failures: list[dict[str, Any]] = []

    for row in sorted(files, key=lambda item: str(item["path"])):
        remote_path = str(row["path"])
        target = (download_root / Path(*Path(remote_path).parts)).resolve()
        try:
            target.relative_to(download_root.resolve())
        except ValueError as exc:
            raise StockPilotIntegrityError(f"path escapes download root: {remote_path}") from exc
        if str(row["roll_id"]) in sealed_rolls:
            raise StockPilotIntegrityError(f"official-test roll leaked into acquisition: {row['roll_id']}")
        if not target.is_file():
            raise StockPilotIntegrityError(f"missing downloaded file: {remote_path}")
        size = target.stat().st_size
        digest = sha256_file(target)
        if size != int(row["size"]) or digest != str(row["sha256"]):
            raise StockPilotIntegrityError(f"size/hash mismatch before decode: {remote_path}")
        frame_id = str(row["frame_id"])
        meta = frames_by_id.get(frame_id)
        if meta is None:
            raise StockPilotIntegrityError(f"missing BlueNeg frame metadata for {frame_id}")
        roll_id = str(row["roll_id"])
        source_label = str(row["source_label"])
        stock_id = str(row["film_stock_id"])
        lane = str(row["lane"])
        summary = metadata_report.get("stock_summaries", {}).get(stock_id)
        if not isinstance(summary, Mapping):
            raise StockPilotIntegrityError(
                f"missing stock summary for manifest stock id: {stock_id}"
            )
        if str(summary.get("source_label")) != source_label:
            raise StockPilotIntegrityError(
                f"stock/source label mismatch for {frame_id}: {stock_id} / {source_label}"
            )
        if str(meta.get("film_type")) != source_label:
            raise StockPilotIntegrityError(
                f"frame/source label mismatch for {frame_id}: {source_label}"
            )
        if str(meta.get("roll_id")) != roll_id or roll_id not in rolls_by_id:
            raise StockPilotIntegrityError(
                f"frame/manifest roll mismatch for {frame_id}: {roll_id}"
            )
        if str(rolls_by_id[roll_id].get("film_type")) != source_label:
            raise StockPilotIntegrityError(
                f"roll/source label mismatch for {roll_id}: {source_label}"
            )
        expected_path = {
            "negative_preview": meta.get("preview_path"),
            "display_proxy": meta.get("pseudogt_path"),
        }.get(lane)
        if expected_path is None or str(expected_path) != remote_path:
            raise StockPilotIntegrityError(
                f"lane/path mismatch for {frame_id}: {lane} / {remote_path}"
            )
        if lane == "display_proxy" and not bool(meta.get("alignment_available")):
            raise StockPilotIntegrityError(
                f"display proxy lacks alignment metadata for {frame_id}"
            )
        try:
            with Image.open(target) as image:
                image.verify()
            with Image.open(target) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                fmt = image.format
                icc = image.info.get("icc_profile") or b""
                if not isinstance(icc, (bytes, bytearray)):
                    icc = b""
                hash_image = image.convert("RGB")
                digest_hash = dhash64(hash_image)
        except Exception as exc:  # noqa: BLE001 - fail-closed inventory
            decode_failures.append(
                {"path": remote_path, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        record = {
            "path": remote_path,
            "film_stock_id": stock_id,
            "source_label": source_label,
            "roll_id": roll_id,
            "frame_id": frame_id,
            "lane": lane,
            "size": size,
            "sha256": digest,
            "width": int(width),
            "height": int(height),
            "mode": str(mode),
            "format": str(fmt),
            "icc_profile_bytes": len(icc),
            "icc_profile_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
            "dhash64": digest_hash,
            "location": str(meta.get("location") or "unknown"),
            "content_cell": _content_cell(
                meta.get("scene_property") if isinstance(meta.get("scene_property"), Mapping) else None
            ),
            "partition": str(meta.get("partition") or "unknown"),
            "alignment_available": bool(meta.get("alignment_available")),
        }
        decoded.append(record)
        sha_groups[digest].append(remote_path)

    if decode_failures:
        raise StockPilotIntegrityError(
            f"decode failures: {decode_failures[:3]!r} ({len(decode_failures)} total)"
        )
    if len(decoded) != len(files):
        raise StockPilotIntegrityError("decoded file count mismatch")

    exact_duplicates = [
        {"sha256": digest, "paths": paths, "count": len(paths)}
        for digest, paths in sorted(sha_groups.items())
        if len(paths) > 1
    ]
    near_pairs = _near_duplicate_pairs(decoded, threshold=dhash_threshold)
    cross_frame_near = [row for row in near_pairs if not row["same_frame"]]
    same_frame_lane_pairs = [row for row in near_pairs if row["same_frame"]]

    stock_ids = sorted({str(row["film_stock_id"]) for row in decoded})
    support: dict[str, Any] = {}
    eligibility: dict[str, Any] = {}
    for stock_id in stock_ids:
        rows = [row for row in decoded if row["film_stock_id"] == stock_id]
        previews = [row for row in rows if row["lane"] == "negative_preview"]
        proxies = [row for row in rows if row["lane"] == "display_proxy"]
        rolls = sorted({row["roll_id"] for row in previews})
        content_counts = Counter(row["content_cell"] for row in previews)
        location_counts = Counter(row["location"] for row in previews)
        summary = metadata_report.get("stock_summaries", {}).get(stock_id, {})
        preview_descriptor_ok = len(rolls) >= 3 and len(previews) >= 3
        display_ok = bool(summary.get("display_operator_candidate")) and len(proxies) > 0
        # Structural content support: at least two content cells with >=2 preview frames.
        supported_cells = sorted(
            cell for cell, count in content_counts.items() if count >= 2
        )
        structural_content_ok = len(supported_cells) >= 2
        decision = "negative_preview_identifiability_candidate"
        if not preview_descriptor_ok:
            decision = "downgrade_insufficient_rolls_or_frames"
        elif display_ok:
            decision = "display_operator_research_candidate"
        support[stock_id] = {
            "source_label": rows[0]["source_label"],
            "decoded_files": len(rows),
            "preview_frames": len(previews),
            "display_proxy_frames": len(proxies),
            "eligible_rolls": rolls,
            "eligible_roll_count": len(rolls),
            "content_cell_counts": dict(sorted(content_counts.items())),
            "supported_content_cells_ge2": supported_cells,
            "location_counts": dict(sorted(location_counts.items())),
            "modes": dict(sorted(Counter(row["mode"] for row in rows).items())),
            "formats": dict(sorted(Counter(row["format"] for row in rows).items())),
            "icc_present_count": sum(1 for row in rows if row["icc_profile_bytes"] > 0),
            "metadata_display_operator_candidate": bool(
                summary.get("display_operator_candidate")
            ),
            "metadata_identifiability_candidate": bool(
                summary.get("metadata_identifiability_candidate")
            ),
        }
        eligibility[stock_id] = {
            "decision": decision,
            "negative_preview_identifiability_eligible": preview_descriptor_ok,
            "display_operator_research_eligible": display_ok,
            "structural_content_diversity": structural_content_ok,
            "notes": (
                "display proxy lane present; still requires RF1.4 stock/content/nuisance gates"
                if decision == "display_operator_research_candidate"
                else (
                    "post-negation 8-bit preview descriptors only; not a physical-density or display-operator lane"
                    if preview_descriptor_ok
                    else "insufficient independent rolls/frames after sealing"
                )
            ),
        }

    mode_counts = Counter(row["mode"] for row in decoded)
    format_counts = Counter(row["format"] for row in decoded)
    checks = {
        "expected_files": len(files),
        "decoded_files": len(decoded),
        "decode_failures": 0,
        "exact_sha_duplicate_groups": len(exact_duplicates),
        "dhash_near_pairs_le_threshold": len(near_pairs),
        "cross_frame_dhash_near_pairs": len(cross_frame_near),
        "same_frame_preview_proxy_near_pairs": len(same_frame_lane_pairs),
        "sealed_official_test_roll_overlap": 0,
        "colour_fit_performed": False,
        "border_mask_implemented": False,
    }
    passed = (
        checks["decoded_files"] == checks["expected_files"]
        and checks["decode_failures"] == 0
        and checks["exact_sha_duplicate_groups"] == 0
        and checks["sealed_official_test_roll_overlap"] == 0
        and not checks["colour_fit_performed"]
    )
    report = {
        "schema_version": 2,
        "audit_id": "stock-first-blueneg-four-pilot-integrity-v2",
        "software_commit": software_commit,
        "download_root": str(download_root.as_posix()),
        "acquisition_manifest_id": acquisition.get("manifest_id"),
        "acquisition_file_count": len(files),
        "acquisition_bytes": int(acquisition.get("bytes", 0)),
        "dhash_threshold": dhash_threshold,
        "checks": checks,
        "passed": passed,
        "mode_counts": dict(sorted(mode_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "exact_sha_duplicates": exact_duplicates,
        "dhash_near_pairs": near_pairs,
        "support_by_stock": support,
        "eligibility_by_stock": eligibility,
        "border_mask_status": "not_implemented",
        "visual_adjudication_status": "pending Codex vision adjudication",
        "claim_ceiling": (
            "integrity/support audit only; negative-preview-8bit is a post-negation "
            "preview, not verified physical density; no stock signal, S2 transfer, "
            "calibration, authenticity or release claim"
        ),
        "file_records": decoded,
    }
    return report


def write_stock_pilot_integrity_report(path: Path, report: Mapping[str, Any]) -> str:
    return _atomic_json(path, report)


def render_stock_pilot_contact_sheets(
    *,
    download_root: Path,
    file_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    per_sheet: int = 24,
) -> list[dict[str, Any]]:
    """Render review-only contact sheets; does not modify downloaded evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    cell_width, cell_height = 280, 230
    columns = 4
    by_stock: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in file_records:
        if row["lane"] == "negative_preview":
            by_stock[str(row["film_stock_id"])].append(row)
    for stock_id, rows in sorted(by_stock.items()):
        ordered = sorted(rows, key=lambda item: (str(item["roll_id"]), str(item["frame_id"])))
        for sheet_index, start in enumerate(range(0, len(ordered), per_sheet)):
            chunk = ordered[start : start + per_sheet]
            canvas = Image.new(
                "RGB",
                (
                    columns * cell_width,
                    ((len(chunk) + columns - 1) // columns) * cell_height,
                ),
                "#202020",
            )
            draw = ImageDraw.Draw(canvas)
            for offset, row in enumerate(chunk):
                x = (offset % columns) * cell_width
                y = (offset // columns) * cell_height
                local = download_root / Path(*Path(str(row["path"])).parts)
                with Image.open(local) as image:
                    preview = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (cell_width - 12, 185),
                    )
                canvas.paste(preview, (x + (cell_width - preview.width) // 2, y + 4))
                label = f"{row['roll_id']} {row['frame_id']}"
                draw.text((x + 6, y + 193), label[:44], fill="white")
            relative = f"contact_{stock_id}_{sheet_index + 1:02d}.jpg"
            path = output_dir / relative
            canvas.save(path, quality=92, subsampling=0)
            sheets.append(
                {
                    "film_stock_id": stock_id,
                    "path": str(path.as_posix()),
                    "frames": len(chunk),
                    "sha256": sha256_file(path),
                    "visual_adjudication_status": "pending Codex vision adjudication",
                }
            )
    return sheets

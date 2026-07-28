"""Independent RAW source acquisition and integrity preflight."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image, ImageDraw, ImageFont, ImageOps


class ConfirmationSourceError(RuntimeError):
    """Raised when the frozen source contract cannot be executed safely."""


REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "source_id",
        "source_url",
        "author",
        "license",
        "license_snapshot_date",
        "rights_scope",
        "scene_group",
        "roll_group",
        "lab_group",
        "scanner_group",
        "uploader_group",
        "decoded_sha256",
        "dhash64",
        "derivation_lineage",
        "allowed_use",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_contract(root: Path, config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfirmationSourceError("unsupported schema_version")
    selection = config["selection"]
    rows = config["candidates"]
    if len(rows) != int(selection["candidate_count"]):
        raise ConfirmationSourceError("candidate_count mismatch")
    ids = [str(row["id"]) for row in rows]
    hashes = [str(row["sha256"]) for row in rows]
    if len(set(ids)) != len(rows) or len(set(hashes)) != len(rows):
        raise ConfirmationSourceError("candidate IDs and hashes must be unique")
    if any(len(value) != 64 for value in hashes):
        raise ConfirmationSourceError("invalid candidate SHA-256")
    if any(
        not str(row["url"]).startswith("https://raw.pixls.us/getfile.php/")
        for row in rows
    ):
        raise ConfirmationSourceError("candidate URL leaves frozen host/path")
    makes = Counter(str(row["make"]) for row in rows)
    if len(makes) != int(selection["expected_makes"]):
        raise ConfirmationSourceError("expected_makes mismatch")
    if max(makes.values()) > int(selection["maximum_rows_per_make"]):
        raise ConfirmationSourceError("maximum_rows_per_make exceeded")

    parent = config["parent_decision"]
    parent_path = root / parent["path"]
    if not parent_path.is_file() or sha256_file(parent_path) != parent["sha256"]:
        raise ConfirmationSourceError("parent_decision hash mismatch")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    required_decision = parent.get("required_decision")
    if (
        required_decision is not None
        and parent_payload.get("decision") != required_decision
    ):
        raise ConfirmationSourceError("parent_decision state mismatch")
    development = root / config["preflight"]["development_manifest"]
    if (
        not development.is_file()
        or sha256_file(development)
        != config["preflight"]["development_manifest_sha256"]
    ):
        raise ConfirmationSourceError("development manifest hash mismatch")
    for comparison in config["preflight"].get("comparison_manifests", []):
        path = root / comparison["path"]
        if (
            comparison.get("format")
            not in {"frozen_set", "rawpixls_manifest"}
            or not path.is_file()
            or sha256_file(path) != comparison["sha256"]
        ):
            raise ConfirmationSourceError("comparison manifest drift")
    if config.get("training_allowed") or config.get("operator_fitting_allowed"):
        raise ConfirmationSourceError("preflight must forbid learning/fitting")


def _download_exact(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[int, str]:
    if destination.is_file():
        digest = sha256_file(destination)
        if digest == expected_sha256:
            return destination.stat().st_size, digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists() and temporary.stat().st_size > maximum_bytes:
        raise ConfirmationSourceError("partial download exceeds byte cap")

    start = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "neuro-film-u5-r2ai1s/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        append = start > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        total = start if append else 0
        with temporary.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ConfirmationSourceError("download exceeds byte cap")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

    digest = sha256_file(temporary)
    if digest != expected_sha256:
        raise ConfirmationSourceError(
            f"download SHA mismatch: expected {expected_sha256}, got {digest}"
        )
    temporary.replace(destination)
    return destination.stat().st_size, digest


def _render_raw(raw_path: Path, output_path: Path, max_side: int) -> Image.Image:
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            gamma=(2.222, 4.5),
            user_flip=0,
        )
    image = ImageOps.exif_transpose(Image.fromarray(rgb, mode="RGB"))
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG", compress_level=6)
    return image


def dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    packed = 0
    for bit in bits.flat:
        packed = (packed << 1) | int(bit)
    return f"{packed:016x}"


def hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def image_diagnostics(image: Image.Image) -> dict[str, Any]:
    small = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.float32) / 255.0
    mean = float(values.mean())
    channel_delta = float(
        np.mean(
            np.abs(values[..., 0] - values[..., 1])
            + np.abs(values[..., 1] - values[..., 2])
        )
    )
    saturation_proxy = float(
        np.std(values[..., 0] - values[..., 1])
        + np.std(values[..., 1] - values[..., 2])
    )
    near_empty_or_monochrome = bool(
        mean < 0.055
        or mean > 0.96
        or channel_delta <= 0.012
        or saturation_proxy <= 0.018
    )
    return {
        "mean_rgb": mean,
        "channel_delta": channel_delta,
        "saturation_proxy": saturation_proxy,
        "near_empty_or_monochrome": near_empty_or_monochrome,
    }


def validate_manifest_row(row: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - row.keys())
    if missing:
        raise ConfirmationSourceError(
            f"manifest row lacks required lineage fields: {missing}"
        )
    if any(row[key] in (None, "") for key in REQUIRED_MANIFEST_FIELDS):
        raise ConfirmationSourceError("manifest lineage fields must be explicit")
    if row["author"] != "unknown" or row["uploader_group"] != "unknown":
        raise ConfirmationSourceError(
            "raw.pixls rows do not expose author/uploader identity"
        )
    if row["roll_group"] != "unknown" or row["scanner_group"] != "unknown":
        raise ConfirmationSourceError(
            "digital input must not invent film roll/scanner metadata"
        )


def duplicate_pairs(
    rows: list[dict[str, Any]],
    *,
    threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["decoded_sha256"] == right["decoded_sha256"]:
                exact.append({"left": left["id"], "right": right["id"]})
            distance = hamming64(left["dhash64"], right["dhash64"])
            if distance <= threshold:
                near.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "distance": distance,
                    }
                )
    return exact, near


def cross_pool_pairs(
    current: list[dict[str, Any]],
    development: list[dict[str, Any]],
    *,
    threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for left in current:
        for right in development:
            if left["decoded_sha256"] == right["decoded_sha256"]:
                exact.append({"left": left["id"], "right": right["id"]})
            distance = hamming64(left["dhash64"], right["dhash64"])
            if distance <= threshold:
                near.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "distance": distance,
                    }
                )
    return exact, near


def _frozen_set_rows(root: Path, path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in document["frozen_set"]["samples"]:
        path = root / source["source_path"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            rows.append(
                {
                    "id": str(source["id"]),
                    "decoded_sha256": sha256_file(path),
                    "dhash64": dhash64(image),
                }
            )
    return rows


def _comparison_rows(
    root: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    descriptors = config["preflight"].get("comparison_manifests")
    if not descriptors:
        return _frozen_set_rows(
            root, root / config["preflight"]["development_manifest"]
        )
    rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        path = root / descriptor["path"]
        if descriptor["format"] == "frozen_set":
            rows.extend(_frozen_set_rows(root, path))
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            raise ConfirmationSourceError("rawpixls manifest must be a list")
        for source in document:
            decoded_path = root / str(source["decoded_path"])
            if sha256_file(decoded_path) != source["decoded_sha256"]:
                raise ConfirmationSourceError(
                    "comparison decoded-image hash mismatch"
                )
            rows.append(
                {
                    "id": str(source["id"]),
                    "decoded_sha256": str(source["decoded_sha256"]),
                    "dhash64": str(source["dhash64"]),
                }
            )
    identities = [
        (row["decoded_sha256"], row["dhash64"]) for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ConfirmationSourceError("comparison manifests overlap")
    return rows


def make_contact_sheet(
    root: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    experiment_id: str,
) -> None:
    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    for row in rows:
        with Image.open(root / row["decoded_path"]) as opened:
            image = opened.convert("RGB")
            image.thumbnail((420, 260), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (440, 300), "white")
        tile.paste(image, ((440 - image.width) // 2, 0))
        ImageDraw.Draw(tile).text(
            (8, 268),
            f"{row['id']} | {row['make']} {row['model']}",
            fill="black",
            font=font,
        )
        tiles.append(tile)
    columns = 3
    sheet = Image.new(
        "RGB",
        (columns * 440, ((len(tiles) + columns - 1) // columns) * 300 + 36),
        "white",
    )
    ImageDraw.Draw(sheet).text(
        (8, 10),
        f"{experiment_id} source-only RAW preflight (operator not applied)",
        fill="black",
        font=font,
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 440, 36 + (index // columns) * 300))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")


def run_preflight(
    *,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    raw_root = root / config["selection"]["download_root"]
    decoded_root = root / config["selection"]["decoded_root"]
    max_side = int(config["selection"]["maximum_decoded_side"])
    per_file_cap = max(
        int(config["maximum_download_bytes"]) // len(config["candidates"]) * 2,
        32 * 1024 * 1024,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    downloaded_bytes = 0
    for source in config["candidates"]:
        extension = Path(str(source["filename"])).suffix.lower()
        raw_path = raw_root / f"{source['id']}{extension}"
        decoded_path = decoded_root / f"{source['id']}.png"
        try:
            size, raw_hash = _download_exact(
                url=str(source["url"]),
                destination=raw_path,
                expected_sha256=str(source["sha256"]),
                maximum_bytes=per_file_cap,
            )
            downloaded_bytes += size
            if downloaded_bytes > int(config["maximum_download_bytes"]):
                raise ConfirmationSourceError("aggregate download exceeds cap")
            image = _render_raw(raw_path, decoded_path, max_side)
            diagnostics = image_diagnostics(image)
            row = {
                "manifest_schema_version": 1,
                "id": source["id"],
                "source_id": f"rawpixls:{source['id']}",
                "make": source["make"],
                "model": source["model"],
                "mode": source["mode"],
                "date": source["date"],
                "author": "unknown",
                "uploader_group": "unknown",
                "scene_group": "unknown",
                "roll_group": "unknown",
                "lab_group": "unknown",
                "scanner_group": "unknown",
                "process_type": "unknown",
                "license": config["source"]["declared_license"],
                "license_snapshot_date": config["source"]["api_observed_date"],
                "rights_scope": "CC0_public_domain_internal_evaluation",
                "source_url": source["url"],
                "raw_path": raw_path.relative_to(root).as_posix(),
                "raw_bytes": size,
                "raw_sha256": raw_hash,
                "decoded_path": decoded_path.relative_to(root).as_posix(),
                "decoded_bytes": decoded_path.stat().st_size,
                "decoded_sha256": sha256_file(decoded_path),
                "dhash64": dhash64(image),
                "width": image.width,
                "height": image.height,
                "decoded_color_state": "relative_display_srgb_approximation",
                **diagnostics,
                "derivation_lineage": {
                    "source_raw_sha256": raw_hash,
                    "decoder": "rawpy_libraw",
                    "decoder_version": rawpy.__version__,
                    "camera_white_balance": True,
                    "auto_bright_disabled": False,
                    "gamma": [2.222, 4.5],
                    "output_bits": 8,
                    "orientation": "fixed_user_flip_0_then_exif_transpose",
                    "maximum_side": max_side,
                    "resampler": "Pillow_LANCZOS",
                    "encoding": "RGB8_PNG_compress_level_6",
                },
                "allowed_use": "internal_independent_digital_ood_confirmation",
            }
            validate_manifest_row(row)
            rows.append(row)
        except Exception as exc:  # preserve all frozen-row failures
            failures.append({"id": str(source["id"]), "error": repr(exc)})

    within_exact, within_near = duplicate_pairs(rows, threshold=4)
    development = _comparison_rows(root, config)
    cross_exact, cross_near = cross_pool_pairs(
        rows, development, threshold=4
    )
    makes = Counter(row["make"] for row in rows)
    near_empty = sum(bool(row["near_empty_or_monochrome"]) for row in rows)
    largest_make_fraction = max(makes.values(), default=0) / max(len(rows), 1)
    gates = config["preflight"]
    automatic_gates = {
        "minimum_rows": len(rows) >= int(gates["minimum_hash_clean_decoded_rows"]),
        "minimum_makes": len(makes) >= int(gates["minimum_camera_makes"]),
        "largest_make_fraction": largest_make_fraction
        <= float(gates["maximum_largest_make_fraction"]),
        "within_exact": len(within_exact)
        <= int(gates["maximum_exact_within_pool_pairs"]),
        "within_dhash": len(within_near)
        <= int(gates["maximum_dhash_within_pool_pairs_le_4"]),
        "cross_exact": len(cross_exact)
        <= int(gates["maximum_exact_cross_pool_pairs"]),
        "cross_dhash": len(cross_near)
        <= int(gates["maximum_dhash_cross_pool_pairs_le_4"]),
        "near_empty_or_monochrome": near_empty
        <= int(gates["maximum_near_empty_or_monochrome_rows"]),
    }
    automatic_pass = all(automatic_gates.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "candidate_count": len(config["candidates"]),
        "decoded_row_count": len(rows),
        "failure_count": len(failures),
        "downloaded_bytes": downloaded_bytes,
        "camera_make_count": len(makes),
        "make_counts": dict(sorted(makes.items())),
        "largest_make_fraction": largest_make_fraction,
        "near_empty_or_monochrome_count": near_empty,
        "within_exact_pairs": within_exact,
        "within_dhash_pairs_le_4": within_near,
        "cross_exact_pairs": cross_exact,
        "cross_dhash_pairs_le_4": cross_near,
        "automatic_gates": automatic_gates,
        "automatic_pass": automatic_pass,
        "visual_review_required": automatic_pass
        and bool(gates["require_autonomous_visual_content_and_severe_audit"]),
        "operator_applied": False,
        "rows": rows,
        "failures": failures,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "automatic_report.json"
    manifest_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_contact_sheet(
        root,
        rows,
        output_dir / "source_contact_sheet.png",
        str(config["experiment_id"]),
    )
    return {
        "report": report,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "report_path": report_path,
        "report_sha256": sha256_file(report_path),
    }

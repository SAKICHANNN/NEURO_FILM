"""U1.4C5S rights, byte, ICC and pixel preflight for Commons ProPhoto images."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import urllib.request
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw

from scripts.build_fivek_freeze_pack import (
    D50_TO_D65_BRADFORD,
    PROPHOTO_TO_XYZ_D50,
    XYZ_D65_TO_SRGB,
    prophoto_decode,
    resize_float,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.preprocess.color_management import convert_linear_rgb

SCHEMA = "neuro-film.u1-4c5-commons-prophoto-source-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c5-commons-prophoto-source-report.v1"
EXPERIMENT_ID = "U1.4C5S"
CONTRACT_SHA256 = "0f86588a7b26206cffe7b7d34b61594dbfcd7de86a2866a6658087f5e747cf7f"
_D50 = np.array([0.9642, 1.0, 0.8249], dtype=np.float64)


class CommonsProPhotoSourceError(RuntimeError):
    """Raised when source bytes, rights, ICC structure or pixels drift."""


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise CommonsProPhotoSourceError("C5S contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CommonsProPhotoSourceError("C5S contract structure drift")
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != int(payload["eligibility"]["required_source_count"])
        or len({row["id"] for row in rows}) != len(rows)
        or len({row["url"] for row in rows}) != len(rows)
        or sum(int(row["bytes"]) for row in rows)
        > int(payload["storage"]["maximum_total_download_bytes"])
        or any(
            row["mime"] not in payload["eligibility"]["allowed_mime_types"]
            for row in rows
        )
        or any(
            row["license"] not in payload["eligibility"]["allowed_licenses"]
            for row in rows
        )
    ):
        raise CommonsProPhotoSourceError("C5S source inventory drift")
    root = Path(payload["storage"]["logical_root"])
    if root.is_absolute() or ".." in root.parts:
        raise CommonsProPhotoSourceError("C5S storage root must be repository-relative")
    return payload


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_exact(row: Mapping[str, Any], destination: Path) -> None:
    if destination.is_file():
        if (
            destination.stat().st_size == int(row["bytes"])
            and _sha1_file(destination) == row["sha1"]
        ):
            return
        raise CommonsProPhotoSourceError(f"existing source identity drift: {row['id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    if temporary.exists():
        raise CommonsProPhotoSourceError(f"owned temporary already exists: {temporary}")
    request = urllib.request.Request(
        row["url"],
        headers={"User-Agent": "neuro-film-research/1.0 (internal source audit)"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if (
            temporary.stat().st_size != int(row["bytes"])
            or _sha1_file(temporary) != row["sha1"]
        ):
            raise CommonsProPhotoSourceError(f"download identity mismatch: {row['id']}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _icc_xyz_tag(profile: bytes, signature: bytes) -> np.ndarray:
    if len(profile) < 132 or profile[16:20] != b"RGB ":
        raise CommonsProPhotoSourceError("embedded ICC is not an RGB profile")
    count = struct.unpack_from(">I", profile, 128)[0]
    if count > 256 or 132 + count * 12 > len(profile):
        raise CommonsProPhotoSourceError("embedded ICC tag table is invalid")
    for index in range(count):
        tag, offset, size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if tag != signature:
            continue
        if (
            size < 20
            or offset + size > len(profile)
            or profile[offset : offset + 4] != b"XYZ "
        ):
            raise CommonsProPhotoSourceError("embedded ICC XYZ tag is invalid")
        values = struct.unpack_from(">iii", profile, offset + 8)
        return np.asarray(values, dtype=np.float64) / 65536.0
    raise CommonsProPhotoSourceError(
        f"embedded ICC missing {signature.decode('ascii')}"
    )


def _profile_facts(profile: bytes) -> dict[str, Any]:
    expected = {
        b"rXYZ": PROPHOTO_TO_XYZ_D50[:, 0].astype(np.float64),
        b"gXYZ": PROPHOTO_TO_XYZ_D50[:, 1].astype(np.float64),
        b"bXYZ": PROPHOTO_TO_XYZ_D50[:, 2].astype(np.float64),
        b"wtpt": _D50,
    }
    values = {key: _icc_xyz_tag(profile, key) for key in expected}
    maximum_error = max(
        float(np.max(np.abs(values[key] - expected[key]))) for key in expected
    )
    try:
        cms_profile = ImageCms.ImageCmsProfile(BytesIO(profile))
        description = ImageCms.getProfileDescription(cms_profile).strip()
        name = ImageCms.getProfileName(cms_profile).strip()
    except Exception as exc:
        raise CommonsProPhotoSourceError(
            "embedded ICC cannot be opened by LittleCMS"
        ) from exc
    if maximum_error > 5e-4 or not any(
        token in (description + " " + name).lower() for token in ("prophoto", "romm")
    ):
        raise CommonsProPhotoSourceError(
            "embedded ICC is not the required ProPhoto/ROMM family"
        )
    return {
        "icc_profile_sha256": hashlib.sha256(profile).hexdigest(),
        "icc_profile_bytes": len(profile),
        "icc_profile_description": description,
        "icc_profile_name": name,
        "maximum_prophoto_xyz_absolute_error": maximum_error,
    }


def _png_bit_depth(path: Path) -> int:
    raw = path.read_bytes()[:29]
    if len(raw) < 29 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise CommonsProPhotoSourceError("invalid PNG header")
    return int(raw[24])


def _decode_rgb(path: Path, mime: str) -> tuple[np.ndarray, int]:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        data is None
        or data.ndim != 3
        or data.shape[2] != 3
        or data.dtype not in (np.uint8, np.uint16)
    ):
        raise CommonsProPhotoSourceError(
            "source is not a supported three-channel integer raster"
        )
    rgb = np.ascontiguousarray(data[..., ::-1])
    bit_depth = _png_bit_depth(path) if mime == "image/png" else 8
    if (bit_depth == 8 and rgb.dtype != np.uint8) or (
        bit_depth == 16 and rgb.dtype != np.uint16
    ):
        raise CommonsProPhotoSourceError("container and decoder bit depth disagree")
    return rgb, bit_depth


def _dhash64(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    array = np.asarray(small, dtype=np.uint8)
    value = 0
    for bit in (array[:, 1:] > array[:, :-1]).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _preview_and_dhash(path: Path, profile: bytes) -> tuple[Image.Image, str]:
    with Image.open(path) as image:
        oriented = image.convert("RGB")
        try:
            converted = ImageCms.profileToProfile(
                oriented,
                ImageCms.ImageCmsProfile(BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except Exception as exc:
            raise CommonsProPhotoSourceError(
                "ICC-to-sRGB preview conversion failed"
            ) from exc
        converted.thumbnail((320, 240), Image.Resampling.LANCZOS)
        preview = converted.copy()
    return preview, _dhash64(preview)


def _rec2020_oog_fraction(rgb: np.ndarray) -> float:
    maximum = np.float32(np.iinfo(rgb.dtype).max)
    encoded = resize_float(rgb.astype(np.float32) / maximum, 1600)
    linear = prophoto_decode(encoded)
    xyz_d50 = linear @ PROPHOTO_TO_XYZ_D50.T
    xyz_d65 = xyz_d50 @ D50_TO_D65_BRADFORD.T
    linear_srgb = xyz_d65 @ XYZ_D65_TO_SRGB.T
    rec2020 = convert_linear_rgb(
        np.asarray(linear_srgb, dtype=np.float32),
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    return float(np.mean(np.any((rec2020 < 0.0) | (rec2020 > 1.0), axis=-1)))


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    source_root = root / Path(contract["storage"]["logical_root"])
    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    unrequested_rows: list[dict[str, Any]] = []
    previews: list[Image.Image] = []
    for selection_rank, expected in enumerate(contract["rows"]):
        remaining = len(contract["rows"]) - selection_rank
        if len(rows) + remaining < int(
            contract["eligibility"]["minimum_eligible_source_count"]
        ):
            unrequested_rows.extend(
                {
                    "id": row["id"],
                    "selection_rank": index,
                    "title": row["title"],
                    "source_url": row["url"],
                    "reason": "not requested after minimum eligibility became mathematically impossible",
                }
                for index, row in enumerate(
                    contract["rows"][selection_rank:], start=selection_rank
                )
            )
            break
        extension = ".png" if expected["mime"] == "image/png" else ".jpg"
        path = source_root / "originals" / f"{expected['id']}{extension}"
        _download_exact(expected, path)
        with Image.open(path) as image:
            size = image.size
            profile = image.info.get("icc_profile")
        base_facts = {
            "id": expected["id"],
            "selection_rank": selection_rank,
            "path": path.relative_to(root).as_posix(),
            "sha256": hash_file(path),
            "sha1": _sha1_file(path),
            "bytes": path.stat().st_size,
            "title": expected["title"],
            "artist": expected["artist"],
            "license": expected["license"],
            "source_url": expected["url"],
            "source_page": expected["page"],
            "mime": expected["mime"],
            "width": size[0],
            "height": size[1],
        }
        try:
            if size != (int(expected["width"]), int(expected["height"])):
                raise CommonsProPhotoSourceError("dimension mismatch")
            if not isinstance(profile, bytes):
                raise CommonsProPhotoSourceError("embedded ICC is absent")
            profile_facts = _profile_facts(profile)
            rgb, bit_depth = _decode_rgb(path, expected["mime"])
            preview, dhash = _preview_and_dhash(path, profile)
        except CommonsProPhotoSourceError as exc:
            rejected_rows.append({**base_facts, "rejection_reason": str(exc)})
            continue
        rows.append(
            {
                **base_facts,
                "bit_depth": bit_depth,
                "pixel_array_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "dhash64": dhash,
                "precompression_rec2020_out_of_gamut_fraction": _rec2020_oog_fraction(
                    rgb
                ),
                **profile_facts,
            }
        )
        previews.append(preview)
    exact_pixel_duplicates = len(rows) - len(
        {row["pixel_array_sha256"] for row in rows}
    )
    near_pairs = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            distance = (
                int(left["dhash64"], 16) ^ int(right["dhash64"], 16)
            ).bit_count()
            if distance <= int(
                contract["eligibility"]["maximum_dhash_distance_for_cross_row_pair"]
            ):
                near_pairs.append([left["id"], right["id"], distance])
    gates = contract["eligibility"]
    requested_count = len(rows) + len(rejected_rows)
    checks = {
        "source_inventory_accounted": requested_count + len(unrequested_rows)
        == int(gates["required_source_count"]),
        "network_request_count": requested_count
        == int(contract["storage"]["network_requests_exact"]),
        "eligible_source_count": len(rows)
        >= int(gates["minimum_eligible_source_count"]),
        "artist_count": len({row["artist"] for row in rows})
        >= int(gates["minimum_distinct_artist_count"]),
        "native_16bit_count": sum(row["bit_depth"] == 16 for row in rows)
        >= int(gates["minimum_native_16bit_sources"]),
        "rec2020_oog_count": sum(
            row["precompression_rec2020_out_of_gamut_fraction"] > 0.0 for row in rows
        )
        >= int(gates["minimum_sources_with_nonzero_rec2020_oog_fraction"]),
        "exact_pixel_duplicates": exact_pixel_duplicates
        <= int(gates["maximum_exact_pixel_duplicate_pairs"]),
        "near_duplicate_pairs": len(near_pairs)
        <= int(gates["maximum_cross_row_pairs_at_or_below_dhash_threshold"]),
    }
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "requested_source_count": requested_count,
        "unrequested_source_count": len(unrequested_rows),
        "eligible_source_count": len(rows),
        "rejected_source_count": len(rejected_rows),
        "artist_count": len({row["artist"] for row in rows}),
        "native_16bit_count": sum(row["bit_depth"] == 16 for row in rows),
        "nonzero_rec2020_oog_count": sum(
            row["precompression_rec2020_out_of_gamut_fraction"] > 0.0 for row in rows
        ),
        "exact_pixel_duplicate_pairs": exact_pixel_duplicates,
        "near_duplicate_pairs": near_pairs,
        "rows": rows,
        "rejected_rows": rejected_rows,
        "unrequested_rows": unrequested_rows,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "visual_review_required": bool(
            contract["eligibility"]["visual_review_required_before_algorithm_role"]
        ),
        "algorithm_evaluation_allowed": False,
        "decision": "open_visual_source_review_only"
        if all(checks.values())
        else "close_u1_4c5_before_visual_review",
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable["stable_evidence_id"] = hashlib.sha256(canonical_json(stable)).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (640, max(1, len(previews)) * 270), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (row, preview) in enumerate(zip(rows, previews, strict=True)):
        y = index * 270
        draw.text(
            (5, y + 5),
            f"{row['id']} | {row['bit_depth']}b | {row['artist']}",
            fill="white",
        )
        sheet.paste(preview, (5, y + 25))
    sheet.save(output_dir / "contact_sheet.png")
    return stable


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CommonsProPhotoSourceError",
    "_icc_xyz_tag",
    "evaluate",
    "load_contract",
    "write_report",
]

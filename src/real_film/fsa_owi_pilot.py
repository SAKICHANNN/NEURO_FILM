"""Bounded pixel pilot for the canonical FSA/OWI archive subset."""

from __future__ import annotations

import hashlib
import io
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps

from src.real_film.fsa_owi import FsaOwiAcquisitionError


UNKNOWN_CREATOR = "__unknown_creator__"


def select_creator_balanced(
    records: Sequence[Mapping[str, Any]], *, count: int, seed: str
) -> list[dict[str, Any]]:
    """Round-robin a stable hash order across curated creator groups."""
    if count <= 0 or count > len(records):
        raise FsaOwiAcquisitionError("invalid pilot selection count")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        group = str(record.get("creator_group") or UNKNOWN_CREATOR)
        groups[group].append(record)
    ordered: dict[str, list[Mapping[str, Any]]] = {}
    for group, members in groups.items():
        ordered[group] = sorted(
            members,
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}\0{group}\0{row['loc_fsac_id']}".encode()
                ).hexdigest(),
                str(row["loc_fsac_id"]),
            ),
        )
    positions = Counter()
    selected: list[dict[str, Any]] = []
    group_names = sorted(ordered)
    while len(selected) < count:
        progressed = False
        for group in group_names:
            index = positions[group]
            if index >= len(ordered[group]):
                continue
            selected.append(dict(ordered[group][index]))
            positions[group] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise FsaOwiAcquisitionError("pilot selection exhausted unexpectedly")
    for index, record in enumerate(selected):
        record["pilot_index"] = index
        record["pilot_creator_group"] = str(
            record.get("creator_group") or UNKNOWN_CREATOR
        )
    return selected


def dhash64(image: Image.Image) -> str:
    """Compute a deterministic 64-bit difference hash for duplicate triage."""
    gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def download_pilot(
    selected: Sequence[Mapping[str, Any]],
    *,
    image_dir: Path,
    maximum_total_bytes: int,
    timeout_seconds: int,
    max_retries: int,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download, decode and hash a bounded pilot without modifying pixels."""
    client = session or requests.Session()
    client.headers["User-Agent"] = (
        "neuro-film-research/0.1 "
        "(https://github.com/SAKICHANNN/NEURO_FILM; bounded archive pilot)"
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = 0
    for record in selected:
        url = str(record.get("derivative_url") or record.get("original_url") or "")
        if not url:
            raise FsaOwiAcquisitionError("pilot record has no image URL")
        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.get(url, timeout=timeout_seconds)
                response.raise_for_status()
                last_error = None
                break
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 == max_retries:
                    break
        if response is None or last_error is not None:
            raise FsaOwiAcquisitionError(f"pilot image request failed: {last_error}")
        payload = response.content
        if not payload or total + len(payload) > maximum_total_bytes:
            raise FsaOwiAcquisitionError("pilot aggregate byte cap would be exceeded")
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in {"image/jpeg", "image/png", "image/tiff"}:
            raise FsaOwiAcquisitionError(f"unexpected pilot content type: {content_type}")
        try:
            with Image.open(io.BytesIO(payload)) as verify_image:
                verify_image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                image_format = image.format
                icc = image.info.get("icc_profile", b"")
                exif = image.getexif()
                difference_hash = dhash64(image)
        except Exception as error:
            raise FsaOwiAcquisitionError("pilot image failed Pillow decode") from error
        suffix = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}.get(
            str(image_format), ".bin"
        )
        identifier = str(record["loc_fsac_id"]).replace(".", "_")
        path = image_dir / f"{int(record['pilot_index']):03d}_{identifier}{suffix}"
        path.write_bytes(payload)
        total += len(payload)
        rows.append(
            {
                "pilot_index": int(record["pilot_index"]),
                "loc_fsac_id": record["loc_fsac_id"],
                "creator_group": record["pilot_creator_group"],
                "commons_page_id": record["commons_page_id"],
                "commons_title": record["commons_title"],
                "source_url": record["source_url"],
                "download_url": url,
                "local_path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "dhash64": difference_hash,
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "content_type": content_type,
                "icc_profile_bytes": len(icc),
                "icc_profile_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
                "exif_tags": len(exif),
            }
        )
    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            distance = hamming64(str(left["dhash64"]), str(right["dhash64"]))
            if distance <= 4:
                near_pairs.append(
                    {
                        "left_loc_fsac_id": left["loc_fsac_id"],
                        "right_loc_fsac_id": right["loc_fsac_id"],
                        "hamming_distance": distance,
                    }
                )
    return rows, near_pairs


def render_contact_sheets(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, *, per_sheet: int = 32
) -> list[Path]:
    """Render review-only contact sheets without altering downloaded evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cell_width, cell_height = 280, 230
    columns = 4
    for sheet_index, start in enumerate(range(0, len(rows), per_sheet)):
        chunk = rows[start : start + per_sheet]
        canvas = Image.new(
            "RGB",
            (columns * cell_width, ((len(chunk) + columns - 1) // columns) * cell_height),
            "#202020",
        )
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(chunk):
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            with Image.open(str(row["local_path"])) as image:
                preview = ImageOps.contain(
                    ImageOps.exif_transpose(image).convert("RGB"), (cell_width - 12, 185)
                )
            px = x + (cell_width - preview.width) // 2
            py = y + 4
            canvas.paste(preview, (px, py))
            label = f"{row['pilot_index']:02d} {row['loc_fsac_id']} {row['creator_group']}"
            draw.text((x + 6, y + 193), label[:45], fill="white")
        path = output_dir / f"contact_sheet_{sheet_index + 1:02d}.jpg"
        canvas.save(path, quality=92, subsampling=0)
        paths.append(path)
    return paths

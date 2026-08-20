"""Deterministic selection, bounded download and audit for Commons stock pixels."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import requests
from PIL import Image, ImageDraw, ImageOps, ImageStat

from src.real_film.commons_stock_source import normalize_author
from src.real_film.fsa_owi_pilot import dhash64, hamming64


class CommonsStockPilotError(ValueError):
    """Raised when the frozen SF0.5 contract fails closed."""


def resolve_download_url(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Replace Commons' non-thumbnail tracking URL with a real bounded JPEG thumb."""

    source = str(row["derivative_1600_url"])
    cap = config.get("download_limits", {}).get("thumbnail_unscaled_max_width")
    if cap is None or "utm_content=thumbnail_unscaled" not in source:
        return source
    parsed = urlsplit(source)
    prefix = "/wikipedia/commons/"
    if parsed.scheme != "https" or parsed.netloc != "upload.wikimedia.org":
        raise CommonsStockPilotError("unscaled Commons URL host drifted")
    if not parsed.path.startswith(prefix) or str(row.get("mime")) != "image/jpeg":
        raise CommonsStockPilotError("unsupported unscaled Commons payload")
    relative = parsed.path[len(prefix) :]
    parts = relative.split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise CommonsStockPilotError("unscaled Commons path drifted")
    filename = parts[-1]
    width = int(cap)
    if int(row["width"]) <= width:
        raise CommonsStockPilotError("source lacks the required standard Commons thumbnail")
    if unquote(filename).casefold().endswith((".jpg", ".jpeg")) is False:
        raise CommonsStockPilotError("invalid bounded Commons JPEG thumbnail")
    thumbnail_path = f"{prefix}thumb/{relative}/{width}px-{filename}"
    return urlunsplit((parsed.scheme, parsed.netloc, thumbnail_path, "", ""))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return sha256_bytes(encoded)


def _rights_eligible(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    policy = config["rights_filter"]
    licence = str(row.get("license_short_name", ""))
    common = all(
        str(row.get(key, "")).strip()
        for key in ("author_raw_html", "file_page_url", "original_url", "derivative_1600_url")
    )
    if not common or row["derivative_1600_url"] == row["original_url"]:
        return False
    if licence in set(policy["allowed_licenses_with_explicit_url"]):
        return bool(str(row.get("license_url", "")).strip())
    if licence == "Public domain" and policy["allow_public_domain_with_usage_terms_and_file_page"]:
        return bool(str(row.get("usage_terms", "")).strip())
    return False


def select_pixel_rows(
    snapshot: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply the frozen rights filter and stable dual-group-capped round robin."""
    allowed = set(config["allowed_stock_ids"])
    selection = config["selection"]
    selected: list[dict[str, Any]] = []
    categories = {
        str(category["film_stock_id"]): category
        for category in snapshot.get("categories", [])
    }
    if set(categories).intersection(allowed) != allowed:
        raise CommonsStockPilotError("snapshot is missing an allowed exact stock")
    for stock_id in sorted(allowed):
        category = categories[stock_id]
        if category.get("label_scope") != config["allowed_label_scope"]:
            raise CommonsStockPilotError(f"non-exact label scope: {stock_id}")
        eligible = [dict(row) for row in category["files"] if _rights_eligible(row, config)]
        by_uploader: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            author_group = normalize_author(str(row["author_raw_html"]))
            if not author_group:
                raise CommonsStockPilotError(f"empty normalized author: {row['title']}")
            row["normalized_author_group"] = author_group
            by_uploader[str(row["uploader"])].append(row)
        for rows in by_uploader.values():
            rows.sort(key=lambda item: str(item["title"]).casefold())
        author_counts: Counter[str] = Counter()
        uploader_counts: Counter[str] = Counter()
        stock_rows: list[dict[str, Any]] = []
        maximum = int(selection["maximum_files_per_stock"])
        author_cap = int(selection["maximum_files_per_normalized_author_per_stock"])
        uploader_cap = int(selection["maximum_files_per_uploader_per_stock"])
        while len(stock_rows) < maximum:
            progressed = False
            for uploader in sorted(by_uploader, key=str.casefold):
                while by_uploader[uploader]:
                    row = by_uploader[uploader].pop(0)
                    author = row["normalized_author_group"]
                    if author_counts[author] >= author_cap or uploader_counts[uploader] >= uploader_cap:
                        continue
                    row["selection_index_within_stock"] = len(stock_rows)
                    stock_rows.append(row)
                    author_counts[author] += 1
                    uploader_counts[uploader] += 1
                    progressed = True
                    break
                if len(stock_rows) >= maximum:
                    break
            if not progressed:
                break
        expected = int(selection["preflight_expected_selected"][stock_id])
        if len(stock_rows) != expected:
            raise CommonsStockPilotError(
                f"selection drift for {stock_id}: {len(stock_rows)} != {expected}"
            )
        for row in stock_rows:
            row["film_stock_id"] = stock_id
            row["label_scope"] = category["label_scope"]
            row["selection_index"] = len(selected)
            selected.append(row)
    if len(selected) > int(selection["maximum_files_total"]):
        raise CommonsStockPilotError("global selection count exceeds contract")
    return selected


def build_selection_manifest(
    snapshot: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    selected = select_pixel_rows(snapshot, config)
    return {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "metadata_snapshot_sha256": config["metadata_snapshot_sha256"],
        "selection_algorithm": config["selection"]["algorithm"],
        "selected_files": len(selected),
        "selected_by_stock": dict(sorted(Counter(row["film_stock_id"] for row in selected).items())),
        "rows": selected,
        "image_payloads_downloaded_or_decoded": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def merge_metadata_snapshots(
    snapshots: Sequence[Mapping[str, Any]], *, allowed_stock_ids: Sequence[str]
) -> dict[str, Any]:
    """Merge only named stock categories from already hash-verified snapshots."""
    allowed = set(allowed_stock_ids)
    categories: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        for category in snapshot.get("categories", []):
            stock = str(category["film_stock_id"])
            if stock not in allowed:
                continue
            if stock in categories:
                raise CommonsStockPilotError(f"duplicate stock across snapshots: {stock}")
            categories[stock] = dict(category)
    if set(categories) != allowed:
        raise CommonsStockPilotError(
            f"merged snapshot stock mismatch: {sorted(categories)} != {sorted(allowed)}"
        )
    return {
        "schema_version": 1,
        "merge_policy": "exact allowed stock categories from hash-verified snapshots only",
        "image_payloads_downloaded_or_decoded": False,
        "categories": [categories[stock] for stock in sorted(categories)],
    }


def _verify_image_payload(
    payload: bytes,
    minimum_short_dimension: int,
    allowed_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            if min(width, height) < minimum_short_dimension:
                raise CommonsStockPilotError("downloaded derivative is below minimum dimension")
            if allowed_modes is not None and image.mode not in set(allowed_modes):
                raise CommonsStockPilotError(f"decoded mode is forbidden: {image.mode}")
            return {"width": width, "height": height, "mode": image.mode, "format": image.format}
    except CommonsStockPilotError:
        raise
    except Exception as exc:
        raise CommonsStockPilotError("downloaded derivative failed Pillow decode") from exc


def download_selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    config: Mapping[str, Any],
    prior_manifest: Mapping[str, Any] | None = None,
    session: requests.Session | None = None,
    timeout_seconds: int = 60,
    retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    request_interval_seconds: float = 0.0,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Download selected derivatives atomically, with strict aggregate limits."""
    limits = config["download_limits"]
    audit = config["pixel_audit"]
    client = session or requests.Session()
    client.headers["User-Agent"] = "K-MCFM-research-pixel-pilot/1.0 (bounded Commons audit)"
    prior = {
        (str(row["film_stock_id"]), int(row["page_id"])): row
        for row in (prior_manifest or {}).get("rows", [])
    }
    records: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        stock = str(row["film_stock_id"])
        page_id = int(row["page_id"])
        relative = Path(stock) / f"{page_id}.img"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = prior.get((stock, page_id))
        if target.exists():
            if previous is None:
                raise CommonsStockPilotError(f"untracked resumed file: {target}")
            payload = target.read_bytes()
            if sha256_bytes(payload) != previous["sha256"] or len(payload) != int(previous["bytes"]):
                raise CommonsStockPilotError(f"resumed file hash/size mismatch: {target}")
            content_type = str(previous.get("content_type", ""))
        else:
            response = None
            error: Exception | None = None
            effective_url = resolve_download_url(row, config)
            for attempt in range(retries):
                try:
                    response = client.get(
                        effective_url,
                        timeout=timeout_seconds,
                        stream=True,
                    )
                    response.raise_for_status()
                    error = None
                    break
                except requests.RequestException as exc:
                    error = exc
                    if attempt + 1 < retries:
                        retry_after = 0.0
                        if getattr(exc, "response", None) is not None:
                            try:
                                retry_after = float(exc.response.headers.get("retry-after", 0))
                            except (TypeError, ValueError):
                                retry_after = 0.0
                        time.sleep(min(60.0, max(retry_after, retry_backoff_seconds * float(2 ** attempt))))
            if response is None or error is not None:
                raise CommonsStockPilotError(f"download failed for page {page_id}: {error}")
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type and not content_type.startswith("image/"):
                raise CommonsStockPilotError(f"non-image response for page {page_id}: {content_type}")
            payload_buffer = bytearray()
            chunks = (
                response.iter_content(chunk_size=1024 * 1024)
                if hasattr(response, "iter_content")
                else (response.content,)
            )
            for chunk in chunks:
                if not chunk:
                    continue
                payload_buffer.extend(chunk)
                if len(payload_buffer) > int(limits["maximum_bytes_per_file"]):
                    raise CommonsStockPilotError(f"per-file byte cap failed for page {page_id}")
            payload = bytes(payload_buffer)
            if not payload:
                raise CommonsStockPilotError(f"per-file byte cap failed for page {page_id}")
            response.close()
        if total + len(payload) > int(limits["maximum_bytes_total"]):
            raise CommonsStockPilotError("aggregate byte cap would be exceeded")
        decoded = _verify_image_payload(
            payload,
            int(audit["minimum_short_dimension"]),
            audit.get("allowed_decoded_modes"),
        )
        if not target.exists():
            temporary = target.with_suffix(target.suffix + str(limits["temporary_suffix"]))
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        total += len(payload)
        records.append({
            "selection_index": int(row["selection_index"]),
            "film_stock_id": stock,
            "page_id": page_id,
            "title": row["title"],
            "normalized_author_group": row["normalized_author_group"],
            "uploader": row["uploader"],
            "author_raw_html": row["author_raw_html"],
            "credit_raw_html": row["credit_raw_html"],
            "license_short_name": row["license_short_name"],
            "license_url": row["license_url"],
            "usage_terms": row["usage_terms"],
            "file_page_url": row["file_page_url"],
            "original_url": row["original_url"],
            "api_original_sha1_base36": row["api_sha1_base36"],
            "derivative_url": row["derivative_1600_url"],
            "effective_download_url": resolve_download_url(row, config),
            "local_path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "content_type": content_type,
            **decoded,
        })
        if checkpoint_path is not None:
            atomic_json(checkpoint_path, {
                "schema_version": 1,
                "pilot_id": config["pilot_id"],
                "files": len(records),
                "bytes": total,
                "complete": False,
                "all_files_sha256_and_decode_verified": True,
                "rows": records,
                "claim_ceiling": config["claim_ceiling"],
            })
        if previous is None and request_interval_seconds > 0:
            time.sleep(request_interval_seconds)
    return {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "files": len(records),
        "bytes": total,
        "complete": True,
        "all_files_sha256_and_decode_verified": True,
        "rows": records,
        "claim_ceiling": config["claim_ceiling"],
    }


def _image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
        gray = rgb.convert("L")
        extrema = ImageStat.Stat(rgb).extrema
        pixels = list(gray.get_flattened_data())
        return {
            "dhash64": dhash64(rgb),
            "black_fraction_luma_le_1": sum(value <= 1 for value in pixels) / len(pixels),
            "white_fraction_luma_ge_254": sum(value >= 254 for value in pixels) / len(pixels),
            "rgb_extrema": [[int(low), int(high)] for low, high in extrema],
        }


def audit_download_manifest(
    manifest: Mapping[str, Any], *, root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sha_groups: dict[str, list[str]] = defaultdict(list)
    for source in sorted(manifest["rows"], key=lambda row: int(row["selection_index"])):
        path = root / Path(str(source["local_path"]))
        if not path.is_file():
            raise CommonsStockPilotError(f"missing downloaded pixel: {path}")
        digest = sha256_file(path)
        if digest != source["sha256"] or path.stat().st_size != int(source["bytes"]):
            raise CommonsStockPilotError(f"pixel hash/size drift: {path}")
        decoded = _verify_image_payload(
            path.read_bytes(),
            int(config["pixel_audit"]["minimum_short_dimension"]),
            config["pixel_audit"].get("allowed_decoded_modes"),
        )
        metrics = _image_metrics(path)
        record = {**dict(source), **decoded, **metrics}
        rows.append(record)
        sha_groups[digest].append(str(source["local_path"]))
    exact = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(sha_groups.items()) if len(paths) > 1
    ]
    threshold = int(config["pixel_audit"]["near_duplicate_hamming_threshold"])
    near: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            distance = hamming64(left["dhash64"], right["dhash64"])
            if distance <= threshold:
                near.append({
                    "left": left["local_path"], "right": right["local_path"],
                    "left_stock": left["film_stock_id"], "right_stock": right["film_stock_id"],
                    "hamming": distance,
                })
    by_stock: dict[str, Any] = {}
    gates = config["pixel_audit"]
    for stock in sorted(config["allowed_stock_ids"]):
        stock_rows = [row for row in rows if row["film_stock_id"] == stock]
        authors = Counter(row["normalized_author_group"] for row in stock_rows)
        largest = max(authors.values(), default=0) / max(len(stock_rows), 1)
        checks = {
            "minimum_retained_files": len(stock_rows) >= int(gates["minimum_retained_files_per_stock"]),
            "minimum_author_groups": len(authors) >= int(gates["minimum_normalized_author_groups_for_learning"]),
            "largest_author_share": largest <= float(gates["maximum_largest_normalized_author_share_for_learning"]),
        }
        by_stock[stock] = {
            "files": len(stock_rows),
            "normalized_author_groups": len(authors),
            "largest_author_share": largest,
            "uploader_groups": len({row["uploader"] for row in stock_rows}),
            "checks": checks,
            "learning_source_gate_passed": all(checks.values()),
        }
    cross_stock_near = [row for row in near if row["left_stock"] != row["right_stock"]]
    cross_stock_exact = [
        group for group in exact
        if len({next(row["film_stock_id"] for row in rows if row["local_path"] == path) for path in group["paths"]}) > 1
    ]
    integrity_passed = not exact and not cross_stock_near
    return {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "integrity_passed": integrity_passed,
        "exact_duplicate_groups": exact,
        "near_duplicate_pairs_dhash_le_4": near,
        "cross_stock_exact_duplicate_groups": cross_stock_exact,
        "cross_stock_near_duplicate_pairs": cross_stock_near,
        "stock_source_gates": by_stock,
        "visual_adjudication_status": "pending Codex vision adjudication",
        "training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
        "file_records": rows,
    }


def render_contact_sheets(
    records: Sequence[Mapping[str, Any]], *, root: Path, output_dir: Path, per_sheet: int = 24
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    by_stock: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_stock[str(row["film_stock_id"])].append(row)
    cell_width, cell_height, columns = 320, 250, 4
    for stock, stock_rows in sorted(by_stock.items()):
        ordered = sorted(stock_rows, key=lambda row: int(row["selection_index"]))
        for sheet_index, start in enumerate(range(0, len(ordered), per_sheet)):
            chunk = ordered[start:start + per_sheet]
            rows_count = (len(chunk) + columns - 1) // columns
            canvas = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "#202020")
            draw = ImageDraw.Draw(canvas)
            for offset, row in enumerate(chunk):
                x, y = (offset % columns) * cell_width, (offset // columns) * cell_height
                with Image.open(root / Path(str(row["local_path"]))) as image:
                    preview = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (306, 205))
                canvas.paste(preview, (x + (cell_width - preview.width) // 2, y + 4))
                draw.text((x + 7, y + 214), f"{row['selection_index']:03d} p{row['page_id']}", fill="white")
            path = output_dir / f"contact_{stock}_{sheet_index + 1:02d}.jpg"
            canvas.save(path, quality=92, subsampling=0)
            sheets.append({"film_stock_id": stock, "path": path.as_posix(), "files": len(chunk), "sha256": sha256_file(path)})
    return sheets

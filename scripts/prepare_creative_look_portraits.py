"""Bounded, source-locked LOOK2 portrait development intake, never confirmation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.raster_decode import (
    load_jpeg_preview_working_image,
    working_image_to_legacy_srgb8,
)

CONFIG = ROOT / "configs/creative_looks_v2_portrait_development_v1.json"
HEADERS = {"User-Agent": "NeuroFilm-LookDevelopment/1.0 (bounded private evaluation)"}


def verify_payload(payload: bytes, row: dict) -> None:
    if len(payload) != row["size"] or hashlib.sha1(payload).hexdigest() != row["sha1"]:
        raise ValueError("source size/SHA1 mismatch")


def validate_metadata(config: dict, pages: dict) -> None:
    for row in config["rows"]:
        page = pages[str(row["page_id"])]
        info = page["imageinfo"][0]
        if page["title"] != row["title"]:
            raise ValueError("source title drift")
        if page["revisions"][0]["revid"] != row["page_revision"]:
            raise ValueError("rights-page revision drift")
        for key in ("size", "sha1", "width", "height"):
            if info[key] != row[key]:
                raise ValueError(f"source {key} drift")
        if info["extmetadata"]["LicenseShortName"]["value"] != "CC0":
            raise ValueError("source license drift")
        if info["url"].split("?", 1)[0] != row["url"]:
            raise ValueError("source URL drift")


def data_root(relative: str) -> Path:
    path = ROOT / relative
    if not path.resolve().is_relative_to(Path("P:/neuro_film_storage")):
        raise ValueError("requires canonical project P-backed storage")
    return path


def fetch_exact(row: dict) -> bytes:
    if urlsplit(row["url"]).netloc != "upload.wikimedia.org":
        raise ValueError("unexpected media host")
    with requests.get(
        row["url"],
        headers=HEADERS,
        timeout=(15, 60),
        stream=True,
        allow_redirects=False,
    ) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("unexpected media response")
        payload = bytearray()
        for part in response.iter_content(65536):
            payload.extend(part)
            if len(payload) > row["size"]:
                raise ValueError("source exceeds frozen size")
    verify_payload(payload, row)
    return bytes(payload)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def prepare() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["role"] != "development_only_never_confirmation":
        raise ValueError("incorrect source role")
    if (
        sum(row["size"] for row in config["rows"])
        > config["maximum_total_source_bytes"]
    ):
        raise ValueError("source budget exceeded")
    destination = data_root(config["source_directory"])
    output = data_root(config["output_directory"])
    if output.exists():
        raise FileExistsError(output)
    response = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "prop": "imageinfo|revisions",
            "rvprop": "ids",
            "iiprop": "url|size|sha1|extmetadata",
            "titles": "|".join(row["title"] for row in config["rows"]),
        },
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    snapshot = response.json()
    validate_metadata(config, snapshot["query"]["pages"])
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination / "source_metadata.json"
    if not snapshot_path.exists():
        write_json(snapshot_path, snapshot)
    else:
        validate_metadata(
            config, json.loads(snapshot_path.read_text())["query"]["pages"]
        )
    # Verify every source before any decoder/pixel access.
    for row in config["rows"]:
        path = destination / (row["id"] + ".jpg")
        if path.exists():
            verify_payload(path.read_bytes(), row)
        else:
            payload = fetch_exact(row)
            with path.open("xb") as stream:
                stream.write(payload)
        print(f"source verified: {row['id']}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rows = []
    for row in config["rows"]:
        source = destination / (row["id"] + ".jpg")
        ratio = 1024 / max(row["width"], row["height"])
        working = load_jpeg_preview_working_image(
            source,
            target_width=int(row["width"] * ratio),
            target_height=int(row["height"] * ratio),
        )
        image = working_image_to_legacy_srgb8(working)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        target = output / (row["id"] + "_source.png")
        with target.open("xb") as stream:
            image.save(stream, format="PNG", icc_profile=profile)
        rows.append(
            {
                "id": row["id"],
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "preview_path": target.relative_to(ROOT).as_posix(),
                "preview_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "preview_size": list(image.size),
                "decode_warnings": [asdict(warning) for warning in working.warnings],
                "color_state": "ICC-aware display-sRGB preview, not RAW or colour truth",
            }
        )
    report = {
        "status": "DEVELOPMENT_COVERAGE_PENDING_VISUAL_REVIEW",
        "role": config["role"],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(
            snapshot_path.read_bytes()
        ).hexdigest(),
        "total_source_bytes": sum(row["size"] for row in config["rows"]),
        "candidate_renders": 0,
        "confirmation_reads": 0,
        "rows": rows,
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))

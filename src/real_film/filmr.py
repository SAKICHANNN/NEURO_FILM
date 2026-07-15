"""Fail-closed acquisition and manifest helpers for the FILM-R dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable


FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
RESTORED_SUFFIX = "_restored.jpg"
FAMILY_PATTERN = re.compile(r"^(?P<family>.+?)_(?:half|135)_\d+$")


class FilmRContractError(ValueError):
    """Raised when remote metadata or local bytes violate the frozen contract."""


def hash_file(path: Path, algorithm: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_article(article_id: int) -> dict[str, Any]:
    request = urllib.request.Request(
        FIGSHARE_API.format(article_id=article_id),
        headers={"User-Agent": "neuro-film-real-film-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def validate_article(article: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    checks = {
        "article id": (int(article["id"]), int(config["article_id"])),
        "article version": (int(article["version"]), int(config["article_version"])),
        "DOI": (str(article["doi"]), str(config["doi"])),
        "license name": (str(article["license"]["name"]), str(config["license"]["name"])),
        "license URL": (str(article["license"]["url"]), str(config["license"]["url"])),
        "file count": (len(article["files"]), int(config["expected_files"])),
        "total bytes": (
            sum(int(item["size"]) for item in article["files"]),
            int(config["expected_bytes"]),
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise FilmRContractError(f"FILM-R {label} mismatch: {observed!r} != {expected!r}")
    names = [str(item["name"]) for item in article["files"]]
    if len(names) != len(set(names)):
        raise FilmRContractError("FILM-R metadata contains duplicate filenames")
    for item in article["files"]:
        name = str(item["name"])
        if Path(name).name != name or Path(name).is_absolute():
            raise FilmRContractError(f"unsafe FILM-R filename: {name!r}")
        supplied = str(item.get("supplied_md5", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{32}", supplied):
            raise FilmRContractError(f"missing/invalid FILM-R MD5: {name}")
        if not str(item.get("download_url", "")).startswith("https://"):
            raise FilmRContractError(f"unsafe FILM-R download URL: {name}")
    return sorted(article["files"], key=lambda item: str(item["name"]))


def build_pair_manifest(
    files: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    by_name = {str(item["name"]): item for item in files}
    originals = sorted(name for name in by_name if not name.endswith(RESTORED_SUFFIX))
    rows: list[dict[str, Any]] = []
    for original in originals:
        stem = original[: -len(".jpg")]
        restored = stem + RESTORED_SUFFIX
        if restored not in by_name:
            raise FilmRContractError(f"FILM-R original lacks restored pair: {original}")
        match = FAMILY_PATTERN.fullmatch(stem)
        if not match:
            raise FilmRContractError(f"FILM-R filename family is unparseable: {original}")
        original_path = output_dir / original
        restored_path = output_dir / restored
        rows.append(
            {
                "schema_version": "real_film.filmr_pair.v1",
                "pair_id": stem,
                "filename_family_claim": match.group("family"),
                "original_scan": _file_record(by_name[original], original_path),
                "expert_restoration": _file_record(by_name[restored], restored_path),
                "physical_medium": "verified_35mm_colour_film_scan",
                "source_contributor": "d@analog.cafe",
                "physical_roll_id": None,
                "process_id": None,
                "scanner_id": None,
                "scanner_settings": None,
                "development_batch": None,
                "allowed_use": "CC-BY-4.0-with-attribution",
                "claim_ceiling": config["claim_ceiling"],
            }
        )
    expected_pairs = int(config["expected_pairs"])
    if len(rows) != expected_pairs or len(rows) * 2 != len(files):
        raise FilmRContractError(
            f"FILM-R pair count mismatch: {len(rows)} != {expected_pairs}"
        )
    return rows


def _file_record(remote: dict[str, Any], path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FilmRContractError(f"FILM-R file is missing after acquisition: {path.name}")
    size = path.stat().st_size
    if size != int(remote["size"]):
        raise FilmRContractError(f"FILM-R file size mismatch: {path.name}")
    md5 = hash_file(path, "md5")
    if md5.lower() != str(remote["supplied_md5"]).lower():
        raise FilmRContractError(f"FILM-R file MD5 mismatch: {path.name}")
    return {
        "figshare_file_id": int(remote["id"]),
        "filename": path.name,
        "bytes": size,
        "supplied_md5": str(remote["supplied_md5"]).lower(),
        "sha256": hash_file(path, "sha256"),
    }


def download_file(
    remote: dict[str, Any],
    destination: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(remote["size"])
    expected_md5 = str(remote["supplied_md5"]).lower()
    if destination.is_file():
        if destination.stat().st_size == expected_size and hash_file(destination, "md5") == expected_md5:
            return
        raise FilmRContractError(f"existing FILM-R destination is invalid: {destination.name}")
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise FilmRContractError(f"oversized FILM-R partial: {destination.name}")
    headers = {"User-Agent": "neuro-film-real-film-audit/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(str(remote["download_url"]), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(4 * 1024 * 1024):
                handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise FilmRContractError(f"incomplete FILM-R download: {destination.name}")
    if hash_file(partial, "md5") != expected_md5:
        raise FilmRContractError(f"FILM-R downloaded MD5 mismatch: {destination.name}")
    os.replace(partial, destination)
    if progress:
        progress(destination.name)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

#!/usr/bin/env python
"""Acquire the exact bounded CHAM4 same-scene Portra 400 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import gdown
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "neuro-film.u5-r2cham4-portra400-same-scene-source-contract.v1"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


class SourceError(RuntimeError):
    """Raised when the frozen public inventory or payload drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["acquisition"]["expected_files"]
    if (
        value.get("schema") != SCHEMA
        or len(rows) != 6
        or len({row["file_id"] for row in rows}) != 6
        or len({row["name"] for row in rows}) != 6
        or len({row["role"] for row in rows}) != 6
        or sum(int(row["bytes"]) for row in rows)
        != int(value["acquisition"]["expected_total_bytes"])
        or int(value["acquisition"]["expected_total_bytes"])
        > int(value["acquisition"]["maximum_total_bytes"])
        or value["source"]["explicit_reuse_licence_observed"]
        or value["gates"]["operator_fit_allowed"]
        or value["gates"]["training_allowed"]
        or value["gates"]["product_integration_allowed"]
    ):
        raise SourceError("unsupported CHAM4 source contract")
    return value


def enumerate_remote(contract: dict) -> list[dict]:
    items = gdown.download_folder(
        id=contract["source"]["folder_id"],
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if items is None:
        raise SourceError("CHAM4 remote inventory unavailable")
    observed = {item.id: Path(item.path).name for item in items}
    expected = {
        str(row["file_id"]): str(row["name"])
        for row in contract["acquisition"]["expected_files"]
    }
    if observed != expected:
        raise SourceError("CHAM4 remote inventory drift")
    return [dict(row) for row in contract["acquisition"]["expected_files"]]


def _download(session: requests.Session, row: dict, path: Path) -> dict:
    expected = int(row["bytes"])
    if path.is_file() and path.stat().st_size == expected:
        return {**row, "sha256": _sha256(path)}
    if path.exists():
        raise SourceError(f"non-file destination: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    last: Exception | None = None
    for attempt in range(4):
        try:
            with session.get(
                DOWNLOAD_URL.format(file_id=row["file_id"]),
                stream=True,
                timeout=(15, 180),
            ) as response:
                response.raise_for_status()
                if int(response.headers.get("content-length", "-1")) != expected:
                    raise SourceError(f"remote size drift: {row['name']}")
                with part.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if part.stat().st_size != expected:
                raise SourceError(f"download size drift: {row['name']}")
            part.replace(path)
            return {**row, "sha256": _sha256(path)}
        except (OSError, requests.RequestException, SourceError) as error:
            last = error
            if part.exists():
                part.unlink()
            if attempt < 3:
                time.sleep(2**attempt)
    raise SourceError(f"download failed: {row['name']}") from last


def acquire(contract: dict, root: Path) -> dict:
    rows = enumerate_remote(contract)
    destination = root / contract["acquisition"]["root"]
    session = requests.Session()
    session.headers["User-Agent"] = "neuro-film-u5-r2cham4/1"
    files = [_download(session, row, destination / row["name"]) for row in rows]
    total = sum(int(row["bytes"]) for row in files)
    if total != int(contract["acquisition"]["expected_total_bytes"]):
        raise SourceError("CHAM4 byte total drift")
    return {
        "schema": "neuro-film.u5-r2cham4-download-manifest.v1",
        "experiment_id": contract["experiment_id"],
        "file_count": len(files),
        "bytes": total,
        "files": files,
        "allowed_use": contract["source"]["allowed_use"],
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_manifest(value: dict, path: Path) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2cham4_portra400_same_scene_source_v1.json",
    )
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.config.resolve())
    if args.inventory_only:
        rows = enumerate_remote(contract)
        print(
            json.dumps(
                {"files": len(rows), "bytes": sum(int(row["bytes"]) for row in rows)},
                sort_keys=True,
            )
        )
        return 0
    manifest = acquire(contract, ROOT)
    digest = write_manifest(manifest, ROOT / contract["acquisition"]["manifest"])
    print(
        json.dumps(
            {
                "manifest_sha256": digest,
                "files": manifest["file_count"],
                "bytes": manifest["bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

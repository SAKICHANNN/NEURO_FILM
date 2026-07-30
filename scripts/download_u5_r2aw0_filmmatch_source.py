"""Acquire the exact bounded FilmMatch Ektachrome/Sony chart source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FilmMatchAcquisitionError(RuntimeError):
    """Raised when the frozen remote or local inventory drifts."""


_SEQUENCE = re.compile(r"_1\.(\d+)\.1\.tif$", re.IGNORECASE)
_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm=t"
)


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence(name: str) -> int:
    match = _SEQUENCE.search(name)
    if not match:
        raise FilmMatchAcquisitionError(f"unexpected chart filename: {name}")
    return int(match.group(1))


def validate_remote_inventory(rows: list[dict], folder: dict) -> list[dict]:
    expected_count = int(folder["expected_files"])
    if len(rows) != expected_count:
        raise FilmMatchAcquisitionError(
            f"{folder['lane']} count drifted: {len(rows)} != {expected_count}"
        )
    ids = [str(row["file_id"]) for row in rows]
    names = [str(row["name"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise FilmMatchAcquisitionError(f"{folder['lane']} contains duplicates")
    if "expected_sequence" in folder:
        start, end = map(int, folder["expected_sequence"])
        observed = sorted(_sequence(name) for name in names)
        if observed != list(range(start, end + 1)):
            raise FilmMatchAcquisitionError(
                f"{folder['lane']} sequence drifted: {observed}"
            )
        rows.sort(key=lambda row: _sequence(str(row["name"])))
    else:
        rows.sort(key=lambda row: str(row["name"]))
    return rows


def enumerate_folder(folder: dict, destination: Path) -> list[dict]:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise FilmMatchAcquisitionError("gdown==5.2.0 is required") from exc

    files = gdown.download_folder(
        id=str(folder["folder_id"]),
        output=str(destination),
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if files is None:
        raise FilmMatchAcquisitionError(
            f"could not enumerate {folder['lane']} folder"
        )
    rows = [
        {
            "file_id": item.id,
            "name": Path(item.path).name,
            "lane": folder["lane"],
        }
        for item in files
    ]
    return validate_remote_inventory(rows, folder)


def _expected_size(row: dict, folder: dict) -> int:
    if "expected_file_bytes" in folder:
        return int(folder["expected_file_bytes"])
    suffix = Path(str(row["name"])).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return int(folder["expected_tiff_file_bytes"])
    sizes = sorted(map(int, folder["expected_drx_file_bytes"]))
    names = sorted(
        name
        for name in (str(item) for item in row.get("lane_names", []))
        if Path(name).suffix.lower() == ".drx"
    )
    if names:
        return sizes[names.index(str(row["name"]))]
    number = _sequence_validation(str(row["name"]))
    return sizes[number]


def _sequence_validation(name: str) -> int:
    if name.endswith("1.19.1.drx"):
        return 0
    if name.endswith("1.20.1.drx"):
        return 1
    raise FilmMatchAcquisitionError(f"unexpected validation filename: {name}")


def download_one(
    session: requests.Session,
    row: dict,
    destination: Path,
    expected_size: int,
    *,
    retries: int,
) -> dict:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return {
            **row,
            "bytes": expected_size,
            "sha256": _sha256(destination),
            "relative_path": destination.as_posix(),
        }
    if destination.exists() and not destination.is_file():
        raise FilmMatchAcquisitionError(f"destination is not a file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = _DOWNLOAD_URL.format(file_id=row["file_id"])

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", "-1"))
                if content_length != expected_size:
                    raise FilmMatchAcquisitionError(
                        f"{row['name']} remote size drifted: "
                        f"{content_length} != {expected_size}"
                    )
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise FilmMatchAcquisitionError(
                    f"{row['name']} local size drifted after download"
                )
            partial.replace(destination)
            return {
                **row,
                "bytes": expected_size,
                "sha256": _sha256(destination),
                "relative_path": destination.as_posix(),
            }
        except (OSError, requests.RequestException, FilmMatchAcquisitionError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise FilmMatchAcquisitionError(f"download failed for {row['name']}") from last_error


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def acquire(config: dict, *, root: Path, manifest_path: Path) -> dict:
    rows: list[dict] = []
    session = requests.Session()
    session.headers["User-Agent"] = "neuro-film-u5-r2aw0/1"
    for folder in config["acquisition"]["folders"]:
        destination = root / str(folder["lane"])
        remote_rows = enumerate_folder(folder, destination)
        lane_names = [str(row["name"]) for row in remote_rows]
        for row in remote_rows:
            row["lane_names"] = lane_names
            expected_size = _expected_size(row, folder)
            relative_destination = Path(str(folder["lane"])) / str(row["name"])
            result = download_one(
                session,
                row,
                root / relative_destination,
                expected_size,
                retries=4,
            )
            result["relative_path"] = relative_destination.as_posix()
            result.pop("lane_names", None)
            rows.append(result)
            time.sleep(0.15)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    if len(rows) != int(config["acquisition"]["expected_total_files"]):
        raise FilmMatchAcquisitionError("total file count drifted")
    if total_bytes != int(config["acquisition"]["expected_total_bytes"]):
        raise FilmMatchAcquisitionError("total byte count drifted")
    manifest = {
        "schema_version": "u5-r2aw0-filmmatch-download-manifest-v1",
        "experiment_id": config["experiment_id"],
        "files": rows,
        "file_count": len(rows),
        "bytes": total_bytes,
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = ROOT / config["acquisition"]["root"]
    manifest_path = ROOT / config["acquisition"]["manifest"]
    manifest = acquire(config, root=root, manifest_path=manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "files": manifest["file_count"],
                "bytes": manifest["bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

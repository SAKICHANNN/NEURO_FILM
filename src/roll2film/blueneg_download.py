"""Exact-path, resumable and hash-verified BlueNeg pilot acquisition."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


class BlueNegDownloadError(ValueError):
    """Raised when the frozen acquisition or downloaded bytes fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BlueNegDownloadConfig:
    root: Path
    acquisition_manifest: Path
    acquisition_manifest_sha256: str
    metadata_report: Path
    metadata_report_sha256: str
    output_report: Path
    repo_id: str
    revision: str
    expected_files: int
    expected_bytes: int
    workers: int = 8
    software_commit: str = "unknown"


FetchFile = Callable[[str, bool], Path]


def _safe_target(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise BlueNegDownloadError(f"unsafe acquisition path: {remote_path!r}")
    if not remote_path.startswith(("negative-preview-8bit/", "pseudogt-8bit/")):
        raise BlueNegDownloadError(f"path outside frozen BlueNeg lanes: {remote_path!r}")
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise BlueNegDownloadError(f"acquisition path escapes root: {remote_path!r}") from exc
    return target


def _load_contract(config: BlueNegDownloadConfig) -> list[dict[str, Any]]:
    if sha256_file(config.acquisition_manifest) != config.acquisition_manifest_sha256:
        raise BlueNegDownloadError("acquisition manifest hash mismatch")
    if sha256_file(config.metadata_report) != config.metadata_report_sha256:
        raise BlueNegDownloadError("metadata report hash mismatch")
    payload = json.loads(config.acquisition_manifest.read_text(encoding="utf-8"))
    if payload.get("repo_id") != config.repo_id or payload.get("revision") != config.revision:
        raise BlueNegDownloadError("acquisition repository/revision mismatch")
    rows = payload.get("files")
    if not isinstance(rows, list) or len(rows) != config.expected_files:
        raise BlueNegDownloadError("acquisition file count mismatch")
    if sum(int(row["size"]) for row in rows) != config.expected_bytes:
        raise BlueNegDownloadError("acquisition byte count mismatch")
    names = [str(row["path"]) for row in rows]
    if len(names) != len(set(names)):
        raise BlueNegDownloadError("duplicate acquisition path")
    for row in rows:
        _safe_target(config.root, str(row["path"]))
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BlueNegDownloadError(f"missing LFS SHA-256 for {row['path']}")
    return rows


def _existing_lane_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for lane in ("negative-preview-8bit", "pseudogt-8bit"):
        directory = root / lane
        if directory.is_dir():
            paths.update(
                path.relative_to(root).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )
    return paths


def download_blueneg_acquisition(
    config: BlueNegDownloadConfig,
    fetch: FetchFile,
) -> dict[str, Any]:
    """Fetch exactly the frozen files, verify them, and reject lane extras."""
    rows = _load_contract(config)
    expected_names = {str(row["path"]) for row in rows}
    extras_before = _existing_lane_paths(config.root) - expected_names
    if extras_before:
        raise BlueNegDownloadError(
            f"manifest-external BlueNeg lane files already exist: {sorted(extras_before)[:3]}"
        )

    def acquire(row: dict[str, Any]) -> dict[str, Any]:
        remote_path = str(row["path"])
        target = _safe_target(config.root, remote_path)
        expected_size = int(row["size"])
        expected_hash = str(row["sha256"])
        valid_existing = (
            target.is_file()
            and target.stat().st_size == expected_size
            and sha256_file(target) == expected_hash
        )
        if not valid_existing:
            fetched = fetch(remote_path, target.exists())
            if fetched.resolve() != target:
                raise BlueNegDownloadError(
                    f"fetch returned unexpected path for {remote_path}: {fetched}"
                )
        if not target.is_file() or target.stat().st_size != expected_size:
            raise BlueNegDownloadError(f"downloaded size mismatch: {remote_path}")
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise BlueNegDownloadError(f"downloaded SHA-256 mismatch: {remote_path}")
        return {"path": remote_path, "size": expected_size, "sha256": actual_hash}

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        verified = list(executor.map(acquire, rows))
    extras_after = _existing_lane_paths(config.root) - expected_names
    if extras_after:
        raise BlueNegDownloadError(
            f"fetch created manifest-external lane files: {sorted(extras_after)[:3]}"
        )
    verified.sort(key=lambda row: row["path"])
    report = {
        "schema_version": 1,
        "repo_id": config.repo_id,
        "revision": config.revision,
        "software_commit": config.software_commit,
        "acquisition_manifest_sha256": config.acquisition_manifest_sha256,
        "metadata_report_sha256": config.metadata_report_sha256,
        "files": len(verified),
        "bytes": sum(row["size"] for row in verified),
        "all_sizes_and_lfs_sha256_verified": True,
        "manifest_external_lane_files": 0,
        "image_payloads_decoded": False,
        "file_inventory": verified,
    }
    report_hash = _atomic_json(config.output_report, report)
    return {**report, "report_sha256": report_hash}

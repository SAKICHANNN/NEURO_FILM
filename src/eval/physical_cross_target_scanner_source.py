"""U6.P6H bounded acquisition and connectivity audit for scanner test scans."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np

from src.real_film.scanner_nuisance import _native_rgb


SCHEMA = "neuro_film.u6_p6h_cross_target_scanner_source_contract.v1"
USER_AGENT = "K-MCFM-U6-P6H/1.0 (bounded research source audit)"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6H contract")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_bytes(url: str, maximum_bytes: int) -> tuple[bytes, dict[str, str | int]]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with closing(urlopen(request, timeout=60)) as response:
        status = int(response.status)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"response exceeds {maximum_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks), {
            "status": status,
            "content_length": int(response.headers.get("Content-Length", total)),
            "etag": str(response.headers.get("ETag", "")),
            "last_modified": str(response.headers.get("Last-Modified", "")),
            "content_type": str(response.headers.get("Content-Type", "")),
        }


def _head(url: str) -> dict[str, str | int]:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with closing(urlopen(request, timeout=30)) as response:
        return {
            "status": int(response.status),
            "content_length": int(response.headers.get("Content-Length", "0")),
            "etag": str(response.headers.get("ETag", "")),
            "last_modified": str(response.headers.get("Last-Modified", "")),
            "content_type": str(response.headers.get("Content-Type", "")),
        }


def _safe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(
        normalized
        and not normalized.startswith("/")
        and not path.is_absolute()
        and ".." not in path.parts
        and ":" not in path.parts[0]
    )


def _acquire_exact(url: str, path: Path, expected_bytes: int) -> None:
    if path.is_file() and path.stat().st_size == expected_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with closing(urlopen(request, timeout=120)) as response, temporary.open(
            "wb"
        ) as handle:
            if int(response.status) != 200:
                raise ValueError(f"download status is {response.status}")
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise ValueError("download exceeds expected byte count")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if total != expected_bytes:
            raise ValueError(
                f"download size {total} does not match {expected_bytes}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inspect_archive(
    path: Path,
    *,
    maximum_members: int,
    maximum_member_bytes: int,
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    rgb_members = 0
    with ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > maximum_members:
            raise ValueError(f"{path.name} archive member count is invalid")
        if archive.testzip() is not None:
            raise ValueError(f"{path.name} failed ZIP CRC")
        for info in infos:
            if not _safe_member_name(info.filename):
                raise ValueError(f"unsafe ZIP member {info.filename!r}")
            if info.file_size > maximum_member_bytes:
                raise ValueError(f"oversize ZIP member {info.filename!r}")
            record: dict[str, Any] = {
                "name": info.filename,
                "uncompressed_bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "is_directory": info.is_dir(),
            }
            suffix = Path(info.filename).suffix.lower()
            if not info.is_dir() and suffix in {".tif", ".tiff"}:
                payload = archive.read(info)
                rgb = _native_rgb(payload)
                if not np.all(np.isfinite(rgb)):
                    raise ValueError(f"non-finite pixels in {info.filename!r}")
                rgb_members += 1
                record.update(
                    {
                        "sha256": _sha256_bytes(payload),
                        "shape": list(rgb.shape),
                        "normalized_minimum": float(np.min(rgb)),
                        "normalized_maximum": float(np.max(rgb)),
                    }
                )
            members.append(record)
    return {
        "member_count": len(members),
        "rgb_member_count": rgb_members,
        "members": members,
    }


def _connectivity(assets: list[dict[str, Any]]) -> dict[str, Any]:
    by_set: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scanner_sets: dict[str, set[int]] = defaultdict(set)
    for row in assets:
        target_set = int(row["target_set"])
        by_set[target_set].append(row)
        scanner_sets[str(row["scanner"])].add(target_set)
    return {
        "target_set_archive_counts": {
            str(key): len(value) for key, value in sorted(by_set.items())
        },
        "target_set_unique_scanner_counts": {
            str(key): len({str(row["scanner"]) for row in value})
            for key, value in sorted(by_set.items())
        },
        "target_set_unique_pipeline_counts": {
            str(key): len({str(row["role"]) for row in value})
            for key, value in sorted(by_set.items())
        },
        "scanners_repeated_across_target_sets": sorted(
            scanner
            for scanner, target_sets in scanner_sets.items()
            if len(target_sets) > 1
        ),
        "scanner_target_graph_connected_across_sets": any(
            len(target_sets) > 1 for target_sets in scanner_sets.values()
        ),
    }


def audit_cross_target_scanner_source(
    root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = load_contract(contract_path)
    parents = contract["parents"]
    for path_key, hash_key in (
        ("p6g_decision_path", "p6g_decision_sha256"),
        ("sf2_7r_contract_path", "sf2_7r_contract_sha256"),
    ):
        if _sha256_file(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"{path_key} hash mismatch")
    p6g = json.loads(
        (root / parents["p6g_decision_path"]).read_text(encoding="utf-8")
    )
    if (
        parents["p6g_safe_rgb_route_closed"] is not True
        or p6g.get("decision")
        != "close_safe_rgb_canonicalizer_on_tail_gate_retain_real_scanner_nuisance_evidence"
    ):
        raise ValueError("P6G closure is not exact")

    acquisition = contract["acquisition"]
    archives = acquisition["archives"]
    if len(archives) != int(acquisition["expected_archive_count"]):
        raise ValueError("archive contract count mismatch")
    expected_total = sum(int(row["expected_bytes"]) for row in archives)
    if expected_total != int(acquisition["expected_total_bytes"]):
        raise ValueError("archive contract total mismatch")
    if expected_total > int(acquisition["maximum_total_bytes"]):
        raise ValueError("archive contract exceeds bounded cap")

    page_payload, page_response = _fetch_bytes(
        contract["source"]["page_url"], 1024 * 1024
    )
    page_text = page_payload.decode("latin-1", errors="strict")
    required_page_phrases = (
        "made available here on this site for free",
        "scaled version",
        "Velvia 100F",
    )
    if not all(phrase in page_text for phrase in required_page_phrases):
        raise ValueError("official page statements changed")

    data_root = root / acquisition["data_root"]
    assets: list[dict[str, Any]] = []
    for row in archives:
        head = _head(str(row["url"]))
        path = data_root / str(row["path"])
        _acquire_exact(str(row["url"]), path, int(row["expected_bytes"]))
        inspection = _inspect_archive(
            path,
            maximum_members=int(acquisition["maximum_archive_members"]),
            maximum_member_bytes=int(
                acquisition["maximum_member_uncompressed_bytes"]
            ),
        )
        assets.append(
            {
                "target_set": int(row["target_set"]),
                "scanner": str(row["scanner"]),
                "software": str(row["software"]),
                "role": str(row["role"]),
                "path": str(row["path"]),
                "url": str(row["url"]),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "head": head,
                "head_contract_exact": bool(
                    head["status"] == 200
                    and head["content_length"] == int(row["expected_bytes"])
                    and head["etag"] == str(row["etag"])
                ),
                **inspection,
            }
        )

    archive_hash_counts = Counter(row["sha256"] for row in assets)
    exact_archive_duplicates = sorted(
        digest for digest, count in archive_hash_counts.items() if count > 1
    )
    connectivity = _connectivity(assets)
    gates = contract["automatic_gates"]
    checks = {
        "all_http_status_200": all(
            row["head"]["status"] == 200 for row in assets
        ),
        "all_lengths_and_etags_exact": all(
            row["head_contract_exact"] for row in assets
        ),
        "all_zip_crc_and_member_safety_pass": True,
        "all_rgb_members_decode": all(
            row["rgb_member_count"] > 0 for row in assets
        ),
        "all_archives_have_minimum_rgb_members": all(
            row["rgb_member_count"]
            >= int(acquisition["minimum_rgb_members_per_archive"])
            for row in assets
        ),
        "zero_exact_archive_duplicates": not exact_archive_duplicates,
        "same_target_multiple_scanners": all(
            connectivity["target_set_unique_scanner_counts"].get(
                str(target_set), 0
            )
            >= 2
            for target_set in gates["same_target_multiple_scanners_for_sets"]
        ),
        "bounded_total_bytes": sum(row["bytes"] for row in assets)
        == int(acquisition["expected_total_bytes"]),
    }
    stable_payload = {
        "schema": "neuro_film.u6_p6h_cross_target_scanner_source_report.v1",
        "node": contract["node"],
        "config_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "parent_hashes_verified": {
            "p6g_decision_sha256": parents["p6g_decision_sha256"],
            "sf2_7r_contract_sha256": parents["sf2_7r_contract_sha256"],
        },
        "official_page": {
            "url": contract["source"]["page_url"],
            "sha256": _sha256_bytes(page_payload),
            "response": page_response,
            "required_statements_verified": True,
        },
        "assets": assets,
        "archive_count": len(assets),
        "total_bytes": sum(row["bytes"] for row in assets),
        "total_rgb_members": sum(row["rgb_member_count"] for row in assets),
        "exact_archive_duplicates": exact_archive_duplicates,
        "connectivity": connectivity,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": contract["claim_ceiling"],
        "forbidden_claims": contract["forbidden_claims"],
    }
    stable_id = _sha256_bytes(
        json.dumps(
            stable_payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    )
    return {
        **stable_payload,
        "stable_evidence_id": stable_id,
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "_connectivity",
    "_safe_member_name",
    "audit_cross_target_scanner_source",
    "load_contract",
    "write_report",
]

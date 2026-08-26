"""Acquire and audit two frozen real DNGs for the P245 ingress guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.preprocess.dng_forward_raster as raster
from src.preprocess.dng_metadata import _walk_pages, canonical_json_bytes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _encoded_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, urllib.parse.quote(parsed.path), parsed.query, "")
    )


def _snapshot_row(snapshot: dict[str, object], row_id: int) -> list[object]:
    matches = [
        row
        for row in snapshot["data"]
        if isinstance(row, list) and f"/getfile.php/{row_id}/" in " ".join(map(str, row))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one snapshot row for {row_id}, found {len(matches)}")
    return matches[0]


def _validate_snapshot(config: dict[str, object]) -> None:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    path = ROOT / str(bindings["p240_snapshot_path"])
    if path.stat().st_size != int(bindings["p240_snapshot_bytes"]):
        raise RuntimeError("P240 snapshot byte mismatch")
    if _sha256_file(path) != bindings["p240_snapshot_sha256"]:
        raise RuntimeError("P240 snapshot SHA mismatch")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    for item in config["rows"]:
        assert isinstance(item, dict)
        source = " ".join(map(str, _snapshot_row(snapshot, int(item["id"]))))
        if "creativecommons.org/publicdomain/zero/1.0/" not in source:
            raise RuntimeError(f"CC0 marker missing for {item['source_id']}")
        if str(item["source_sha256"]) not in source:
            raise RuntimeError(f"snapshot source SHA missing for {item['source_id']}")
        if str(item["url"]) not in source:
            raise RuntimeError(f"snapshot URL missing for {item['source_id']}")


def acquire(config_path: Path, output: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_snapshot(config)
    root = ROOT / str(config["destination_root"])
    root.mkdir(parents=True, exist_ok=True)
    total_network = 0
    rows: list[dict[str, object]] = []
    for item in config["rows"]:
        assert isinstance(item, dict)
        destination = root / str(item["filename"])
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        downloaded = 0
        if not destination.exists():
            request = urllib.request.Request(
                _encoded_url(str(item["url"])),
                headers={"User-Agent": "neuro-film-p245/1.0"},
            )
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                    "xb"
                ) as stream:
                    final = urllib.parse.urlsplit(response.geturl())
                    if final.scheme != "https" or final.hostname != config["allowed_host"]:
                        raise RuntimeError("download redirected outside the frozen host")
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        total_network += len(chunk)
                        if total_network > int(config["network_body_ceiling_bytes"]):
                            raise RuntimeError("network body ceiling exceeded")
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != item["source_sha256"]:
                    raise RuntimeError(f"download SHA mismatch: {item['source_id']}")
                partial.replace(destination)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        if _sha256_file(destination) != item["source_sha256"]:
            raise RuntimeError(f"retained source SHA mismatch: {item['source_id']}")
        rows.append(
            {
                "bytes": destination.stat().st_size,
                "downloaded_bytes": downloaded,
                "logical_path": destination.relative_to(ROOT).as_posix(),
                "source_id": item["source_id"],
                "source_sha256": item["source_sha256"],
            }
        )
    report = {
        "experiment_id": "P245",
        "mode": "acquisition",
        "network_body_bytes": total_network,
        "partial_residue_count": len(list(root.glob("*.partial"))),
        "rows": sorted(rows, key=lambda item: str(item["source_id"])),
        "schema": "neuro-film.p245-dng-profile-real-file-acquisition.v1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    return report


def _profile_facts(path: Path) -> tuple[list[dict[str, object]], list[int]]:
    facts: list[dict[str, object]] = []
    dimensions: list[int] = []
    with tifffile.TiffFile(path) as document:
        for ifd_path, page in _walk_pages(document.pages):
            for code, name in raster._PROFILE_HUESATMAP_TAGS.items():
                if code not in page.tags:
                    continue
                tag = page.tags[code]
                facts.append(
                    {
                        "code": code,
                        "count": int(tag.count),
                        "ifd_path": ifd_path,
                        "name": name,
                        "tiff_type": tag.dtype.name,
                    }
                )
                if code == 50937:
                    dimensions = [
                        int(item) for item in np.asarray(tag.value).reshape(-1)
                    ]
    facts.sort(key=lambda item: (int(item["code"]), str(item["ifd_path"])))
    return facts, dimensions


def formal(config_path: Path, output: Path, reverse: bool) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_snapshot(config)
    root = ROOT / str(config["destination_root"])
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    results: list[dict[str, object]] = []
    for item in rows:
        assert isinstance(item, dict)
        source = root / str(item["filename"])
        before_sha = _sha256_file(source)
        if before_sha != item["source_sha256"]:
            raise RuntimeError(f"source SHA mismatch: {item['source_id']}")
        facts, dimensions = _profile_facts(source)
        decode_calls = 0

        def decode(_path: Path):
            nonlocal decode_calls
            decode_calls += 1
            raise AssertionError("camera decode must not run")

        with patch.object(raster, "_decode_camera_linear_dng", decode):
            try:
                raster.load_dng_forward_working_image(
                    source,
                    expected_source_bytes=source.stat().st_size,
                    expected_source_sha256=before_sha,
                )
            except raster.DngForwardRasterError as exc:
                message = str(exc)
            else:
                raise AssertionError("real profile-bearing DNG bypassed the guard")
        actual_codes = [int(fact["code"]) for fact in facts]
        results.append(
            {
                "actual_guarded_tags": facts,
                "decode_calls": decode_calls,
                "dimensions": dimensions,
                "dimensions_match": dimensions == item["expected_dimensions"],
                "message": message,
                "message_names_all_actual_tags": all(
                    raster._PROFILE_HUESATMAP_TAGS[code] in message for code in actual_codes
                ),
                "required_tags_present": all(
                    int(code) in actual_codes for code in item["expected_required_tags"]
                ),
                "source_bytes": source.stat().st_size,
                "source_id": item["source_id"],
                "source_sha256": before_sha,
                "source_unchanged": _sha256_file(source) == before_sha,
            }
        )
    results.sort(key=lambda item: str(item["source_id"]))
    gates = {
        "expected_dimensions": all(row["dimensions_match"] for row in results),
        "exact_source_hashes": all(row["source_unchanged"] for row in results),
        "rejection_names_all_actual_guarded_tags": all(
            row["message_names_all_actual_tags"] for row in results
        ),
        "required_real_tags": all(row["required_tags_present"] for row in results),
        "required_rows": len(results) == int(config["gates"]["required_rows"]),
        "snapshot_rights_rows": True,
        "zero_camera_decode_calls": all(row["decode_calls"] == 0 for row in results),
        "zero_partial_residue": not list(root.glob("*.partial")),
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_REAL_DNG_PROFILE_GUARD_CONFIRMATION"
        if all(gates.values())
        else "FAIL_CLOSED_REAL_DNG_PROFILE_GUARD_CONFIRMATION",
        "execution_commit": _git_head(),
        "experiment_id": "P245",
        "gates": gates,
        "rows": results,
        "schema": "neuro-film.p245-dng-profile-real-file-guard-result.v1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("acquire", "formal"), required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.mode == "acquire":
        if args.reverse:
            raise ValueError("acquisition does not accept --reverse")
        acquire(args.config, args.output)
    else:
        formal(args.config, args.output, args.reverse)


if __name__ == "__main__":
    main()

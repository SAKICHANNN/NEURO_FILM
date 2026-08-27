"""Run the frozen P258 real-file ProfileGainTableMap refusal audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.preprocess.dng_forward_raster as raster
from src.preprocess.dng_metadata import _walk_pages, canonical_json_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _bound(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
    }


def _verify_file_binding(bindings: dict[str, object], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    return (
        path.is_file()
        and path.stat().st_size == int(bindings[f"{prefix}_bytes"])
        and _sha256(path) == str(bindings[f"{prefix}_sha256"])
    )


def _verify_parent_bindings(config: dict[str, object]) -> dict[str, bool]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    return {
        prefix: _verify_file_binding(bindings, prefix)
        for prefix in (
            "contract",
            "implementation",
            "p257_config",
            "p257_evidence",
            "p7h_config",
            "runner",
            "test",
        )
    }


def _verify_p7h_source_row(config: dict[str, object]) -> dict[str, bool]:
    bindings = config["bindings"]
    source = config["source"]
    assert isinstance(bindings, dict)
    assert isinstance(source, dict)
    p7h = json.loads(
        (ROOT / str(bindings["p7h_config_path"])).read_text(encoding="utf-8")
    )
    matches = [
        row
        for row in p7h["candidates"]
        if int(row["repository_id"]) == int(source["repository_id"])
    ]
    if len(matches) != 1:
        raise RuntimeError("P7H source row is not unique")
    row = matches[0]
    return {
        "cc0_binding": "CC0" in p7h["source"]["declared_license"],
        "identity": row["id"] == source["source_id"],
        "path": row["path"] == source["logical_path"],
        "sha256": row["sha256"] == source["sha256"],
        "url": row["url"] == source["url"],
    }


def _sdk_tag_codes(config: dict[str, object]) -> dict[int, str]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    archive = ROOT / str(bindings["sdk_zip_path"])
    if archive.stat().st_size != int(bindings["sdk_zip_bytes"]):
        raise RuntimeError("SDK ZIP byte count mismatch")
    if _sha256(archive) != bindings["sdk_zip_sha256"]:
        raise RuntimeError("SDK ZIP SHA-256 mismatch")
    with ZipFile(archive) as bundle:
        text = bundle.read(str(bindings["sdk_tag_code_member"])).decode("utf-8")
    result: dict[int, str] = {}
    for item in config["guarded_tags"]:
        assert isinstance(item, dict)
        code = int(item["code"])
        name = str(item["name"])
        matching = [
            line
            for line in text.splitlines()
            if line.strip().split(maxsplit=1)[0:1] == [f"tc{name}"] and "=" in line
        ]
        if len(matching) != 1 or f"= {code}" not in matching[0]:
            raise RuntimeError(f"SDK tag-code authority mismatch for {name}")
        result[code] = name
    return result


def _real_guarded_tags(
    path: Path, guarded: dict[int, str], order: str
) -> list[dict[str, object]]:
    codes = sorted(guarded, reverse=order == "reverse")
    found: list[dict[str, object]] = []
    with tifffile.TiffFile(path) as document:
        pages = list(_walk_pages(document.pages))
        if order == "reverse":
            pages.reverse()
        for ifd_path, page in pages:
            for code in codes:
                if code not in page.tags:
                    continue
                tag = page.tags[code]
                found.append(
                    {
                        "code": code,
                        "count": int(tag.count),
                        "ifd_path": ifd_path,
                        "name": guarded[code],
                        "tiff_type": tag.dtype.name,
                    }
                )
    return sorted(found, key=lambda item: (int(item["code"]), str(item["ifd_path"])))


def build_report(config_path: Path, order: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_config = config["source"]
    assert isinstance(source_config, dict)
    guarded = _sdk_tag_codes(config)
    if guarded != raster._PROFILE_GAIN_TABLE_MAP_TAGS:
        raise RuntimeError("SDK and implementation guarded tag maps differ")

    parent_bindings = _verify_parent_bindings(config)
    p7h_source_row = _verify_p7h_source_row(config)
    source = ROOT / str(source_config["logical_path"])
    before_sha = _sha256(source)
    if source.stat().st_size != int(source_config["bytes"]):
        raise RuntimeError("source byte count mismatch")
    if before_sha != source_config["sha256"]:
        raise RuntimeError("source SHA-256 mismatch")

    actual_tags = _real_guarded_tags(source, guarded, order)
    actual_codes = sorted({int(item["code"]) for item in actual_tags})
    expected_present = sorted(
        int(code) for code in source_config["expected_present_guarded_tags"]
    )
    expected_absent = sorted(
        int(code) for code in source_config["expected_absent_guarded_tags"]
    )
    decode_calls = 0

    def fail_decode(_path: Path):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("camera-raster decode must not run")

    with patch.object(raster, "_decode_camera_linear_dng", fail_decode):
        try:
            raster.load_dng_forward_working_image(
                source,
                expected_source_bytes=int(source_config["bytes"]),
                expected_source_sha256=str(source_config["sha256"]),
            )
        except raster.DngForwardRasterError as exc:
            diagnostic = str(exc)
        else:
            raise AssertionError("real ProfileGainTableMap DNG bypassed P257 guard")

    source_unchanged = _sha256(source) == before_sha
    diagnostic_names_all = all(
        guarded[int(item["code"])] in diagnostic for item in actual_tags
    )
    diagnostic_ifds_all = all(
        f"@{item['ifd_path']}" in diagnostic for item in actual_tags
    )
    filesystem_writes = {
        "copied_dng_files": 0,
        "metadata_extract_files": 0,
        "network_requests": 0,
        "partial_files": 0,
        "pixel_artifacts": 0,
        "preview_artifacts": 0,
    }
    gates = {
        "absent_v2_tag": not any(code in actual_codes for code in expected_absent),
        "exact_real_guarded_tag_set": actual_codes == expected_present,
        "parent_bindings_exact": all(parent_bindings.values()),
        "p7h_source_row_exact": all(p7h_source_row.values()),
        "rejection_names_all_actual_tags_and_ifds": diagnostic_names_all
        and diagnostic_ifds_all,
        "sdk_tag_codes_exact": guarded
        == {int(item["code"]): str(item["name"]) for item in config["guarded_tags"]},
        "source_immutable": source_unchanged,
        "zero_camera_decode_calls": decode_calls == 0,
        "zero_copied_or_pixel_artifacts": all(
            count == 0 for key, count in filesystem_writes.items() if key != "network_requests"
        ),
        "zero_network_requests": filesystem_writes["network_requests"] == 0,
    }
    passed = all(gates.values())
    return {
        "bindings": {
            "config": _bound(config_path),
            "contract": _bound(ROOT / str(config["bindings"]["contract_path"])),
            "implementation": _bound(
                ROOT / str(config["bindings"]["implementation_path"])
            ),
            "runner": _bound(Path(__file__).resolve()),
            "source": _bound(source),
            "test": _bound(ROOT / str(config["bindings"]["test_path"])),
        },
        "claim_ceiling": config["claim_ceiling"],
        "decision": (
            "PASS_PRIVATE_REAL_DNG_PROFILE_GAIN_TABLE_MAP_GUARD_CONFIRMATION"
            if passed
            else "FAIL_CLOSED_REAL_DNG_PROFILE_GAIN_TABLE_MAP_GUARD_CONFIRMATION"
        ),
        "diagnostic": diagnostic,
        "execution_commit": _git_head(),
        "experiment_id": "P258",
        "filesystem_writes": filesystem_writes,
        "gate_results": gates,
        "parent_bindings": parent_bindings,
        "p7h_source_row": p7h_source_row,
        "real_guarded_tags": actual_tags,
        "schema": "neuro-film.p258-dng-profile-gain-table-map-real-file-guard-result.v1",
        "source_unchanged": source_unchanged,
        "zero_camera_decode_calls": decode_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))


if __name__ == "__main__":
    main()

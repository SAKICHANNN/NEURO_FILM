"""Run the frozen P257 DNG ProfileGainTableMap no-silent-drop audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

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


def _bound(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sdk_codes(
    config: dict[str, object], key: str
) -> dict[int, str]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    archive = Path(str(bindings["sdk_zip_path"]))
    if archive.stat().st_size != int(bindings["sdk_zip_bytes"]):
        raise RuntimeError("SDK ZIP byte count mismatch")
    if _sha256_file(archive) != bindings["sdk_zip_sha256"]:
        raise RuntimeError("SDK ZIP SHA-256 mismatch")
    member = str(bindings["sdk_tag_code_member"])
    with ZipFile(archive) as bundle:
        text = bundle.read(member).decode("utf-8")
    result: dict[int, str] = {}
    for item in config[key]:
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


def _real_profile_tags(
    path: Path, guarded: dict[int, str]
) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    with tifffile.TiffFile(path) as document:
        for ifd_path, page in _walk_pages(document.pages):
            for code, name in guarded.items():
                if code in page.tags:
                    found.append({"code": code, "ifd_path": ifd_path, "name": name})
    return sorted(found, key=lambda item: (int(item["code"]), str(item["ifd_path"])))


def _fake_document(codes: tuple[int, ...]):
    page = SimpleNamespace(tags={code: object() for code in codes}, pages=None)

    class Document:
        def __init__(self) -> None:
            self.pages = [page]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    return Document()


def _loader_rejection_control(codes: tuple[int, ...]) -> dict[str, object]:
    decode_calls = 0

    def decode(_path: Path):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("camera decode must not run")

    with tempfile.TemporaryDirectory(prefix="p257-guard-") as directory:
        source = Path(directory) / "profile.dng"
        source.write_bytes(b"P257 metadata-only rejection control")
        with (
            patch.object(raster.tifffile, "TiffFile", lambda _path: _fake_document(codes)),
            patch.object(raster, "_decode_camera_linear_dng", decode),
        ):
            try:
                raster.load_dng_forward_working_image(source)
            except raster.DngForwardRasterError as exc:
                message = str(exc)
            else:
                raise AssertionError("guard accepted an unsupported profile tag")
    return {"codes": list(codes), "decode_calls": decode_calls, "message": message}


def _direct_rejection_control(
    codes: tuple[int, ...], *, huesat: bool
) -> dict[str, object]:
    pages = [("0", _fake_document(codes).pages[0])]
    guard = (
        raster._guard_unsupported_profile_huesatmap
        if huesat
        else raster._guard_unsupported_profile_gain_table_map
    )
    try:
        guard(pages)
    except raster.DngForwardRasterError as exc:
        message = str(exc)
    else:
        raise AssertionError("direct guard accepted an unsupported profile tag")
    return {"codes": list(codes), "message": message}


def _verify_parent_bindings(config: dict[str, object]) -> dict[str, bool]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    keys = (
        "contract",
        "p244_config",
        "p244_evidence",
        "p98_config",
        "p98_evidence",
    )
    return {
        key: _sha256_file(Path(str(bindings[f"{key}_path"])))
        == bindings[f"{key}_sha256"]
        for key in keys
    }


def build_report(config_path: Path, order: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guarded = _sdk_codes(config, "guarded_tags")
    huesat = _sdk_codes(config, "preserved_huesat_tags")
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    results: list[dict[str, object]] = []
    for row in rows:
        assert isinstance(row, dict)
        source = Path(str(row["logical_path"]))
        before_sha = _sha256_file(source)
        if source.stat().st_size != int(row["source_bytes"]):
            raise RuntimeError(f"source byte mismatch: {row['source_id']}")
        if before_sha != row["source_sha256"]:
            raise RuntimeError(f"source SHA mismatch: {row['source_id']}")
        present = _real_profile_tags(source, guarded)
        working = raster.load_dng_forward_working_image(
            source,
            expected_source_bytes=int(row["source_bytes"]),
            expected_source_sha256=str(row["source_sha256"]),
        )
        output_sha = hashlib.sha256(working.pixels.tobytes(order="C")).hexdigest()
        results.append(
            {
                "profile_gain_table_map_tags": present,
                "source_id": row["source_id"],
                "source_sha256": before_sha,
                "source_unchanged": _sha256_file(source) == before_sha,
                "working": {
                    "dtype": str(working.pixels.dtype),
                    "finite": bool(np.isfinite(working.pixels).all()),
                    "pixel_sha256": output_sha,
                    "pixel_sha256_matches_p98": output_sha
                    == row["working_float32_sha256"],
                    "source_transfer_state": working.source_transfer_state,
                    "transfer_state": working.transfer_state,
                    "warning_codes": [warning.code for warning in working.warnings],
                    "working_space": working.working_space,
                },
            }
        )
    results.sort(key=lambda item: str(item["source_id"]))

    controls = [_loader_rejection_control((code,)) for code in sorted(guarded)]
    controls.append(_loader_rejection_control(tuple(sorted(guarded, reverse=True))))
    huesat_controls = [
        _direct_rejection_control((code,), huesat=True) for code in sorted(huesat)
    ]
    parent_bindings = _verify_parent_bindings(config)
    expected_warning_codes = config["expected_warning_codes"]
    multi_message = str(controls[-1]["message"])
    sorted_codes = sorted(guarded)
    gate_results = {
        "all_real_rows_tag_free": all(
            not row["profile_gain_table_map_tags"] for row in results
        ),
        "exact_p98_working_hashes": all(
            bool(row["working"]["pixel_sha256_matches_p98"]) for row in results
        ),
        "exact_working_contract": all(
            row["working"]["dtype"] == "float32"
            and row["working"]["finite"]
            and row["working"]["working_space"] == "linear_rec2020"
            and row["working"]["transfer_state"] == "scene_linear"
            and row["working"]["source_transfer_state"] == "scene_linear"
            and row["working"]["warning_codes"] == expected_warning_codes
            for row in results
        ),
        "multi_tag_sorted_diagnostic": all(
            multi_message.find(guarded[code]) < multi_message.find(guarded[next_code])
            for code, next_code in pairwise(sorted_codes)
        ),
        "p244_huesat_guard_unchanged": len(huesat_controls)
        == int(config["gates"]["required_preserved_huesat_rejections"])
        and all(
            huesat[control["codes"][0]] in control["message"]
            for control in huesat_controls
        ),
        "parent_bindings_exact": all(parent_bindings.values()),
        "row_count": len(results) == int(config["gates"]["required_rows"]),
        "sdk_tag_codes_exact": guarded
        == {int(item["code"]): str(item["name"]) for item in config["guarded_tags"]},
        "single_tag_rejections": len(controls) - 1
        == int(config["gates"]["required_single_tag_rejections"])
        and all(
            guarded[control["codes"][0]] in control["message"]
            for control in controls[:-1]
        ),
        "source_hashes_unchanged": all(row["source_unchanged"] for row in results),
        "zero_decode_calls_on_rejection": all(
            control["decode_calls"] == 0 for control in controls
        ),
    }
    passed = all(gate_results.values())
    return {
        "bindings": {
            "config": _bound(config_path),
            "contract": _bound(Path(config["bindings"]["contract_path"])),
            "implementation": _bound(Path("src/preprocess/dng_forward_raster.py")),
            "runner": _bound(Path(__file__).resolve().relative_to(Path.cwd())),
            "test": _bound(
                Path("tests/test_p257_dng_profile_gain_table_map_ingress_guard.py")
            ),
        },
        "claim_ceiling": config["claim_ceiling"],
        "decision": (
            "PASS_PRIVATE_DNG_PROFILE_GAIN_TABLE_MAP_INGRESS_GUARD"
            if passed
            else "FAIL_CLOSED_DNG_PROFILE_GAIN_TABLE_MAP_INGRESS_GUARD"
        ),
        "execution_commit": _git_head(),
        "experiment_id": "P257",
        "gate_results": gate_results,
        "guarded_tags": [
            {"code": code, "name": guarded[code]} for code in sorted(guarded)
        ],
        "parent_bindings": parent_bindings,
        "preserved_huesat_rejection_controls": huesat_controls,
        "rejection_controls": controls,
        "rows": results,
        "schema": "neuro-film.p257-dng-profile-gain-table-map-ingress-guard-result.v1",
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

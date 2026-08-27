"""Run the frozen P262 DNG ProfileLookTable no-silent-drop audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_p257_dng_profile_gain_table_map_ingress_guard import (
    _bound,
    _fake_document,
    _git_head,
    _sdk_codes,
    _sha256_file,
)

import src.preprocess.dng_forward_raster as raster
from src.preprocess.dng_metadata import _walk_pages, canonical_json_bytes


def _profile_tags(path: Path, guarded: dict[int, str]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    with tifffile.TiffFile(path) as document:
        for ifd_path, page in _walk_pages(document.pages):
            for code, name in guarded.items():
                if code in page.tags:
                    found.append({"code": code, "ifd_path": ifd_path, "name": name})
    return sorted(found, key=lambda item: (int(item["code"]), str(item["ifd_path"])))


def _loader_control(codes: tuple[int, ...], *, real_path: Path | None = None):
    decode_calls = 0

    def decode(_path: Path):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("camera decode must not run")

    if real_path is not None:
        source = real_path
        context = patch.object(raster, "_decode_camera_linear_dng", decode)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="p262-guard-")
        source = Path(temporary.name) / "profile.dng"
        source.write_bytes(b"P262 metadata-only rejection control")
        context = patch.multiple(
            raster,
            _decode_camera_linear_dng=decode,
        )
    try:
        with context:
            if real_path is None:
                with patch.object(
                    raster.tifffile,
                    "TiffFile",
                    lambda _path: _fake_document(codes),
                ):
                    raster.load_dng_forward_working_image(source)
            else:
                raster.load_dng_forward_working_image(source)
    except raster.DngForwardRasterError as exc:
        message = str(exc)
    else:
        raise AssertionError("guard accepted ProfileLookTable")
    finally:
        if real_path is None:
            temporary.cleanup()
    return {"codes": list(codes), "decode_calls": decode_calls, "message": message}


def _preserved_controls() -> list[dict[str, object]]:
    controls = []
    for name, code, guard in (
        ("ProfileHueSatMap", 50937, raster._guard_unsupported_profile_huesatmap),
        (
            "ProfileGainTableMap",
            52525,
            raster._guard_unsupported_profile_gain_table_map,
        ),
        ("ProfileToneCurve", 50940, raster._guard_unsupported_profile_tone_curve),
    ):
        try:
            guard([("0", _fake_document((code,)).pages[0])])
        except raster.DngForwardRasterError as exc:
            controls.append({"code": code, "guard": name, "message": str(exc)})
        else:
            raise AssertionError(f"{name} guard drifted")
    return controls


def _verify_bindings(config: dict[str, object]) -> dict[str, bool]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    keys = (
        "contract",
        "implementation",
        "p98_config",
        "p98_evidence",
        "p244_evidence",
        "p257_evidence",
        "p261_evidence",
        "runner",
        "test",
    )
    return {
        key: _sha256_file(Path(str(bindings[f"{key}_path"])))
        == bindings[f"{key}_sha256"]
        for key in keys
    }


def build_report(config_path: Path, order: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guarded = _sdk_codes(config, "guarded_tags")
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    results = []
    for row in rows:
        source = Path(str(row["logical_path"]))
        before = _sha256_file(source)
        if (
            source.stat().st_size != int(row["source_bytes"])
            or before != row["source_sha256"]
        ):
            raise RuntimeError(f"source identity mismatch: {row['source_id']}")
        present = _profile_tags(source, guarded)
        working = raster.load_dng_forward_working_image(
            source,
            expected_source_bytes=int(row["source_bytes"]),
            expected_source_sha256=str(row["source_sha256"]),
        )
        pixel_sha = hashlib.sha256(working.pixels.tobytes(order="C")).hexdigest()
        results.append(
            {
                "look_table_tags": present,
                "source_id": row["source_id"],
                "source_unchanged": _sha256_file(source) == before,
                "working": {
                    "dtype": str(working.pixels.dtype),
                    "finite": bool(np.isfinite(working.pixels).all()),
                    "pixel_sha256": pixel_sha,
                    "pixel_sha256_matches_p98": pixel_sha
                    == row["working_float32_sha256"],
                    "source_transfer_state": working.source_transfer_state,
                    "transfer_state": working.transfer_state,
                    "warning_codes": [item.code for item in working.warnings],
                    "working_space": working.working_space,
                },
            }
        )
    results.sort(key=lambda item: str(item["source_id"]))

    real = config["real_source"]
    real_path = Path(str(real["logical_path"]))
    real_before = _sha256_file(real_path)
    if (
        real_path.stat().st_size != int(real["source_bytes"])
        or real_before != real["source_sha256"]
    ):
        raise RuntimeError("real source identity mismatch")
    real_tags = _profile_tags(real_path, guarded)
    synthetic = [_loader_control((code,)) for code in sorted(guarded)]
    synthetic.append(_loader_control(tuple(sorted(guarded, reverse=True))))
    preserved = _preserved_controls()
    bindings = _verify_bindings(config)
    expected_warnings = config["expected_warning_codes"]
    multi_message = synthetic[-1]["message"]
    gates = {
        "all_p98_rows_tag_free": all(not row["look_table_tags"] for row in results),
        "exact_p98_outputs": all(
            row["working"]["pixel_sha256_matches_p98"] for row in results
        ),
        "exact_working_contract": all(
            row["working"]["dtype"] == "float32"
            and row["working"]["finite"]
            and row["working"]["working_space"] == "linear_rec2020"
            and row["working"]["transfer_state"] == "scene_linear"
            and row["working"]["source_transfer_state"] == "scene_linear"
            and row["working"]["warning_codes"] == expected_warnings
            for row in results
        ),
        "parent_bindings_exact": all(bindings.values()),
        "preserved_guards_exact": all(
            item["guard"] in item["message"] for item in preserved
        ),
        "real_source_exact": _sha256_file(real_path) == real_before,
        "real_source_tags_exact": [item["code"] for item in real_tags]
        == [50981, 50982],
        "real_source_metadata_confirmed": bool(real_tags),
        "sdk_tag_codes_exact": guarded
        == {
            50981: "ProfileLookTableDims",
            50982: "ProfileLookTableData",
            51108: "ProfileLookTableEncoding",
        },
        "single_tag_rejections": all(
            guarded[item["codes"][0]] in item["message"] and item["decode_calls"] == 0
            for item in synthetic[:-1]
        ),
        "sorted_multi_tag_diagnostic": all(
            multi_message.find(guarded[code]) < multi_message.find(guarded[next_code])
            for code, next_code in pairwise(sorted(guarded))
        ),
        "source_hashes_unchanged": all(row["source_unchanged"] for row in results),
    }
    passed = all(gates.values())
    return {
        "bindings": {
            "config": _bound(config_path),
            "contract": _bound(Path(config["bindings"]["contract_path"])),
            "implementation": _bound(Path("src/preprocess/dng_forward_raster.py")),
            "runner": _bound(Path(__file__).resolve().relative_to(Path.cwd())),
            "test": _bound(
                Path("tests/test_p262_dng_profile_look_table_ingress_guard.py")
            ),
        },
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_PROFILE_LOOK_TABLE_INGRESS_GUARD"
        if passed
        else "FAIL_CLOSED_DNG_PROFILE_LOOK_TABLE_INGRESS_GUARD",
        "execution_commit": _git_head(),
        "experiment_id": "P262",
        "gate_results": gates,
        "guarded_tags": [
            {"code": code, "name": guarded[code]} for code in sorted(guarded)
        ],
        "parent_bindings": bindings,
        "preserved_guard_controls": preserved,
        "real_source": {
            "identity": real,
            "tags": real_tags,
        },
        "rows": results,
        "schema": "neuro-film.p262-dng-profile-look-table-ingress-guard-result.v1",
        "synthetic_rejection_controls": synthetic,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(build_report(args.config, args.order)))


if __name__ == "__main__":
    main()

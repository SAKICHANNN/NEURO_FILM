"""Formal U1.2D high-precision TIFF orientation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import imagecodecs
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_working_image
from src.preprocess.output_encode import srgb_icc_profile
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020

SCHEMA = "kmcfm.u1-2d-high-precision-tiff-orientation-result.v1"

_PIL_ORIENTATIONS = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _fixture() -> np.ndarray:
    positions = np.arange(15, dtype=np.uint16).reshape(3, 5)
    return np.stack(
        (
            positions * 101 + 17,
            positions * 211 + 1003,
            positions * 307 + 5001,
        ),
        axis=2,
    )


def _pillow_oracle(array: np.ndarray, orientation: int) -> np.ndarray:
    operation = _PIL_ORIENTATIONS[orientation]
    if operation is None:
        return array.copy()
    return np.stack(
        [
            np.asarray(Image.fromarray(array[..., channel]).transpose(operation))
            for channel in range(3)
        ],
        axis=2,
    ).astype(np.uint16, copy=False)


def _srgb_linear(array: np.ndarray) -> np.ndarray:
    encoded = array.astype(np.float32) / 65535.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _prophoto_profile() -> bytes:
    profile = bytearray(
        imagecodecs.cms_profile(
            "rgb",
            whitepoint=(0.3457, 0.3585, 1.0),
            primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001),
            gamma=1.8,
        )
    )
    profile[24:36] = bytes.fromhex("07d001010000000000000000")
    profile[84:100] = b"\x00" * 16
    return bytes(profile)


def _write_tiff(
    path: Path,
    array: np.ndarray,
    orientation: int,
    *,
    profile: bytes | None = None,
) -> None:
    tags: list[tuple[int, str, int, object, bool]] = [
        (274, "H", 1, orientation, False)
    ]
    if profile is not None:
        tags.append((34675, "B", len(profile), profile, False))
    tifffile.imwrite(
        path,
        array,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=tags,
    )


def _rejection_controls(scratch: Path, source: np.ndarray) -> dict[str, bool]:
    controls: dict[str, bool] = {}

    invalid = scratch / "invalid_orientation.tiff"
    _write_tiff(invalid, source, 9, profile=srgb_icc_profile())
    try:
        load_working_image(invalid)
    except ValueError as exc:
        controls["invalid_orientation_rejects"] = (
            str(exc) == "invalid TIFF Orientation value: 9"
        )
    else:
        controls["invalid_orientation_rejects"] = False

    unknown_icc = scratch / "unknown_icc.tiff"
    _write_tiff(unknown_icc, source, 6, profile=b"not-a-supported-icc-profile")
    try:
        load_working_image(unknown_icc)
    except ValueError as exc:
        controls["unknown_icc_rejects"] = "ICC conversion is not implemented" in str(
            exc
        )
    else:
        controls["unknown_icc_rejects"] = False

    multipage = scratch / "multipage.tiff"
    with tifffile.TiffWriter(multipage) as writer:
        writer.write(source, photometric="rgb", metadata=None)
        writer.write(source, photometric="rgb", metadata=None)
    try:
        load_working_image(multipage)
    except ValueError as exc:
        controls["multipage_rejects"] = "frame-zero fallback" in str(exc)
    else:
        controls["multipage_rejects"] = False

    rgba = scratch / "rgba.tiff"
    rgba_source = np.concatenate(
        (source, np.full((*source.shape[:2], 1), 65535, dtype=np.uint16)), axis=2
    )
    tifffile.imwrite(
        rgba,
        rgba_source,
        photometric="rgb",
        planarconfig="contig",
        extrasamples="unassalpha",
        metadata=None,
        extratags=[(274, "H", 1, 6, False)],
    )
    try:
        load_working_image(rgba)
    except ValueError as exc:
        controls["alpha_rejects"] = "requires contiguous uint16 RGB" in str(exc)
    else:
        controls["alpha_rejects"] = False

    return controls


def run(config_path: Path, output_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    source = _fixture()
    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    scratch_path: Path | None = None
    rows: list[dict[str, Any]] = []
    controls: dict[str, bool] = {}
    prophoto: dict[str, Any] = {}
    renderer: dict[str, Any] = {}
    try:
        scratch_path = Path(
            tempfile.mkdtemp(prefix="u1_2d_orientation_", dir=tmp_root)
        )
        orientations = list(range(1, 9))
        if reverse:
            orientations.reverse()
        for orientation in orientations:
            path = scratch_path / f"orientation_{orientation}.tiff"
            _write_tiff(path, source, orientation, profile=srgb_icc_profile())
            before = _sha256(path)
            working = load_working_image(path)
            after = _sha256(path)
            expected_samples = _pillow_oracle(source, orientation)
            expected = _srgb_linear(expected_samples)
            rows.append(
                {
                    "orientation": orientation,
                    "source_sha256": before,
                    "source_immutable": before == after,
                    "dimensions": [working.pixels.shape[1], working.pixels.shape[0]],
                    "pixel_sha256": _sha256_bytes(working.pixels.tobytes()),
                    "expected_pixel_sha256": _sha256_bytes(expected.tobytes()),
                    "pixel_exact": working.pixels.tobytes() == expected.tobytes(),
                    "sample_multiset_exact": np.array_equal(
                        np.sort(expected_samples.reshape(-1)), np.sort(source.reshape(-1))
                    ),
                    "bit_depth_in": working.bit_depth_in,
                    "orientation_applied": working.orientation_applied,
                }
            )

        rows.sort(key=lambda row: row["orientation"])
        profile = _prophoto_profile()
        prophoto_path = scratch_path / "prophoto_orientation_6.tiff"
        _write_tiff(prophoto_path, source, 6, profile=profile)
        prophoto_working = load_working_image(prophoto_path)
        prophoto_expected = decode_prophoto_rgb16_to_linear_rec2020(
            _pillow_oracle(source, 6), profile
        )
        prophoto = {
            "profile_sha256": _sha256_bytes(profile),
            "working_space": prophoto_working.working_space,
            "dimensions": [
                prophoto_working.pixels.shape[1],
                prophoto_working.pixels.shape[0],
            ],
            "pixel_sha256": _sha256_bytes(prophoto_working.pixels.tobytes()),
            "expected_pixel_sha256": _sha256_bytes(prophoto_expected.tobytes()),
            "pixel_exact": (
                prophoto_working.pixels.tobytes() == prophoto_expected.tobytes()
            ),
        }
        controls = _rejection_controls(scratch_path, source)

        render_input = scratch_path / "render_orientation_6.tiff"
        render_output = scratch_path / "render_orientation_6.png"
        _write_tiff(render_input, source, 6, profile=srgb_icc_profile())
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_film.py"),
                str(render_input),
                "--output",
                str(render_output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if render_output.is_file():
            with Image.open(render_output) as image:
                output_dimensions = list(image.size)
            output_sha256 = _sha256(render_output)
        else:
            output_dimensions = []
            output_sha256 = ""
        renderer = {
            "returncode": completed.returncode,
            "output_dimensions": output_dimensions,
            "output_sha256": output_sha256,
            "stderr_empty": not completed.stderr,
        }
    finally:
        if scratch_path is not None:
            shutil.rmtree(scratch_path, ignore_errors=False)

    all_orientations_exact = len(rows) == 8 and all(
        row["pixel_exact"] and row["orientation_applied"] for row in rows
    )
    gates = {
        "all_orientations_exact": all_orientations_exact,
        "sample_multiset_exact": all(row["sample_multiset_exact"] for row in rows),
        "srgb_path_exact": all(
            row["bit_depth_in"] == 16 and row["source_immutable"] for row in rows
        ),
        "prophoto_path_exact": (
            prophoto["working_space"] == "linear_rec2020"
            and prophoto["pixel_exact"]
            and prophoto["dimensions"] == [3, 5]
        ),
        "identity_legacy_exact": rows[0]["pixel_exact"],
        "invalid_orientation_rejects": controls["invalid_orientation_rejects"],
        "existing_safety_gates_unchanged": all(controls.values()),
        "renderer_geometry_exact": (
            renderer["returncode"] == 0
            and renderer["output_dimensions"] == [3, 5]
            and renderer["stderr_empty"]
        ),
        "scratch_residue_zero": scratch_path is not None and not scratch_path.exists(),
    }
    report = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_RGB16_TIFF_ORIENTATION_INGRESS"
            if all(gates.values())
            else "FAIL_CLOSED_RGB16_TIFF_ORIENTATION_INGRESS"
        ),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "contract_sha256": _sha256(ROOT / bindings["contract_path"]),
            "core_sha256": _sha256(ROOT / bindings["core_path"]),
            "test_sha256": _sha256(ROOT / bindings["test_path"]),
            "runner_sha256": _sha256(ROOT / bindings["runner_path"]),
        },
        "orientation_rows": rows,
        "prophoto": prophoto,
        "rejection_controls": controls,
        "renderer": renderer,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u1_2d_high_precision_tiff_orientation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, reverse=args.reverse)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

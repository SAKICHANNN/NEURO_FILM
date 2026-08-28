"""Formal U1.2E strict Adobe-RGB-compatible RGB16 TIFF audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import imagecodecs
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import load_working_image
from src.preprocess.adobe_rgb_icc import (
    AdobeRGBICCError,
    adobe_rgb_icc_facts,
)
from src.preprocess.output_encode import srgb_icc_profile
from src.preprocess.prophoto_icc import (
    d50_xyz_to_linear_rec2020,
    decode_prophoto_rgb16_to_linear_rec2020,
)

SCHEMA = "kmcfm.u1-2e-high-precision-adobe-rgb-tiff-result.v1"


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


def _generated_profile(*, gamma: float = 563.0 / 256.0) -> bytes:
    profile = bytearray(
        imagecodecs.cms_profile(
            "rgb",
            whitepoint=(0.3127, 0.3290, 1.0),
            primaries=(0.64, 0.33, 0.21, 0.71, 0.15, 0.06),
            gamma=gamma,
        )
    )
    profile[24:36] = bytes.fromhex("07d001010000000000000000")
    profile[84:100] = b"\x00" * 16
    return bytes(profile)


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


def _samples() -> np.ndarray:
    return np.asarray(
        [
            [[0, 32768, 65535], [65535, 0, 4096], [1234, 23456, 45678]],
            [[8192, 16384, 32768], [60000, 40000, 20000], [1, 2, 3]],
        ],
        dtype=np.uint16,
    )


def _write(path: Path, samples: np.ndarray, profile: bytes, orientation: int = 1) -> None:
    tifffile.imwrite(
        path,
        samples,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[
            (274, "H", 1, orientation, False),
            (34675, "B", len(profile), profile, False),
        ],
    )


def _oracle(samples: np.ndarray, profile: bytes) -> np.ndarray:
    facts = adobe_rgb_icc_facts(profile)
    encoded = samples.astype(np.float64) / 65535.0
    xyz = np.power(encoded, float(facts["gamma"])) @ np.asarray(
        facts["matrix"], dtype=np.float64
    ).T
    return np.asarray(d50_xyz_to_linear_rec2020(xyz), dtype=np.float32)


def _mutate_tag(profile: bytes, signature: bytes, delta: int) -> bytes:
    mutated = bytearray(profile)
    count = int.from_bytes(mutated[128:132], "big")
    for index in range(count):
        cursor = 132 + index * 12
        if mutated[cursor : cursor + 4] == signature:
            offset = int.from_bytes(mutated[cursor + 4 : cursor + 8], "big")
            value = int.from_bytes(mutated[offset + 12 : offset + 16], "big", signed=True)
            mutated[offset + 12 : offset + 16] = (value + delta).to_bytes(
                4, "big", signed=True
            )
            return bytes(mutated)
    raise RuntimeError(f"missing ICC tag {signature!r}")


def _rejects(profile: bytes) -> bool:
    try:
        adobe_rgb_icc_facts(profile)
    except AdobeRGBICCError:
        return True
    return False


def run(config_path: Path, output_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_profile = ROOT / config["test_profile"]["path"]
    profiles = [
        ("generated_d50", _generated_profile()),
        ("clayrgb_v2_d65_media_white", test_profile.read_bytes()),
    ]
    if reverse:
        profiles.reverse()
    source = _samples()
    scratch: Path | None = None
    rows: list[dict[str, Any]] = []
    legacy: dict[str, Any] = {}
    try:
        (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="u1_2e_adobe_rgb_", dir=ROOT / "tmp"))
        for name, profile in profiles:
            path = scratch / f"{name}.tiff"
            _write(path, source, profile)
            source_before = _sha256(path)
            profile_before = _sha256_bytes(profile)
            working = load_working_image(path)
            expected = _oracle(source, profile)
            facts = adobe_rgb_icc_facts(profile)
            rows.append(
                {
                    "name": name,
                    "source_sha256": source_before,
                    "source_immutable": source_before == _sha256(path),
                    "profile_sha256": profile_before,
                    "profile_immutable": profile_before == _sha256_bytes(profile),
                    "pixel_sha256": _sha256_bytes(working.pixels.tobytes()),
                    "expected_pixel_sha256": _sha256_bytes(expected.tobytes()),
                    "pixel_exact": working.pixels.tobytes() == expected.tobytes(),
                    "working_space": working.working_space,
                    "warning_present": any(
                        warning.code == "embedded_adobe_rgb_to_linear_rec2020"
                        for warning in working.warnings
                    ),
                    "matrix_error": facts["maximum_matrix_absolute_error"],
                    "gamma_error": facts["maximum_gamma_absolute_error"],
                }
            )
        rows.sort(key=lambda row: row["name"])

        oriented_path = scratch / "orientation6.tiff"
        generated = _generated_profile()
        _write(oriented_path, source, generated, orientation=6)
        oriented_samples = np.ascontiguousarray(np.rot90(source, k=3, axes=(0, 1)))
        oriented = load_working_image(oriented_path)
        oriented_expected = _oracle(oriented_samples, generated)

        srgb_path = scratch / "srgb.tiff"
        _write(srgb_path, source, srgb_icc_profile())
        srgb = load_working_image(srgb_path)
        prophoto_path = scratch / "prophoto.tiff"
        prophoto_profile = _prophoto_profile()
        _write(prophoto_path, source, prophoto_profile)
        prophoto = load_working_image(prophoto_path)
        prophoto_expected = decode_prophoto_rgb16_to_linear_rec2020(
            source, prophoto_profile
        )
        legacy = {
            "orientation6_exact": oriented.pixels.tobytes()
            == oriented_expected.tobytes(),
            "orientation6_dimensions": list(oriented.pixels.shape),
            "srgb_working_space": srgb.working_space,
            "prophoto_exact": prophoto.pixels.tobytes()
            == prophoto_expected.tobytes(),
            "prophoto_working_space": prophoto.working_space,
        }
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=False)

    controls = {
        "nonuniform_trc_rejects": _rejects(
            _mutate_tag(_generated_profile(), b"gTRC", 1024)
        ),
        "wrong_primaries_rejects": _rejects(
            _mutate_tag(_generated_profile(), b"rXYZ", 6554)
        ),
        "malformed_profile_rejects": _rejects(b"not-an-icc-profile"),
    }
    gates = {
        "both_profile_encodings_exact": len(rows) == 2
        and all(row["pixel_exact"] for row in rows),
        "semantic_facts_within_tolerance": all(
            row["matrix_error"] <= config["icc"]["matrix_tolerance"]
            and row["gamma_error"] <= config["icc"]["gamma_tolerance"]
            for row in rows
        ),
        "source_and_profile_immutable": all(
            row["source_immutable"] and row["profile_immutable"] for row in rows
        ),
        "working_image_identity_exact": all(
            row["working_space"] == "linear_rec2020" and row["warning_present"]
            for row in rows
        ),
        "orientation_before_colour_exact": legacy["orientation6_exact"]
        and legacy["orientation6_dimensions"] == [3, 2, 3],
        "legacy_srgb_and_prophoto_exact": legacy["srgb_working_space"]
        == "linear_srgb"
        and legacy["prophoto_working_space"] == "linear_rec2020"
        and legacy["prophoto_exact"],
        "invalid_semantics_reject": all(controls.values()),
        "scratch_residue_zero": scratch is not None and not scratch.exists(),
    }
    bindings = config["bindings"]
    report = {
        "schema": SCHEMA,
        "experiment_id": "U1.2E_HIGH_PRECISION_ADOBE_RGB_TIFF",
        "status": (
            "PASS_RGB16_ADOBE_RGB_TIFF_INGRESS"
            if all(gates.values())
            else "FAIL_CLOSED_RGB16_ADOBE_RGB_TIFF_INGRESS"
        ),
        "bindings": {
            "config_sha256": _sha256(config_path),
            "contract_sha256": _sha256(ROOT / bindings["contract_path"]),
            "core_sha256": _sha256_bytes(
                b"".join((ROOT / path).read_bytes() for path in bindings["core_paths"])
            ),
            "test_sha256": _sha256(ROOT / bindings["test_path"]),
            "runner_sha256": _sha256(ROOT / bindings["runner_path"]),
            "test_profile_sha256": _sha256(test_profile),
        },
        "profile_rows": rows,
        "legacy_and_orientation": legacy,
        "rejection_controls": controls,
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
        default=ROOT / "configs/u1_2e_high_precision_adobe_rgb_tiff_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.output, reverse=args.reverse)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())


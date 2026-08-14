from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_u1_4c12_official_romm_sources import _tiff_bytes, _write_exact
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.preprocess import load_working_image

SCHEMA = "neuro-film.u1-4c13-cc0-synthetic-romm-source-manifest.v1"
EXPERIMENT_ID = "U1.4C13S"
SOURCE_MANIFEST = Path(
    "outputs/u6_p7h0_native_structure_value_source_preflight_v1/run_d/manifest.json"
)
SOURCE_MANIFEST_SHA256 = "3c698d8df93a7b48b87dff81ad6ddbaeb11d7c8478138ddef9545d57e336fcd7"
PROFILE = Path("data/wide_gamut/icc_official/ISO22028-2_ROMM-RGB.icc")
PROFILE_SHA256 = "96b2f2987f83e2a545e607799fbfdff43ef8158fb9b215b187c574db8f145aaf"
DATA_ROOT = Path("data/wide_gamut/u1_4c13_cc0_synthetic_romm")
MINIMUM_OOG_FRACTION = 0.005


class CC0SyntheticROMMSourceError(RuntimeError):
    """Raised when the frozen C13 source transformation drifts."""


def build(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise CC0SyntheticROMMSourceError(
            "C13 source report directory must be create-only"
        )
    source_path = ROOT / SOURCE_MANIFEST
    profile_path = ROOT / PROFILE
    if hash_file(source_path) != SOURCE_MANIFEST_SHA256:
        raise CC0SyntheticROMMSourceError("C13 parent source manifest drift")
    if hash_file(profile_path) != PROFILE_SHA256:
        raise CC0SyntheticROMMSourceError("C13 official ICC profile drift")
    profile = profile_path.read_bytes()
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        len(source_rows) != 9
        or len({row.get("id") for row in source_rows}) != 9
        or len({row.get("make") for row in source_rows}) != 9
    ):
        raise CC0SyntheticROMMSourceError("C13 parent source inventory drift")

    output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for source in source_rows:
        decoded_path = ROOT / Path(source["decoded_path"])
        if (
            source.get("allowed_use")
            != "internal_independent_digital_ood_confirmation"
            or source.get("rights_scope")
            != "CC0_public_domain_internal_evaluation"
            or source.get("decoded_color_state")
            != "relative_display_srgb_approximation"
            or not decoded_path.is_file()
            or hash_file(decoded_path) != source["decoded_sha256"]
        ):
            raise CC0SyntheticROMMSourceError("C13 parent source row drift")
        bgr = cv2.imread(str(decoded_path), cv2.IMREAD_COLOR)
        if (
            bgr is None
            or bgr.dtype != np.uint8
            or bgr.shape
            != (int(source["height"]), int(source["width"]), 3)
        ):
            raise CC0SyntheticROMMSourceError("C13 source decode drift")
        rgb8 = np.ascontiguousarray(bgr[..., ::-1])
        rgb16 = np.ascontiguousarray(rgb8.astype(np.uint16) * np.uint16(257))
        relative = DATA_ROOT / f"{source['id']}.tiff"
        payload = _tiff_bytes(rgb16, profile)
        destination = ROOT / relative
        _write_exact(destination, payload)
        with tifffile.TiffFile(destination) as document:
            page = document.pages[0]
            decoded = np.ascontiguousarray(page.asarray())
            profile_tag = page.tags.get(34675)
            embedded = bytes(profile_tag.value) if profile_tag is not None else b""
        working = load_working_image(destination)
        oog_fraction = float(
            np.mean(np.any((working.pixels < 0.0) | (working.pixels > 1.0), axis=-1))
        )
        if (
            decoded.dtype != np.uint16
            or decoded.shape != rgb16.shape
            or decoded.tobytes() != rgb16.tobytes()
            or embedded != profile
            or working.working_space != "linear_rec2020"
            or working.transfer_state != "display_linear"
            or working.pixels.dtype != np.float32
            or not np.isfinite(working.pixels).all()
            or oog_fraction < MINIMUM_OOG_FRACTION
        ):
            raise CC0SyntheticROMMSourceError("C13 synthetic ROMM source gate failed")
        rows.append(
            {
                "id": source["id"],
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "width": int(source["width"]),
                "height": int(source["height"]),
                "dtype": "uint16",
                "pixel_sha256": hashlib.sha256(rgb16.tobytes()).hexdigest(),
                "embedded_icc_sha256": PROFILE_SHA256,
                "make": source["make"],
                "model": source["model"],
                "source_png_sha256": source["decoded_sha256"],
                "source_raw_sha256": source["raw_sha256"],
                "source_url": source["source_url"],
                "license": source["license"],
                "allowed_use": source["allowed_use"],
                "rights_scope": source["rights_scope"],
                "synthetic_transform": "uint8-srgb-code-times-257-as-romm-rgb16-v1",
                "rec2020_out_of_gamut_fraction": oog_fraction,
            }
        )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_manifest_path": SOURCE_MANIFEST.as_posix(),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "official_icc_path": PROFILE.as_posix(),
        "official_icc_sha256": PROFILE_SHA256,
        "synthetic_transform": "uint8-srgb-code-times-257-as-romm-rgb16-v1",
        "minimum_required_rec2020_out_of_gamut_fraction": MINIMUM_OOG_FRACTION,
        "rows": rows,
        "row_count": len(rows),
        "make_count": len({row["make"] for row in rows}),
        "total_pixels": sum(int(row["width"]) * int(row["height"]) for row in rows),
        "minimum_rec2020_out_of_gamut_fraction": min(
            float(row["rec2020_out_of_gamut_fraction"]) for row in rows
        ),
        "pixel_roundtrip_exact": True,
        "embedded_profile_exact": True,
        "product_decode_pass": True,
        "algorithm_execution_count": 0,
        "claim_ceiling": (
            "CC0 natural-scene structure under a fixed code-preserving synthetic ROMM "
            "reinterpretation; not natural WCG pixels, camera colour, calibrated colour, "
            "arbitrary ICC, film, stock, HDR, ACES, preference, or product promotion."
        ),
    }
    report["stable_source_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = build(output_dir)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.preprocess import load_working_image

SCHEMA = "neuro-film.u1-4c12-official-romm-cross-dataset-source-manifest.v1"
EXPERIMENT_ID = "U1.4C12S"
SOURCE_REPORT = Path("outputs/u1_4c7_gamutmlp_source_audit_v1/run_c/report.json")
SOURCE_REPORT_SHA256 = "cd22fb5dd851f11801adb52208399ba1b59fb51b5f3990229bc288023de3a435"
PROFILE = Path("data/wide_gamut/icc_official/ISO22028-2_ROMM-RGB.icc")
PROFILE_SHA256 = "96b2f2987f83e2a545e607799fbfdff43ef8158fb9b215b187c574db8f145aaf"
DATA_ROOT = Path("data/wide_gamut/u1_4c12_official_romm")


class OfficialROMMSourceError(RuntimeError):
    """Raised when the frozen C12 source transformation drifts."""


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise OfficialROMMSourceError(f"C12 existing source identity drift: {path}")
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        raise OfficialROMMSourceError(f"C12 stale temporary output: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _tiff_bytes(rgb: np.ndarray, profile: bytes) -> bytes:
    buffer = BytesIO()
    tifffile.imwrite(
        buffer,
        rgb,
        photometric="rgb",
        metadata=None,
        byteorder="<",
        rowsperstrip=int(rgb.shape[0]),
        extratags=[(34675, "B", len(profile), profile, False)],
    )
    return buffer.getvalue()


def build(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise OfficialROMMSourceError("C12 source report directory must be create-only")
    source_report_path = ROOT / SOURCE_REPORT
    profile_path = ROOT / PROFILE
    if hash_file(source_report_path) != SOURCE_REPORT_SHA256:
        raise OfficialROMMSourceError("C12 parent source report drift")
    if hash_file(profile_path) != PROFILE_SHA256:
        raise OfficialROMMSourceError("C12 official ICC profile drift")
    profile = profile_path.read_bytes()
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    selected = source_report.get("selected_manifest", {})
    source_rows = selected.get("rows", [])
    if (
        source_report.get("automatic_pass") is not True
        or len(source_rows) != 24
        or selected.get("allowed_use") != "internal_research_only"
    ):
        raise OfficialROMMSourceError("C12 parent source eligibility drift")

    rows: list[dict[str, object]] = []
    ids: set[str] = set()
    output_dir.mkdir(parents=True)
    for source in source_rows:
        source_path = ROOT / Path(source["path"])
        if not source_path.is_file() or hash_file(source_path) != source["member_sha256"]:
            raise OfficialROMMSourceError("C12 selected PNG identity drift")
        bgr = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if (
            bgr is None
            or bgr.dtype != np.uint16
            or bgr.shape != (512, 512, 3)
        ):
            raise OfficialROMMSourceError("C12 selected PNG decode drift")
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        pixel_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        if pixel_sha256 != source["pixel_sha256"]:
            raise OfficialROMMSourceError("C12 selected pixel identity drift")
        row_id = f"{source['camera']}-{source['style']}-{source['source_id']}".lower()
        if row_id in ids:
            raise OfficialROMMSourceError("C12 derived source ID collision")
        ids.add(row_id)
        relative = DATA_ROOT / source["camera"] / f"{row_id}.tiff"
        payload = _tiff_bytes(rgb, profile)
        destination = ROOT / relative
        _write_exact(destination, payload)
        with tifffile.TiffFile(destination) as document:
            page = document.pages[0]
            decoded = np.ascontiguousarray(page.asarray())
            profile_tag = page.tags.get(34675)
            embedded = bytes(profile_tag.value) if profile_tag is not None else b""
        working = load_working_image(destination)
        if (
            decoded.dtype != np.uint16
            or decoded.shape != rgb.shape
            or decoded.tobytes() != rgb.tobytes()
            or embedded != profile
            or working.working_space != "linear_rec2020"
            or working.transfer_state != "display_linear"
            or working.pixels.dtype != np.float32
            or not np.isfinite(working.pixels).all()
        ):
            raise OfficialROMMSourceError("C12 TIFF product roundtrip drift")
        rows.append(
            {
                "id": row_id,
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "width": 512,
                "height": 512,
                "dtype": "uint16",
                "pixel_sha256": pixel_sha256,
                "embedded_icc_sha256": PROFILE_SHA256,
                "camera": source["camera"],
                "style": source["style"],
                "source_id": source["source_id"],
                "source_png_sha256": source["member_sha256"],
                "allowed_use": selected["allowed_use"],
                "rights_scope": selected["rights_scope"],
                "colour_semantics": "official ISO22028-2 ROMM RGB ICC v4",
            }
        )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_report_path": SOURCE_REPORT.as_posix(),
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "official_icc_path": PROFILE.as_posix(),
        "official_icc_sha256": PROFILE_SHA256,
        "rows": rows,
        "row_count": len(rows),
        "camera_count": len({row["camera"] for row in rows}),
        "total_tiff_bytes": sum(int(row["bytes"]) for row in rows),
        "pixel_roundtrip_exact": True,
        "embedded_profile_exact": True,
        "product_decode_pass": True,
        "algorithm_execution_count": 0,
        "claim_ceiling": (
            "Previously consumed GamutMLP internal-research-only photographic content "
            "repackaged without pixel change under the official ROMM RGB ICC profile; "
            "not fresh content, arbitrary ICC, calibrated colour, redistribution, film, "
            "stock, HDR, ACES, preference, or product promotion."
        ),
    }
    report["stable_source_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    manifest = output_dir / "manifest.json"
    manifest.write_text(
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

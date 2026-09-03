from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.style_safe_engine import replay_style_safe_recipe_to_file
from src.inference.three_stock_preview import render_three_stock_previews_to_directory
from src.preprocess import inspect_input, load_working_image
from src.preprocess.heic_sdr import decode_strict_sdr_heic, inspect_strict_sdr_heic

CONFIG = ROOT / "configs/u7_21a_strict_sdr_heic_ingress_v1.json"
SOURCE_ROOT = ROOT / "data/external/u7_21a_strict_sdr_heic_v1"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
PREVIEW_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
WHEEL = (
    ROOT / "tmp/u7_21a_heic_preflight/wheels/pi_heif-1.4.0-cp312-cp312-win_amd64.whl"
)
CANONICAL_PRODUCT_SHA256 = (
    "fc51547d1e00a0a4a36dce96847afe09d27b3b531b65172d3f32fa2a46d2087d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fixture in config["fixtures"]:
        path = SOURCE_ROOT / fixture["path"]
        rows.append(
            {
                "role": fixture["role"],
                "path": fixture["path"],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "expected_bytes": fixture["bytes"],
                "expected_sha256": fixture["sha256"],
            }
        )
    return rows


def _wheel_record(config: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(WHEEL) as archive:
        bundled = archive.read(
            "pi_heif-1.4.0.dist-info/licenses/LICENSES_bundled.txt"
        ).decode("utf-8")
        names = tuple(archive.namelist())
    return {
        "filename": WHEEL.name,
        "bytes": WHEEL.stat().st_size,
        "sha256": _sha256(WHEEL),
        "expected_bytes": config["dependency"]["windows_cp312_wheel_bytes"],
        "expected_sha256": config["dependency"]["windows_cp312_wheel_sha256"],
        "bundled_licenses_sha256": hashlib.sha256(bundled.encode("utf-8")).hexdigest(),
        "declares_libheif_lgplv3": "Name: libheif\nLicense: LGPLv3" in bundled,
        "declares_libde265_lgplv3": "Name: libde265\nLicense: LGPLv3" in bundled,
        "contains_x265": any("x265" in name.casefold() for name in names),
    }


def _positive_record(path: Path) -> dict[str, Any]:
    info = inspect_strict_sdr_heic(path)
    decoded = decode_strict_sdr_heic(path, info)
    source_profile = ImageCms.ImageCmsProfile(BytesIO(decoded.info["icc_profile"]))
    srgb = ImageCms.profileToProfile(
        decoded,
        source_profile,
        ImageCms.createProfile("sRGB"),
        outputMode="RGB",
    )
    working = load_working_image(path)
    inspection = inspect_input(path)
    return {
        "width": info.width,
        "height": info.height,
        "mode": info.mode,
        "bit_depth": info.bit_depth,
        "frame_count": info.frame_count,
        "primary_index": info.primary_index,
        "colour_kind": info.colour_kind,
        "icc_bytes": len(info.icc_profile or b""),
        "exif_orientation": info.exif_orientation,
        "inspection_orientation": inspection.orientation,
        "decoded_srgb8_sha256": hashlib.sha256(srgb.tobytes()).hexdigest(),
        "linear_f32_sha256": _array_sha256(working.pixels),
        "linear_shape": list(working.pixels.shape),
        "linear_owned": bool(working.pixels.flags.owndata),
        "linear_finite": bool(np.isfinite(working.pixels).all()),
        "working_space": working.working_space,
        "transfer_state": working.transfer_state,
    }


def _negative_rows(config: dict[str, Any], order: str) -> list[dict[str, Any]]:
    fixtures = [
        row for row in config["fixtures"] if not row["role"].startswith("positive")
    ]
    execution = fixtures if order == "forward" else list(reversed(fixtures))
    results: dict[str, dict[str, Any]] = {}
    for fixture in execution:
        path = SOURCE_ROOT / fixture["path"]
        inspection = inspect_input(path)
        error = None
        try:
            load_working_image(path)
        except ValueError as exc:
            error = str(exc)
        results[fixture["role"]] = {
            "role": fixture["role"],
            "path": fixture["path"],
            "inspection_status": inspection.hdr_metadata.get("strict_sdr_heic"),
            "inspection_reason": inspection.hdr_metadata.get("strict_sdr_heic_reason"),
            "load_error": error,
            "rejected": error is not None,
        }
    return [results[row["role"]] for row in fixtures]


def _public_product_probe(source: Path, scratch: Path) -> dict[str, Any]:
    output = scratch / "arrow-ektar.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--output",
            str(output),
            "--write-recipe",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    replay = scratch / "arrow-ektar-replay.png"
    replay_digest = replay_style_safe_recipe_to_file(
        recipe,
        profile_path=PRODUCT_PROFILE,
        output_path=replay,
        root=ROOT,
    )
    return {
        "returncode": completed.returncode,
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "replay_sha256": replay_digest,
        "replay_byte_exact": replay.read_bytes() == output.read_bytes(),
        "recipe_style": recipe["render"]["style"],
        "recipe_look_amount": recipe["render"]["look_amount"],
        "output_label": recipe["claim"]["output_label"],
        "evidence_grade": recipe["claim"]["evidence_grade"],
        "calibrated_reference_allowed": recipe["claim"]["calibrated_reference_allowed"],
    }


def _preview_probe(source: Path, scratch: Path) -> dict[str, Any]:
    directory = scratch / "previews"
    manifest = render_three_stock_previews_to_directory(
        source,
        directory,
        root=ROOT,
        profile_path=PREVIEW_PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        max_preview_pixels=48_000,
        max_preview_width=320,
        max_preview_height=320,
        look_amount=0.65,
        tile_size=64,
        tile_workers=1,
        include_input_preview=True,
    )
    return {
        "source_width": manifest["source_width"],
        "source_height": manifest["source_height"],
        "preview_width": manifest["preview_width"],
        "preview_height": manifest["preview_height"],
        "preview_pixels": manifest["preview_pixels"],
        "claim_ceiling": manifest["claim_ceiling"],
        "input_preview_sha256": _sha256(Path(manifest["input_preview"]["output_path"])),
        "rows": [
            {
                "style_id": row["style_id"],
                "sha256": _sha256(Path(row["output_path"])),
            }
            for row in manifest["rows"]
        ],
    }


def _canonical_product_probe(scratch: Path) -> dict[str, Any]:
    y, x = np.mgrid[0:43, 0:61]
    pixels = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 11) % 256,
            (x * 13 + y * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    source = scratch / "canonical-source.png"
    output = scratch / "canonical-ektar.png"
    Image.fromarray(pixels, mode="RGB").save(source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            str(source),
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "output_sha256": _sha256(output) if output.is_file() else None,
        "expected_output_sha256": CANONICAL_PRODUCT_SHA256,
    }


def build_report(*, scratch: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if scratch.exists():
        raise FileExistsError("scratch path must be absent")
    scratch.mkdir(parents=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_before = _source_rows(config)
    try:
        positive_path = SOURCE_ROOT / config["fixtures"][0]["path"]
        wheel = _wheel_record(config)
        positive = _positive_record(positive_path)
        negatives = _negative_rows(config, order)
        product = _public_product_probe(positive_path, scratch)
        preview = _preview_probe(positive_path, scratch)
        canonical = _canonical_product_probe(scratch)
        source_after = _source_rows(config)
        gates = {
            "source_identities_exact": all(
                row["bytes"] == row["expected_bytes"]
                and row["sha256"] == row["expected_sha256"]
                for row in source_before
            ),
            "source_immutable": source_before == source_after,
            "decode_only_wheel_exact": (
                wheel["bytes"] == wheel["expected_bytes"]
                and wheel["sha256"] == wheel["expected_sha256"]
                and wheel["declares_libheif_lgplv3"]
                and wheel["declares_libde265_lgplv3"]
                and not wheel["contains_x265"]
            ),
            "positive_metadata_exact": (
                positive["width"] == 3024
                and positive["height"] == 4032
                and positive["mode"] == "RGB"
                and positive["bit_depth"] == 8
                and positive["frame_count"] == 1
                and positive["primary_index"] == 0
                and positive["colour_kind"] == "icc"
                and positive["inspection_orientation"] == 1
            ),
            "positive_srgb8_exact": (
                positive["decoded_srgb8_sha256"] == config["oracle"]["srgb8_sha256"]
            ),
            "positive_linear_f32_exact": (
                positive["linear_f32_sha256"] == config["oracle"]["linear_f32_sha256"]
                and positive["linear_shape"] == [4032, 3024, 3]
                and positive["linear_owned"]
                and positive["linear_finite"]
            ),
            "negative_controls_reject": all(row["rejected"] for row in negatives),
            "public_ektar_recipe_replay_exact": (
                product.get("returncode") == 0
                and product.get("replay_byte_exact") is True
                and product.get("output_sha256") == product.get("replay_sha256")
                and product.get("recipe_style") == "ektar_100"
                and product.get("recipe_look_amount") == 0.65
            ),
            "look_approximation_claim_exact": (
                product.get("output_label") == "film-inspired"
                and product.get("evidence_grade") == "look-approximation"
                and product.get("calibrated_reference_allowed") is False
                and "non-calibrated Look Approximations" in preview["claim_ceiling"]
            ),
            "bounded_three_look_preview": (
                preview["preview_pixels"] <= 48_000
                and {row["style_id"] for row in preview["rows"]}
                == {"velvia_50", "portra_400", "ektar_100"}
            ),
            "canonical_preexisting_output_exact": (
                canonical["returncode"] == 0
                and canonical["output_sha256"] == CANONICAL_PRODUCT_SHA256
            ),
        }
        return {
            "schema": "kmcfm.u7-21a-strict-sdr-heic-ingress-formal-report.v1",
            "source_commit": _git_head(),
            "config_sha256": _sha256(CONFIG),
            "status": (
                "PASS_PRIVATE_U7_21A_STRICT_SDR_HEIC_INGRESS"
                if all(gates.values())
                else "FAIL_CLOSED_U7_21A_STRICT_SDR_HEIC_INGRESS"
            ),
            "wheel": wheel,
            "sources": source_before,
            "positive": positive,
            "negative_controls": negatives,
            "public_product": product,
            "desktop_preview": preview,
            "canonical_product": canonical,
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output path must be absent")
    report = build_report(scratch=args.scratch, order=args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if report["status"].startswith("PASS_PRIVATE") else 1


if __name__ == "__main__":
    raise SystemExit(main())

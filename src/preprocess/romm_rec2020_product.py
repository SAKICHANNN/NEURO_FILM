"""Strict official-ROMM RGB16 to bounded Rec.2020 SDR RGB16 conversion."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.color_engine.oklab_analytical_interior import (
    analytical_oklab_interior_rec2020,
)

from .color_management import REC2020_SDR_CICP, linear_rec2020_to_rec2020
from .output_encode import save_rec2020_16_png
from .pipeline import load_working_image
from .types import DecodeWarning, WorkingImage

OFFICIAL_ROMM_ICC_SHA256 = (
    "96b2f2987f83e2a545e607799fbfdff43ef8158fb9b215b187c574db8f145aaf"
)
ROMM_REC2020_CAPABILITY_ID = "official-romm-rgb16-to-rec2020-sdr-rgb16-png-v1"
ROMM_REC2020_RECEIPT_SCHEMA = "neuro-film.romm-rec2020-conversion-receipt.v1"
ROMM_REC2020_MAPPER_ID = "oklab-oog-only-soft-interval-maximum-chroma-v1"
ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256 = (
    "c720bc6777c4ac3eb5b481eb98e6b9759d876b68ef720283536180930be70e5c"
)


class ROMMRec2020ConversionError(RuntimeError):
    """Raised when the strict product conversion boundary is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_romm_profile_sha256(path: Path) -> str:
    try:
        with tifffile.TiffFile(path) as document:
            if len(document.pages) != 1:
                raise ROMMRec2020ConversionError("ROMM input must contain one TIFF page")
            tag = document.pages[0].tags.get(34675)
            profile = bytes(tag.value) if tag is not None else b""
    except ROMMRec2020ConversionError:
        raise
    except Exception as exc:
        raise ROMMRec2020ConversionError("ROMM input is not a readable TIFF") from exc
    profile_sha256 = hashlib.sha256(profile).hexdigest()
    if profile_sha256 != OFFICIAL_ROMM_ICC_SHA256:
        raise ROMMRec2020ConversionError("input does not embed the exact official ROMM ICC")
    return profile_sha256


def convert_official_romm_rgb16_to_rec2020_png(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Convert one exact official-ROMM RGB16 TIFF to bounded Rec.2020 RGB16 PNG.

    The analytical mapper changes only pixels outside the linear Rec.2020 cube;
    it performs no component clipping or post-hoc limiting. The emitted PNG is
    reopened and checked against the exact expected RGB16 samples before the
    capability receipt is returned.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise ROMMRec2020ConversionError("ROMM input file is missing")
    if output_path.suffix.casefold() != ".png":
        raise ROMMRec2020ConversionError("Rec.2020 output must use a .png extension")
    if output_path.exists():
        raise ROMMRec2020ConversionError("Rec.2020 output must be create-only")

    mapped_working, mapping = load_and_map_official_romm_rgb16(input_path)
    mapped = mapped_working.pixels
    try:
        save_rec2020_16_png(mapped_working, output_path)
        stored_bgr = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        expected = np.rint(linear_rec2020_to_rec2020(mapped) * 65535.0).astype(
            np.uint16
        )
        if (
            stored_bgr is None
            or stored_bgr.dtype != np.uint16
            or stored_bgr.shape != expected.shape
            or not np.array_equal(stored_bgr[..., ::-1], expected)
        ):
            raise ROMMRec2020ConversionError("Rec.2020 PNG exact sample readback failed")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return {
        "schema": ROMM_REC2020_RECEIPT_SCHEMA,
        "capability_id": ROMM_REC2020_CAPABILITY_ID,
        "qualification": {
            "experiment_id": "U1.4C13",
            "evidence_path": "docs/evidence/U1_4C13_CC0_SYNTHETIC_ROMM_STRESS_RESULT.json",
            "evidence_sha256": ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256,
            "required_status": "PASS_SCOPED_CC0_SYNTHETIC_ROMM_STRESS",
        },
        "input": mapping["input"],
        "transform": mapping["transform"],
        "diagnostics": mapping["diagnostics"],
        "output": {
            "sha256": _sha256(output_path),
            "format": "PNG",
            "bit_depth": 16,
            "working_space": "linear_rec2020",
            "transfer": "BT.2020 SDR",
            "cicp": REC2020_SDR_CICP.hex(),
            "exact_sample_readback": True,
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "strict official-ROMM RGB16 TIFF to bounded relative Rec.2020 SDR RGB16 PNG; "
            "not arbitrary ICC, HDR, camera colour, film, stock, or calibrated colour"
        ),
    }


def load_and_map_official_romm_rgb16(
    input_path: Path,
) -> tuple[WorkingImage, dict[str, Any]]:
    """Load exact official-ROMM RGB16 and return its bounded Rec.2020 image and facts."""

    input_path = Path(input_path)
    if not input_path.is_file():
        raise ROMMRec2020ConversionError("ROMM input file is missing")
    profile_sha256 = _official_romm_profile_sha256(input_path)
    working = load_working_image(input_path)
    if (
        working.bit_depth_in != 16
        or working.source_profile.kind != "icc"
        or working.working_space != "linear_rec2020"
        or working.transfer_state != "display_linear"
    ):
        raise ROMMRec2020ConversionError(
            "official ROMM conversion requires RGB16 ICC display-linear Rec.2020 ingress"
        )

    source = working.pixels
    source_in_gamut = np.all((source >= 0.0) & (source <= 1.0), axis=2)
    mapped, chroma_scale = analytical_oklab_interior_rec2020(
        source,
        softness=1.0 / 64.0,
        margin=2.0 / 65535.0,
        iterations=24,
    )
    if not bool(
        np.all(np.isfinite(mapped))
        and np.all(mapped >= 0.0)
        and np.all(mapped <= 1.0)
    ):
        raise ROMMRec2020ConversionError("analytical ROMM mapping is not bounded")
    if not np.array_equal(mapped[source_in_gamut], source[source_in_gamut]):
        raise ROMMRec2020ConversionError("analytical ROMM mapping changed an in-gamut pixel")

    mapped_working = replace(
        working,
        pixels=mapped,
        warnings=[
            *working.warnings,
            DecodeWarning(
                "analytical_rec2020_interior_mapping",
                f"Mapped only out-of-gamut pixels with {ROMM_REC2020_MAPPER_ID}.",
            ),
        ],
    )
    changed = ~source_in_gamut
    return mapped_working, {
        "input": {
            "sha256": _sha256(input_path),
            "embedded_icc_sha256": profile_sha256,
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
            "bit_depth": 16,
        },
        "transform": {
            "mapper_id": ROMM_REC2020_MAPPER_ID,
            "softness": 1.0 / 64.0,
            "rgb16_margin": 2.0 / 65535.0,
            "bisection_iterations": 24,
            "hard_clipping": False,
            "posthoc_limiting": False,
        },
        "diagnostics": {
            "mapped_pixel_count": int(np.count_nonzero(changed)),
            "mapped_pixel_fraction": float(np.mean(changed)),
            "minimum_chroma_scale": float(np.min(chroma_scale)),
            "output_minimum": float(np.min(mapped)),
            "output_maximum": float(np.max(mapped)),
        },
    }


__all__ = [
    "OFFICIAL_ROMM_ICC_SHA256",
    "ROMM_REC2020_CAPABILITY_ID",
    "ROMM_REC2020_QUALIFICATION_EVIDENCE_SHA256",
    "ROMM_REC2020_RECEIPT_SCHEMA",
    "ROMMRec2020ConversionError",
    "convert_official_romm_rgb16_to_rec2020_png",
    "load_and_map_official_romm_rgb16",
]

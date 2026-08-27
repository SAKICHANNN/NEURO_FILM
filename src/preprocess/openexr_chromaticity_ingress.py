"""Strict opt-in OpenEXR chromaticity-aware WorkingImage ingress."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import numpy as np

from .types import SourceProfile, WorkingImage

OPENEXR_CHROMATICITY_INGRESS_ID = "openexr-chromaticities-to-rec2020-d65-v1"
OPENEXR_DEFAULT_REC709_D65 = (
    0.64,
    0.33,
    0.30,
    0.60,
    0.15,
    0.06,
    0.3127,
    0.3290,
)
TARGET_D65_XY = (0.3127, 0.3290)
MAXIMUM_WIDTH = 16_384
MAXIMUM_HEIGHT = 16_384
MAXIMUM_PIXELS = 100_000_000
MAXIMUM_ABSOLUTE_COMPONENT = 65_504.0

_BRADFORD = np.asarray(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_REC2020 = np.asarray(
    [
        [1.7166511879712674, -0.3556707837763924, -0.25336628137365974],
        [-0.6666843518324892, 1.6164812366349395, 0.01576854581391113],
        [0.017639857445310783, -0.042770613257808524, 0.9421031212354738],
    ],
    dtype=np.float64,
)


class OpenExrChromaticityError(ValueError):
    """Raised when an EXR violates the opt-in chromaticity contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chromaticity_tuple(value: object) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise OpenExrChromaticityError("invalid chromaticities") from exc
    if len(result) != 8 or not np.isfinite(result).all():
        raise OpenExrChromaticityError("invalid chromaticities")
    return result


def _white_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    if not np.isfinite((x, y)).all() or y <= 0.0 or x < 0.0 or x + y > 1.0:
        raise OpenExrChromaticityError("invalid white chromaticity")
    return np.asarray([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _rgb_to_xyz_matrix(chromaticities: tuple[float, ...]) -> np.ndarray:
    xr, yr, xg, yg, xb, yb, xw, yw = chromaticities
    components = np.asarray((xr, yr, xg, yg, xb, yb, xw, yw), dtype=np.float64)
    if not np.isfinite(components).all():
        raise OpenExrChromaticityError("invalid chromaticities")
    primaries = ((xr, yr), (xg, yg), (xb, yb))
    columns: list[list[float]] = []
    for x, y in primaries:
        z = 1.0 - x - y
        if x < 0.0 or y < 0.0 or z < 0.0:
            raise OpenExrChromaticityError("invalid primary chromaticity")
        columns.append([x, y, z])
    basis = np.asarray(columns, dtype=np.float64).T
    if abs(float(np.linalg.det(basis))) <= 1e-12:
        raise OpenExrChromaticityError("singular primary chromaticities")
    white = _white_xyz((xw, yw))
    scales = np.linalg.solve(basis, white)
    matrix = basis @ np.diag(scales)
    if not np.isfinite(matrix).all():
        raise OpenExrChromaticityError("non-finite RGB-to-XYZ matrix")
    return matrix


def _bradford_adaptation(
    source_xy: tuple[float, float], target_xy: tuple[float, float]
) -> np.ndarray:
    source = _BRADFORD @ _white_xyz(source_xy)
    target = _BRADFORD @ _white_xyz(target_xy)
    if np.any(np.abs(source) <= 1e-12):
        raise OpenExrChromaticityError("singular source white adaptation")
    return np.linalg.inv(_BRADFORD) @ np.diag(target / source) @ _BRADFORD


def _conversion_matrix(chromaticities: tuple[float, ...]) -> np.ndarray:
    source_to_xyz = _rgb_to_xyz_matrix(chromaticities)
    adaptation = _bradford_adaptation(
        (chromaticities[6], chromaticities[7]), TARGET_D65_XY
    )
    matrix = _XYZ_D65_TO_REC2020 @ adaptation @ source_to_xyz
    if not np.isfinite(matrix).all():
        raise OpenExrChromaticityError("non-finite conversion matrix")
    return matrix


def _validate_pixels(pixels: object) -> np.ndarray:
    if not isinstance(pixels, np.ndarray):
        raise OpenExrChromaticityError("packed RGB pixels must be a numpy array")
    if pixels.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise OpenExrChromaticityError("packed RGB pixels must be float16 or float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise OpenExrChromaticityError("packed RGB pixels must be HxWx3")
    height, width, _ = pixels.shape
    if height <= 0 or width <= 0:
        raise OpenExrChromaticityError("OpenEXR dimensions must be positive")
    if height > MAXIMUM_HEIGHT or width > MAXIMUM_WIDTH:
        raise OpenExrChromaticityError("OpenEXR dimensions exceed the frozen ceiling")
    if height * width > MAXIMUM_PIXELS:
        raise OpenExrChromaticityError("OpenEXR pixel count exceeds the frozen ceiling")
    if not np.isfinite(pixels).all():
        raise OpenExrChromaticityError("OpenEXR pixels must be finite")
    if float(np.max(np.abs(pixels))) > MAXIMUM_ABSOLUTE_COMPONENT:
        raise OpenExrChromaticityError("OpenEXR component exceeds the frozen ceiling")
    return pixels


def _load_with_module(
    path: Path,
    openexr: ModuleType,
    *,
    allow_standard_rec709_default: bool,
) -> WorkingImage:
    before = _sha256_file(path)
    try:
        with openexr.File(str(path)) as exr_file:
            parts = getattr(exr_file, "parts", None)
            if parts is None or len(parts) != 1:
                raise OpenExrChromaticityError("exactly one OpenEXR part is required")
            header = dict(exr_file.header())
            if header.get("type") != openexr.scanlineimage:
                raise OpenExrChromaticityError(
                    "only non-deep scanline OpenEXR is accepted"
                )
            channels = exr_file.channels()
            if sorted(channels) != ["RGB"]:
                raise OpenExrChromaticityError(
                    "exactly one packed RGB group is required"
                )
            if "chromaticities" in header:
                chromaticities = _chromaticity_tuple(header["chromaticities"])
                identity_mode = "embedded_chromaticities"
            elif allow_standard_rec709_default:
                chromaticities = OPENEXR_DEFAULT_REC709_D65
                identity_mode = "openexr_standard_default_rec709_d65"
            else:
                raise OpenExrChromaticityError(
                    "missing chromaticities require explicit standard-default authorization"
                )
            source_pixels = _validate_pixels(channels["RGB"].pixels)
            converted = np.asarray(
                source_pixels.astype(np.float64) @ _conversion_matrix(chromaticities).T,
                dtype=np.float32,
            )
    except OpenExrChromaticityError:
        raise
    except Exception as exc:
        raise OpenExrChromaticityError(
            "failed to read chromaticity-aware OpenEXR"
        ) from exc
    if _sha256_file(path) != before:
        raise OpenExrChromaticityError("source changed during decode")
    if not np.isfinite(converted).all():
        raise OpenExrChromaticityError("conversion produced non-finite values")
    pixels = np.ascontiguousarray(converted).copy()
    return WorkingImage(
        pixels=pixels,
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("unknown", "OpenEXR chromaticity identity"),
        hdr_metadata={
            "chromaticities": list(chromaticities),
            "color_identity_mode": identity_mode,
            "ingress_id": OPENEXR_CHROMATICITY_INGRESS_ID,
            "target_white": "D65",
        },
        orientation_applied=False,
        alpha_policy="absent",
        bit_depth_in=16 if source_pixels.dtype == np.float16 else 32,
        source_path=path,
    )


def load_openexr_chromaticity_working_image(
    path: Path | str,
    *,
    allow_standard_rec709_default: bool = False,
) -> WorkingImage:
    """Load one strict chromaticity-identified EXR into linear Rec.2020/D65."""
    source = Path(path)
    if not source.is_file():
        raise OpenExrChromaticityError("source must be an existing file")
    try:
        openexr = importlib.import_module("OpenEXR")
    except ImportError as exc:
        raise OpenExrChromaticityError("OpenEXR 3.4.15 is required") from exc
    if getattr(openexr, "__version__", None) != "3.4.15":
        raise OpenExrChromaticityError("OpenEXR runtime must be exactly 3.4.15")
    return _load_with_module(
        source,
        openexr,
        allow_standard_rec709_default=allow_standard_rec709_default,
    )


__all__ = [
    "OPENEXR_CHROMATICITY_INGRESS_ID",
    "OpenExrChromaticityError",
    "load_openexr_chromaticity_working_image",
]

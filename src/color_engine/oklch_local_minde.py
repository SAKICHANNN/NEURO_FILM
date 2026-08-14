"""W3C CSS Color 4 OKLCh local-MINDE mapping for linear Rec.2020."""

from __future__ import annotations

import numpy as np

# W3C CSS Color 4 sample-conversion matrices, column-vector convention.
_XYZ_TO_LMS = np.asarray(
    [
        [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
        [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
        [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
    ],
    dtype=np.float64,
)
_LMS_TO_OKLAB = np.asarray(
    [
        [0.2104542683093140, 0.7936177747023054, -0.0040720430116193],
        [1.9779985324311684, -2.4285922420485799, 0.4505937096174110],
        [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
    ],
    dtype=np.float64,
)
_OKLAB_TO_LMS = np.asarray(
    [
        [1.0, 0.3963377773761749, 0.2158037573099136],
        [1.0, -0.1055613458156586, -0.0638541728258133],
        [1.0, -0.0894841775298119, -1.2914855480194092],
    ],
    dtype=np.float64,
)
_LMS_TO_XYZ = np.asarray(
    [
        [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
        [-0.0405757452148008, 1.1122868032803170, -0.0717110580655164],
        [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
    ],
    dtype=np.float64,
)
_REC2020_TO_XYZ = np.asarray(
    [
        [63426534 / 99577255, 20160776 / 139408157, 47086771 / 278816314],
        [26158966 / 99577255, 472592308 / 697040785, 8267143 / 139408157],
        [0.0, 19567812 / 697040785, 295819943 / 278816314],
    ],
    dtype=np.float64,
)
_XYZ_TO_REC2020 = np.asarray(
    [
        [30757411 / 17917100, -6372589 / 17917100, -4539589 / 17917100],
        [-19765991 / 29648200, 47925759 / 29648200, 467509 / 29648200],
        [792561 / 44930125, -1921689 / 44930125, 42328811 / 44930125],
    ],
    dtype=np.float64,
)


def _finite_rgb(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray):
        raise TypeError("pixels must be a numpy ndarray")
    if pixels.dtype != np.float32:
        raise TypeError("pixels must be float32")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("pixels must be HxWx3")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite")


def _linear_rec2020_values_to_oklab(values: np.ndarray) -> np.ndarray:
    xyz = np.matmul(np.asarray(values, dtype=np.float64), _REC2020_TO_XYZ.T)
    lms = np.matmul(xyz, _XYZ_TO_LMS.T)
    return np.matmul(np.cbrt(lms), _LMS_TO_OKLAB.T)


def linear_rec2020_to_oklab(pixels: np.ndarray) -> np.ndarray:
    """Convert finite linear Rec.2020 float32 pixels to float64 OKLab."""
    _finite_rgb(pixels)
    return _linear_rec2020_values_to_oklab(pixels)


def oklab_to_linear_rec2020(oklab: np.ndarray) -> np.ndarray:
    """Convert an (...,3) finite float64 OKLab array to linear Rec.2020."""
    values = np.asarray(oklab, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("oklab must end in three channels")
    if not np.isfinite(values).all():
        raise ValueError("oklab must be finite")
    lms_root = np.matmul(values, _OKLAB_TO_LMS.T)
    xyz = np.matmul(lms_root**3, _LMS_TO_XYZ.T)
    return np.matmul(xyz, _XYZ_TO_REC2020.T)


def _in_gamut(values: np.ndarray) -> np.ndarray:
    return np.all((values >= 0.0) & (values <= 1.0), axis=-1)


def _clip_with_delta(oklab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped_rgb = np.clip(oklab_to_linear_rec2020(oklab), 0.0, 1.0)
    clipped_oklab = _linear_rec2020_values_to_oklab(clipped_rgb)
    delta = np.linalg.norm(clipped_oklab - oklab, axis=-1)
    return clipped_rgb, delta


def local_minde_rec2020(
    pixels: np.ndarray,
    *,
    jnd: float = 0.02,
    epsilon: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray]:
    """Map extended linear Rec.2020 into gamut using W3C local-MINDE.

    Returns the mapped float32 image and the retained OKLCh chroma ratio. Pixels
    already inside Rec.2020 are copied bit-exactly and receive ratio one.
    """
    _finite_rgb(pixels)
    if not np.isfinite(jnd) or jnd <= 0.0:
        raise ValueError("jnd must be positive and finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    shape = pixels.shape
    source = pixels.reshape(-1, 3)
    output = source.astype(np.float64, copy=True)
    ratio = np.ones(source.shape[0], dtype=np.float64)
    source_in_gamut = _in_gamut(source)
    if bool(np.all(source_in_gamut)):
        return pixels.copy(), ratio.reshape(shape[:2]).astype(np.float32)

    indexes = np.flatnonzero(~source_in_gamut)
    origin = _linear_rec2020_values_to_oklab(source[indexes])
    lightness = origin[:, 0]
    chroma = np.hypot(origin[:, 1], origin[:, 2])
    unit = np.zeros((origin.shape[0], 2), dtype=np.float64)
    chromatic = chroma > 4e-6
    unit[chromatic] = origin[chromatic, 1:3] / chroma[chromatic, None]

    mapped_rgb = np.empty_like(origin)
    resolved = np.zeros(origin.shape[0], dtype=bool)
    low = lightness <= 0.0
    high = lightness >= 1.0
    mapped_rgb[low] = np.asarray([0.0, 0.0, 0.0])
    mapped_rgb[high] = np.asarray([1.0, 1.0, 1.0])
    ratio[indexes[low | high]] = 0.0
    resolved |= low | high

    work = ~resolved
    if bool(np.any(work)):
        initial_rgb, initial_delta = _clip_with_delta(origin[work])
        local = initial_delta < jnd
        work_indexes = np.flatnonzero(work)
        if bool(np.any(local)):
            mapped_rgb[work_indexes[local]] = initial_rgb[local]
            resolved[work_indexes[local]] = True

    active = np.flatnonzero(~resolved)
    minimum = np.zeros(active.size, dtype=np.float64)
    maximum = chroma[active].copy()
    min_in_gamut = np.ones(active.size, dtype=bool)
    last_clipped = np.zeros((active.size, 3), dtype=np.float64)
    has_clipped = np.zeros(active.size, dtype=bool)
    done = np.zeros(active.size, dtype=bool)

    # The loop count is a defensive ceiling; normal termination is epsilon.
    for _ in range(64):
        pending = (~done) & ((maximum - minimum) > epsilon)
        if not bool(np.any(pending)):
            break
        p = np.flatnonzero(pending)
        mid = (minimum[p] + maximum[p]) * 0.5
        current = np.empty((p.size, 3), dtype=np.float64)
        current[:, 0] = lightness[active[p]]
        current[:, 1:3] = unit[active[p]] * mid[:, None]
        current_rgb = oklab_to_linear_rec2020(current)
        current_in = _in_gamut(current_rgb)
        direct = min_in_gamut[p] & current_in
        if bool(np.any(direct)):
            minimum[p[direct]] = mid[direct]
        other = ~direct
        if not bool(np.any(other)):
            continue
        q = p[other]
        clipped_rgb, delta = _clip_with_delta(current[other])
        last_clipped[q] = clipped_rgb
        has_clipped[q] = True
        below = delta < jnd
        close = below & ((jnd - delta) < epsilon)
        if bool(np.any(close)):
            mapped_rgb[active[q[close]]] = clipped_rgb[close]
            done[q[close]] = True
        continue_below = below & ~close
        if bool(np.any(continue_below)):
            min_in_gamut[q[continue_below]] = False
            minimum[q[continue_below]] = mid[other][continue_below]
        above = ~below
        if bool(np.any(above)):
            maximum[q[above]] = mid[other][above]

    unresolved = np.flatnonzero(~done)
    if unresolved.size:
        if not bool(np.all(has_clipped[unresolved])):
            raise RuntimeError("local-MINDE ended without a clipped candidate")
        mapped_rgb[active[unresolved]] = last_clipped[unresolved]

    output[indexes] = mapped_rgb
    nonzero = chroma > 0.0
    mapped_oklab = _linear_rec2020_values_to_oklab(mapped_rgb)
    mapped_chroma = np.hypot(mapped_oklab[:, 1], mapped_oklab[:, 2])
    local_ratio = np.ones_like(chroma)
    local_ratio[nonzero] = mapped_chroma[nonzero] / chroma[nonzero]
    ratio[indexes] = np.clip(local_ratio, 0.0, 1.0)
    result = np.ascontiguousarray(output.reshape(shape), dtype=np.float32)
    if not np.isfinite(result).all() or not bool(np.all(_in_gamut(result))):
        raise RuntimeError("local-MINDE failed to produce finite in-gamut Rec.2020")
    return result, ratio.reshape(shape[:2]).astype(np.float32)


__all__ = [
    "linear_rec2020_to_oklab",
    "local_minde_rec2020",
    "oklab_to_linear_rec2020",
]

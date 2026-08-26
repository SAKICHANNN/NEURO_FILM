"""Private Adobe DNG ProfileHueSatMap reference-subdomain arithmetic."""

# ruff: noqa: B023

from __future__ import annotations

import math

import numpy as np


def _f32(value: float) -> np.float32:
    return np.float32(value)


def apply_profile_huesatmap(rgb: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Apply a linear-encoding HueSatMap using Adobe's reference operation order."""
    source = np.asarray(rgb, dtype=np.float32)
    lut = np.asarray(table, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("rgb must be finite Nx3 float32")
    if lut.ndim != 4 or lut.shape[3] != 3 or lut.shape[1] < 1 or lut.shape[2] < 2:
        raise ValueError("table must have shape VxHxSx3")
    if not np.isfinite(lut).all():
        raise ValueError("table must be finite")
    out = np.empty_like(source)
    v_div, h_div, s_div, _ = lut.shape
    h_scale = _f32(h_div / 6.0) if h_div >= 2 else _f32(0.0)
    s_scale = _f32(s_div - 1)
    v_scale = _f32(v_div - 1)

    for row, (r0, g0, b0) in enumerate(source):
        r, g, b = _f32(r0), _f32(g0), _f32(b0)
        value = max(r, g, b)
        gap = _f32(value - min(r, g, b))
        if gap > 0:
            if r == value:
                hue = _f32((g - b) / gap)
                if hue < 0:
                    hue = _f32(hue + _f32(6.0))
            elif g == value:
                hue = _f32(_f32(2.0) + _f32((b - r) / gap))
            else:
                hue = _f32(_f32(4.0) + _f32((r - g) / gap))
            saturation = _f32(gap / value)
        else:
            hue, saturation = _f32(0.0), _f32(0.0)

        hs = _f32(hue * h_scale)
        ss = _f32(saturation * s_scale)
        vs = _f32(value * v_scale)
        hi = min(max(int(hs), 0), h_div - 1)
        si = min(max(int(ss), 0), s_div - 2)
        vi = 0 if v_div < 2 else min(max(int(vs), 0), v_div - 2)
        hj = 0 if hi >= h_div - 1 else hi + 1
        hf, sf = _f32(hs - _f32(hi)), _f32(ss - _f32(si))
        vf = _f32(0.0) if v_div < 2 else _f32(vs - _f32(vi))
        h0, s0, v0 = _f32(1.0 - hf), _f32(1.0 - sf), _f32(1.0 - vf)

        def interp(component: int) -> np.float32:
            def hue_at(v_index: int, s_index: int) -> np.float32:
                return _f32(
                    _f32(h0 * lut[v_index, hi, s_index, component])
                    + _f32(hf * lut[v_index, hj, s_index, component])
                )

            if v_div < 2:
                low = hue_at(0, si)
                high = hue_at(0, si + 1)
            else:
                low = _f32(_f32(v0 * hue_at(vi, si)) + _f32(vf * hue_at(vi + 1, si)))
                high = _f32(
                    _f32(v0 * hue_at(vi, si + 1)) + _f32(vf * hue_at(vi + 1, si + 1))
                )
            return _f32(_f32(s0 * low) + _f32(sf * high))

        hue = _f32(hue + _f32(interp(0) * _f32(6.0 / 360.0)))
        saturation = min(_f32(saturation * interp(1)), _f32(1.0))
        value = min(max(_f32(value * interp(2)), _f32(0.0)), _f32(1.0))
        if saturation > 0:
            hue = _f32(math.fmod(float(hue), 6.0))
            if hue < 0:
                hue = _f32(hue + _f32(6.0))
            sector = int(hue)
            fract = _f32(hue - _f32(sector))
            p = _f32(value * _f32(1.0 - saturation))
            q = _f32(value * _f32(1.0 - _f32(saturation * fract)))
            t = _f32(value * _f32(1.0 - _f32(saturation * _f32(1.0 - fract))))
            choices = ((value, t, p), (q, value, p), (p, value, t), (p, q, value), (t, p, value), (value, p, q))
            out[row] = choices[sector % 6]
        else:
            out[row] = value
    return out


__all__ = ["apply_profile_huesatmap"]

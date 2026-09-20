from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit, logit

from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior


@dataclass(frozen=True)
class StatusMPositive:
    prior: ManufacturerCharacteristicPrior
    parameters: dict
    references: dict

    @classmethod
    def build(cls, prior: ManufacturerCharacteristicPrior, parameters: dict) -> "StatusMPositive":
        a = max(c.domain[0] for c in prior.curves)
        b = min(c.domain[1] for c in prior.curves)
        if b <= a:
            raise ValueError("empty common observed interval")
        eps = 2.0 ** -parameters["pseudo_stops"]
        g = parameters["gray"]
        eg = eps + (1 - eps) * g
        qg = a + (b - a) * (np.log2(eg) + parameters["pseudo_stops"]) / parameters["pseudo_stops"]
        rows = []
        for curve in prior.curves:
            d0, d1, dg = curve.apply(np.array([a, b, qg]))
            slopes = np.diff(curve.density_knots) / np.diff(curve.log_exposure_knots)
            i = int(np.searchsorted(curve.log_exposure_knots, qg, side="right") - 1)
            m = slopes[i]
            hits = np.flatnonzero(curve.log_exposure_knots == qg)
            if len(hits):
                j = int(hits[0])
                if j == 0 or j == len(curve.log_exposure_knots) - 1 or min(slopes[j - 1:j + 1]) <= 0:
                    raise ValueError("invalid pivot knot slopes")
                m = np.mean(slopes[j - 1:j + 1])
            if d1 <= d0 or m <= 0:
                raise ValueError("invalid density span or pivot slope")
            zg = (dg - d0) / (d1 - d0)
            if not 0 < zg < 1:
                raise ValueError("gray density outside endpoints")
            v = m / (d1 - d0) * (b - a) * (1 - eps) / (parameters["pseudo_stops"] * np.log(2) * eg)
            alpha = parameters["contrast"] * zg * (1 - zg) / (g * (1 - g) * v)
            if not parameters["alpha_bounds"][0] <= alpha <= parameters["alpha_bounds"][1]:
                raise ValueError("derived exponent outside admission bounds")
            rows.append(dict(channel=curve.layer, D0=float(d0), D1=float(d1), Dg=float(dg), zg=float(zg), v=float(v), alpha=float(alpha)))
        return cls(prior, parameters, dict(a=float(a), b=float(b), epsilon=eps, q_gray=float(qg), channels=rows))

    def pseudo_exposure(self, linear_rgb: np.ndarray) -> np.ndarray:
        x = np.asarray(linear_rgb, dtype=np.float64)
        if x.ndim < 2 or x.shape[-1] != 3 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError("expected finite linear RGB in [0,1]")
        eps = self.references["epsilon"]
        return eps + (1 - eps) * x

    def from_exposure(self, exposure: np.ndarray) -> np.ndarray:
        e = np.asarray(exposure, dtype=np.float64)
        a, b, eps = (self.references[k] for k in ("a", "b", "epsilon"))
        if not np.isfinite(e).all() or np.any(e < eps) or np.any(e > 1):
            raise ValueError("pseudo exposure outside observed-coordinate ingress")
        q = a + (b - a) * (np.log2(e) + self.parameters["pseudo_stops"]) / self.parameters["pseudo_stops"]
        d = self.prior.apply(q)
        out = np.empty_like(d)
        for c, ref in enumerate(self.references["channels"]):
            z = (d[..., c] - ref["D0"]) / (ref["D1"] - ref["D0"])
            if np.any(z < 0) or np.any(z > 1):
                raise ValueError("normalized density outside endpoints")
            interior = (z > 0) & (z < 1)
            y = np.zeros_like(z)
            y[z == 1] = 1
            y[interior] = expit(logit(self.parameters["gray"]) + ref["alpha"] * (logit(z[interior]) - logit(ref["zg"])))
            out[..., c] = y
        return out

    def apply(self, linear_rgb: np.ndarray, *, optics: bool = False) -> np.ndarray:
        exposure = self.pseudo_exposure(linear_rgb)
        if optics:
            exposure = redistribute_highlights(exposure, self.references["epsilon"], self.parameters["optics"])
        return self.from_exposure(exposure)


def redistribute_highlights(exposure: np.ndarray, epsilon: float, parameters: dict) -> np.ndarray:
    e = np.asarray(exposure, dtype=np.float64)
    if e.ndim != 3 or e.shape[-1] != 3 or not np.isfinite(e).all() or np.any(e < epsilon) or np.any(e > 1):
        raise ValueError("optics expects bounded HWC3 pseudo exposure")
    threshold = epsilon + (1 - epsilon) * parameters["threshold_linear"]
    excess = np.maximum(e - threshold, 0)
    sigma = max(parameters["minimum_sigma_pixels"], parameters["sigma_min_dimension_fraction"] * min(e.shape[:2]))
    blurred = gaussian_filter(excess, sigma=(sigma, sigma, 0), mode="reflect", truncate=parameters["truncate"])
    return e + np.asarray(parameters["channel_weights"]) * (blurred - excess)

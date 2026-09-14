import numpy as np
from scipy.special import logit


def canonical_measure(rgb: np.ndarray, *, epsilon: float, scale_floor: float, tensor_limit: float) -> dict:
    values = np.asarray(rgb, dtype=np.float64)
    if (values.ndim != 3 or values.shape[-1] != 3 or not values.size or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError('finite HWC RGB in [0,1] required')
    if not 0 <= epsilon < .5 or scale_floor <= 0 or tensor_limit <= 0:
        raise ValueError('invalid canonical measurement bounds')
    bounded = np.clip(values, epsilon, 1 - epsilon)
    if np.any(bounded <= 0) or np.any(bounded >= 1):
        raise ValueError('unclamped theory measurement requires interior pixels')
    logs = logit(bounded)
    mu = logs.mean(axis=(0, 1))
    centered = logs - mu
    raw_scale = np.sqrt(np.mean(centered ** 2))
    scale = max(float(raw_scale), scale_floor)
    unbounded_tensor = centered / scale
    return {'target': np.r_[mu, np.log(scale)], 'mu': mu, 'scale': scale,
            'tensor': np.clip(unbounded_tensor, -tensor_limit, tensor_limit),
            'clamp_fraction': float(np.mean(bounded != values)), 'scale_floored': bool(raw_scale < scale_floor),
            'tensor_clipping_fraction': float(np.mean(np.abs(unbounded_tensor) > tensor_limit))}


def decode_canonical_prior(after_measure: dict, predicted_canonical: np.ndarray, *,
                           slope_limit: float, offset_limit: float) -> dict:
    target = np.asarray(predicted_canonical, dtype=np.float64)
    if target.shape != (4,) or not np.isfinite(target).all() or slope_limit <= 1 or offset_limit <= 0:
        raise ValueError('four finite predicted statistics and valid transform bounds required')
    log_a = np.log(after_measure['scale']) - target[3]
    a = np.exp(log_a)
    b = after_measure['mu'] - a * target[:3]
    raw = np.r_[log_a / np.log(slope_limit), b / offset_limit]
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite analytic inversion')
    return {'raw': raw, 'u': np.clip(raw, -1, 1), 'clipped': np.abs(raw) > 1}

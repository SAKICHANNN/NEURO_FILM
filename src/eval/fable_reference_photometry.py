import numpy as np
from scipy.special import expit, logit


def quantize8(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('RGB must be finite and in [0,1]')
    return np.floor(255*values+.5)/255


def transform(rgb: np.ndarray, u: np.ndarray, *, slope_limit: float, offset_limit: float) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    parameters = np.asarray(u, dtype=np.float64)
    if values.shape[-1:] != (3,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Expected finite RGB in [0,1]')
    if parameters.shape != (4,) or not np.isfinite(parameters).all() or np.any(abs(parameters) > 1):
        raise ValueError('Expected four normalized parameters in [-1,1]')
    if not np.isfinite(slope_limit) or slope_limit <= 1 or not np.isfinite(offset_limit) or offset_limit <= 0:
        raise ValueError('Invalid transform bounds')
    if np.all(parameters == 0):
        return values.copy()
    a = np.exp(np.log(slope_limit)*parameters[0])
    output = values.copy()
    for c in range(3):
        plane = values[..., c]
        interior = (plane > 0) & (plane < 1)
        output[..., c][interior] = expit(a*logit(plane[interior])+offset_limit*parameters[c+1])
    return output


def dequantize8(canonical: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(canonical, dtype=np.float64)
    if not np.array_equal(quantize8(values), values):
        raise ValueError('Expected canonical eight-bit codes')
    codes = np.floor(values*255+.5)
    low = np.maximum(0, (codes-.5)/255)
    high = np.minimum(1, (codes+.5)/255)
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(low, high)


def antithetic_draws(*, pairs: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    positive = rng.uniform(-1, 1, (pairs, 4))
    return np.stack((positive, -positive), axis=1).reshape(2*pairs, 4)


def canonical_donor(rgb: np.ndarray, *, side: int) -> np.ndarray:
    import torch
    from torch.nn.functional import interpolate

    values = np.asarray(rgb, dtype=np.float32)
    tensor = torch.from_numpy(values).permute(2, 0, 1)[None]
    resized = interpolate(tensor, (side, side), mode='bilinear', align_corners=False, antialias=True)
    return quantize8(resized[0].permute(1, 2, 0).numpy().astype(np.float64))

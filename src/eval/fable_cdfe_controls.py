import numpy as np

from src.eval.fable_canonical_prior import decode_canonical_prior


def decode_assessment_controls(predicted_canonical: np.ndarray, after_mu: np.ndarray,
                               after_scale: np.ndarray, *, fitting_mean: np.ndarray,
                               donor_ids: list[str], cameras: list[str],
                               shuffle_mapping: dict[str, str], slope_limit: float,
                               offset_limit: float) -> dict:
    predicted = np.asarray(predicted_canonical, dtype=np.float64)
    mu, scale, mean = [np.asarray(x, dtype=np.float64) for x in (after_mu, after_scale, fitting_mean)]
    if (predicted.shape != (32, 32, 4) or mu.shape != (32, 32, 3)
            or scale.shape != (32, 32) or mean.shape != (4,)
            or any(not np.isfinite(x).all() for x in (predicted, mu, scale, mean))
            or np.any(scale <= 0)):
        raise ValueError('finite complete assessment statistics required')
    if (len(donor_ids) != 32 or len(set(donor_ids)) != 32 or len(cameras) != 32
            or set(shuffle_mapping) != set(donor_ids)
            or set(shuffle_mapping.values()) != set(donor_ids)):
        raise ValueError('complete bijective donor mapping required')
    positions = {identity: j for j, identity in enumerate(donor_ids)}
    shuffled = [positions[shuffle_mapping[identity]] for identity in donor_ids]
    if any(k == j or cameras[k] != cameras[j] for j, k in enumerate(shuffled)):
        raise ValueError('fixed-point-free within-camera control required')
    outputs = {name: np.empty_like(predicted) for name in ['learned', 'constant', 'shuffled']}
    for j, other in enumerate(shuffled):
        for t in range(32):
            measured = {'mu': mu[j, t], 'scale': scale[j, t]}
            targets = {'learned': predicted[j, t], 'constant': mean, 'shuffled': predicted[other, t]}
            for name, target in targets.items():
                outputs[name][j, t] = decode_canonical_prior(measured, target,
                    slope_limit=slope_limit, offset_limit=offset_limit)['u']
    return outputs

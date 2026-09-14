from collections.abc import Iterator

import numpy as np

from src.eval.fable_canonical_inference import native_canonical_input
from src.eval.fable_canonical_prior import canonical_measure
from src.eval.fable_reference_photometry import dequantize8, quantize8, transform
from src.preprocess.fable_canonical_raw import linear16_to_q8


def donor_training_examples(linear: np.ndarray, treatments: list[dict], *, seed: int,
                            measurement: dict, input_size: int,
                            slope_limit: float, offset_limit: float) -> Iterator[dict]:
    canonical = linear16_to_q8(linear).astype(np.float64) / 255
    target = canonical_measure(canonical, **measurement)['target'].copy()
    latent = dequantize8(canonical, seed=seed)
    del canonical
    for treatment in treatments:
        after = quantize8(transform(latent, np.asarray(treatment['u']),
                                   slope_limit=slope_limit, offset_limit=offset_limit))
        observed = native_canonical_input(after, input_size=input_size, **measurement)
        yield {'treatment_id': treatment['id'], 'input': observed['tensor'],
               'target': target.copy(), 'after_mu': observed['mu'],
               'after_scale': observed['scale'], 'measurement_shape': observed['measurement_shape']}


def fitting_target_normalizer(targets: np.ndarray, identities: list[str], *,
                              expected_identities: list[str], minimum_scale: float) -> dict:
    values = np.asarray(targets, dtype=np.float64)
    if (len(identities) != len(set(identities)) or identities != expected_identities
            or values.shape != (len(identities), 4) or not len(identities)
            or not np.isfinite(values).all() or minimum_scale <= 0):
        raise ValueError('unique ordered fitting targets matching locked identities required')
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= minimum_scale):
        raise ValueError('degenerate fitting target scale; stop without replacing data')
    return {'mean': mean, 'scale': scale, 'ddof': 0, 'identities': list(identities)}

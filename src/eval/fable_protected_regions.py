import numpy as np

from src.eval.fable_photometry_metrics import histogram_errors


def protected_region_counts(source_codes: np.ndarray, regions: dict[str, np.ndarray]) -> dict:
    codes = np.asarray(source_codes)
    if codes.dtype != np.uint8 or codes.ndim != 3 or codes.shape[-1] != 3 or not codes.size:
        raise ValueError('nonempty native uint8 HWC RGB required')
    if not regions:
        raise ValueError('at least one named protected region required')
    result = {}
    for name, mask in regions.items():
        mask = np.asarray(mask)
        if not isinstance(name, str) or not name.strip() or mask.dtype != bool or mask.shape != codes.shape[:2] or not mask.any():
            raise ValueError('nonempty named boolean masks aligned with native pixels required')
        result[name] = np.stack([np.bincount(codes[..., c][mask], minlength=256) for c in range(3)])
    return result


def protected_region_accuracy(region_counts: dict, prediction_tables: np.ndarray,
                              oracle_tables: np.ndarray) -> dict:
    predictions, oracles = [np.asarray(t, dtype=np.float64) for t in (prediction_tables, oracle_tables)]
    if (predictions.ndim != 3 or predictions.shape[1:] != (256, 3) or not len(predictions)
            or predictions.shape != oracles.shape or not region_counts):
        raise ValueError('nonempty named counts and aligned (cases, 256, 3) tables required')
    identity = np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)
    results = {}
    for name, counts in region_counts.items():
        counts = np.asarray(counts)
        if (not isinstance(name, str) or not name.strip() or counts.shape != (3, 256)
                or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)
                or counts.sum() <= 0 or not np.all(counts.sum(1) == counts.sum(1)[0])):
            raise ValueError('each region needs aligned nonnegative integer RGB counts')
        d = [histogram_errors(o, identity, counts)['code_mse'] for o in oracles]
        e = [histogram_errors(p, o, counts)['code_mse'] for p, o in zip(predictions, oracles, strict=True)]
        mean_d, mean_e = float(np.mean(d)), float(np.mean(e))
        results[name] = {'per_case_D': d, 'per_case_E': e, 'D': mean_d, 'E': mean_e,
                         'passed': mean_e <= .25 * mean_d}
    return {'regions': results, 'passed': all(r['passed'] for r in results.values()),
            'scope': 'CHANNELWISE_Q8_REGION_ACCURACY_ONLY_NOT_SEMANTIC_PROOF',
            'aggregation': 'equal cases within each region; conjunction across regions'}

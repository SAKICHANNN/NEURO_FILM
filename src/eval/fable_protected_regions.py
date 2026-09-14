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


def saturation_excess(source_counts: np.ndarray, prediction_tables: np.ndarray,
                      oracle_tables: np.ndarray) -> dict:
    counts = np.asarray(source_counts)
    predictions, oracles = [np.asarray(t, dtype=np.float64) for t in (prediction_tables, oracle_tables)]
    if (counts.shape != (3, 256) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or counts.sum() <= 0
            or not np.all(counts.sum(1) == counts.sum(1)[0])):
        raise ValueError('aligned nonnegative integer native RGB counts required')
    if (predictions.ndim != 3 or predictions.shape[1:] != (256, 3) or not len(predictions)
            or predictions.shape != oracles.shape
            or any(not np.isfinite(t).all() or np.any((t < 0) | (t > 1)) for t in (predictions, oracles))):
        raise ValueError('nonempty aligned finite bounded channel tables required')
    eligible = int(counts[:, 1:255].sum())
    total = int(counts.sum())
    cases = []
    for prediction, oracle in zip(predictions, oracles, strict=True):
        endpoint_counts = []
        for table in (prediction, oracle):
            codes = np.floor(255 * table[1:255] + .5)
            endpoint_counts.append(int(np.sum(counts[:, 1:255].T * ((codes == 0) | (codes == 255)))))
        candidate_count, oracle_count = endpoint_counts
        excess = candidate_count - oracle_count
        cases.append({'candidate_count': candidate_count, 'oracle_count': oracle_count,
                      'source_eligible': eligible, 'source_total': total,
                      'candidate_fraction_of_eligible': candidate_count / eligible if eligible else None,
                      'oracle_fraction_of_eligible': oracle_count / eligible if eligible else None,
                      'excess_fraction_of_eligible': excess / eligible if eligible else None,
                      'candidate_fraction_of_total': candidate_count / total,
                      'oracle_fraction_of_total': oracle_count / total,
                      'applicable': eligible > 0,
                      'passed': 200 * excess <= eligible})
    return {'cases': cases, 'passed': all(c['passed'] for c in cases),
            'denominator': 'previously interior source channel values',
            'maximum_excess': .005, 'aggregation': 'conjunction across cases; no averaging',
            'scope': 'CHANNELWISE_Q8_SATURATION_ONLY_NOT_SEMANTIC_PROOF'}

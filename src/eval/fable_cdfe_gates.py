import math

import numpy as np


def stratified_photometry_gates(error: np.ndarray, ideal_change: np.ndarray, output_change: np.ndarray,
                                *, donor_ids: list[str], query_ids: list[str],
                                donor_cameras: list[str], treatment_regions: list[str]) -> dict:
    error, ideal_change, output_change = [np.asarray(x, dtype=np.float64)
                                         for x in (error, ideal_change, output_change)]
    if any(x.shape != (32, 32, 4) or not np.isfinite(x).all() or np.any(x < 0)
           for x in (error, ideal_change, output_change)):
        raise ValueError('finite nonnegative32x32x4 case metrics required')
    if (len(donor_ids) != 32 or len(set(donor_ids)) != 32 or len(query_ids) != 4
            or len(set(query_ids)) != 4 or len(donor_cameras) != 32
            or len(treatment_regions) != 32 or len(set(treatment_regions)) != 4
            or any(not isinstance(x, str) or not x for x in donor_ids+query_ids+donor_cameras+treatment_regions)):
        raise ValueError('complete donor/query/camera/four-region labels required')

    def accuracy(e, d):
        e, d = float(np.mean(e)), float(np.mean(d))
        return {'E': e, 'D': d, 'passed': e <= .25*d}

    pooled = accuracy(error, ideal_change)
    pooled['P'] = float(output_change.mean())
    pooled['passed'] = pooled['passed'] and pooled['P'] >= 16 and pooled['D'] > 0
    donors = {i: accuracy(error[j], ideal_change[j]) for j, i in enumerate(donor_ids)}
    for donor in donors.values():
        donor['passed'] = donor['passed'] and donor['D'] > 0
    cameras = {}
    for camera in sorted(set(donor_cameras)):
        selected = np.array([c == camera for c in donor_cameras])
        cameras[camera] = accuracy(error[selected], ideal_change[selected])
    regions = {}
    for region in sorted(set(treatment_regions)):
        selected = np.array([r == region for r in treatment_regions])
        regions[region] = accuracy(error[:, selected], ideal_change[:, selected])
    queries = {}
    for j, identity in enumerate(query_ids):
        e, d, p = [float(x[:, :, j].mean()) for x in (error, ideal_change, output_change)]
        queries[identity] = {'E': e, 'D': d, 'P': p, 'passed': e < d and p >= 16}
    groups = [donors, cameras, regions, queries]
    return {'pooled': pooled, 'donors': donors, 'cameras': cameras, 'regions': regions, 'queries': queries,
            'photometry_passed': pooled['passed'] and all(r['passed'] for group in groups for r in group.values()),
            'limits': 'D>0 enforced pooled and per donor; controls, ROI, saturation and comfort remain separate.'}


def validation_gate(predicted: np.ndarray, targets: np.ndarray, constant: np.ndarray) -> dict:
    predicted, targets, constant = [np.asarray(x, dtype=np.float64) for x in (predicted, targets, constant)]
    if (predicted.shape != (32, 8, 4) or targets.shape != predicted.shape or constant.shape != (4,)
            or any(not np.isfinite(x).all() for x in (predicted, targets, constant))):
        raise ValueError('finite standardized 32 donor x8 treatment x4 targets required')
    mse = float(np.mean((predicted-targets)**2))
    baseline = float(np.mean((constant-targets)**2))
    return {'mse': mse, 'constant_mse': baseline,
            'status': 'DEGENERATE_CONSTANT_BASELINE' if baseline == 0 else ('PASS' if mse <= .8*baseline else 'FAIL'),
            'passed': baseline > 0 and mse <= .8*baseline}


def donor_control_gates(learned: np.ndarray, constant: np.ndarray, shuffled: np.ndarray,
                        ideal_change: np.ndarray) -> dict:
    arrays = [np.asarray(x, dtype=np.float64) for x in (learned, constant, shuffled, ideal_change)]
    if any(x.shape != (32, 32, 4) or not np.isfinite(x).all() or np.any(x < 0) for x in arrays):
        raise ValueError('nonnegative finite32 donor x32 treatment x4 query code-MSE arrays required')
    learned, constant, shuffled, ideal_change = arrays
    donor_learned = learned.mean(axis=(1, 2))
    d = float(ideal_change.mean())
    controls = {}
    for name, errors in [('constant', constant), ('shuffled_C', shuffled)]:
        wins = int(np.sum(donor_learned < errors.mean(axis=(1, 2))))
        numerator = sum(math.comb(32, k) for k in range(wins, 33))
        margin = float(errors.mean()-learned.mean())
        sign_pass = 40*numerator <= 2**32
        controls[name] = {'donor_wins': wins, 'ties_are_nonwins': True,
                          'one_sided_p': numerator/2**32, 'alpha': .025,
                          'sign_passed': sign_pass, 'pooled_margin': margin,
                          'required_strict_margin': .1*d, 'margin_passed': margin > .1*d,
                          'passed': sign_pass and margin > .1*d}
    return {'controls': controls, 'passed': all(x['passed'] for x in controls.values()),
            'unit': 'donor mean, conditional independent-block assumption; not a camera-population test'}

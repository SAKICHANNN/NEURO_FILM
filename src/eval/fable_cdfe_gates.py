import math

import numpy as np


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

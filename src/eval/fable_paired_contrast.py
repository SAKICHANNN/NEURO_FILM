from math import comb

import numpy as np
from scipy import linalg

from src.eval.fable_photometry_ridge import crossed_folds


def _blocks(blocks):
    values = [np.asarray(block, dtype=np.float64) for block in blocks]
    if (len(values) != 2 or any(v.ndim != 2 or not v.size or not np.isfinite(v).all() for v in values)
            or len(values[0]) != len(values[1])):
        raise ValueError('two finite nonempty row-aligned raw descriptor blocks required')
    return values


def _standardize(blocks, standards):
    values = _blocks(blocks)
    matrices = []
    for value, standard in zip(values, standards, strict=True):
        if value.shape[1] != standard['dimension']:
            raise ValueError('raw feature dimension mismatch')
        active = standard['active']
        matrices.append((value[:, active] - standard['mean'][active])
                        / standard['scale'][active] / standard['divisor'])
    return np.concatenate(matrices, axis=1)


def fit_contrast_decoder(after_raw_blocks: list[np.ndarray], identity_raw_blocks: list[np.ndarray],
                         targets: np.ndarray, donor_ids: np.ndarray,
                         identity_donor_ids: np.ndarray, ridge: float) -> dict:
    after, identity = _blocks(after_raw_blocks), _blocks(identity_raw_blocks)
    targets = np.asarray(targets, dtype=np.float64)
    donors, identity_ids = np.asarray(donor_ids), np.asarray(identity_donor_ids)
    if (targets.shape != (len(after[0]), 4) or not np.isfinite(targets).all() or np.any(np.abs(targets) > 1)
            or donors.shape != (len(after[0]),) or identity_ids.shape != (len(identity[0]),)
            or len(np.unique(identity_ids)) != len(identity_ids)
            or not np.array_equal(np.sort(identity_ids), np.unique(donors))
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('aligned normalized targets, one identity per training donor, and positive ridge required')
    standards = []
    for edited, original in zip(after, identity, strict=True):
        if edited.shape[1] != original.shape[1]:
            raise ValueError('identity and AFTER raw dimensions differ')
        raw = np.concatenate([edited, original], axis=0)
        mean, scale = raw.mean(0), raw.std(0)
        active = (np.ptp(raw, axis=0) > 0) & (scale > 0)
        standards.append({'mean': mean, 'scale': scale, 'active': active,
                          'dimension': raw.shape[1],
                          'divisor': float(np.sqrt(active.sum())) if active.any() else 1.})
    z_after, z_identity = _standardize(after, standards), _standardize(identity, standards)
    identity_rows = {key: i for i, key in enumerate(identity_ids.tolist())}
    contrast = z_after - z_identity[[identity_rows[key] for key in donors.tolist()]]
    mean_contrast, mean_target = contrast.mean(0), targets.mean(0)
    centered = contrast - mean_contrast
    # Contrast remains in raw-normalizer units; no second feature scaling.
    gram = centered @ centered.T + len(targets) * ridge * np.eye(len(targets))
    coefficients = centered.T @ linalg.cho_solve(linalg.cho_factor(gram, lower=True), targets - mean_target)
    return {'standards': standards, 'coefficients': coefficients,
            'intercept': mean_target - mean_contrast @ coefficients,
            'mean_identity_z': z_identity.mean(0), 'ridge': float(ridge),
            'regression_rows': len(targets), 'normalization_rows': len(targets) + len(identity_ids),
            'identity_rows': len(identity_ids), 'identity_donor_ids': identity_ids.copy(),
            'interface': 'AFTER_RAW_BLOCKS_WITH_FROZEN_MEAN_IDENTITY'}


def _prediction(model, contrasts):
    raw = contrasts @ model['coefficients'] + model['intercept']
    clipped = np.abs(raw) > 1
    return {'raw': raw, 'u': np.clip(raw, -1, 1), 'clipped': clipped,
            'clipping_frequency': float(clipped.mean()),
            'row_clipping_frequency': float(clipped.any(1).mean())}


def predict_after_raw_blocks(model: dict, after_raw_blocks: list[np.ndarray]) -> dict:
    return _prediction(model, _standardize(after_raw_blocks, model['standards']) - model['mean_identity_z'])


def predict_paired_control_raw_blocks(model: dict, after_raw_blocks: list[np.ndarray],
                                      before_raw_blocks: list[np.ndarray]) -> dict:
    after = _standardize(after_raw_blocks, model['standards'])
    before = _standardize(before_raw_blocks, model['standards'])
    if after.shape != before.shape:
        raise ValueError('paired control requires one BEFORE per AFTER')
    return _prediction(model, after - before)


def select_paired_contrast(after_raw_blocks: list[np.ndarray], identity_raw_blocks: list[np.ndarray],
                           targets: np.ndarray, donor_ids: np.ndarray, pair_ids: np.ndarray,
                           ridge_grid: np.ndarray) -> dict:
    after, identity = _blocks(after_raw_blocks), _blocks(identity_raw_blocks)
    targets = np.asarray(targets, dtype=np.float64)
    donors, pairs = np.asarray(donor_ids), np.asarray(pair_ids)
    folds = crossed_folds(donors, pairs)
    grid = np.asarray(ridge_grid, dtype=np.float64)
    if (len(after[0]) != 192 or len(identity[0]) != 6 or targets.shape != (192, 4)
            or not np.isfinite(targets).all() or np.any(np.abs(targets) > 1)
            or grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or np.any(grid <= 0) or len(np.unique(grid)) != len(grid)):
        raise ValueError('192 AFTER rows, six ordered identities, normalized targets and positive distinct grid required')
    for pair in range(16):
        reference = targets[(donors == 0) & (pairs == pair)]
        if not np.array_equal(reference[0], -reference[1]):
            raise ValueError('targets must be exact antithetic pairs')
        for donor in range(1, 6):
            current = targets[(donors == donor) & (pairs == pair)]
            if not (np.array_equal(current, reference) or np.array_equal(current[::-1], reference)):
                raise ValueError('targets must agree across donors')
    errors = np.empty((len(grid), 6))
    out_of_fold = np.empty((len(grid), 192, 4))
    fold_audit = []
    for column, fold in enumerate(folds):
        train, validation = fold['train'], fold['validation']
        fit_ids = np.unique(donors[train])
        for row, penalty in enumerate(grid):
            model = fit_contrast_decoder([v[train] for v in after], [v[fit_ids] for v in identity],
                                         targets[train], donors[train], fit_ids, penalty)
            predicted = predict_paired_control_raw_blocks(model, [v[validation] for v in after],
                                                           [v[donors[validation]] for v in identity])
            out_of_fold[row, validation] = predicted['u']
            errors[row, column] = np.mean((predicted['u'] - targets[validation]) ** 2)
        fold_audit.append({**fold, 'normalization_after_rows': train.copy(),
                           'normalization_identity_donors': fit_ids, 'normalization_rows': 68})
    means = errors.mean(1)
    selected = min(range(len(grid)), key=lambda i: (means[i], -grid[i]))
    model = fit_contrast_decoder(after, identity, targets, donors, np.arange(6), grid[selected])
    return {'model': model, 'grid': grid.copy(), 'fold_errors': errors, 'mean_errors': means,
            'selected_index': selected, 'selected_ridge': float(grid[selected]),
            'paired_oof_u': out_of_fold[selected], 'folds': fold_audit,
            'selection': 'PAIRED_CV_ONLY_EXACT_TIES_LARGER_RIDGE'}


def strict_win_sign_test_16(margins: np.ndarray, alpha: float = .05) -> dict:
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError('exactly 16 finite independent pair margins and valid alpha required')
    wins = int(np.count_nonzero(values > 0))
    probability = sum(comb(16, k) for k in range(wins, 17)) / 65536
    mean = float(values.mean())
    return {'wins': wins, 'nonwins': 16 - wins, 'ties': int(np.count_nonzero(values == 0)),
            'blocks': 16, 'pvalue': probability, 'mean': mean, 'alpha': float(alpha),
            'passed': probability <= alpha and mean > 0}


def joint_code_engineering_gate(oracle_strength: np.ndarray, prediction_error: np.ndarray,
                                prediction_strength: np.ndarray, donor_ids: np.ndarray,
                                query_ids: np.ndarray) -> dict:
    d, e, p = [np.asarray(v, dtype=np.float64) for v in (oracle_strength, prediction_error, prediction_strength)]
    donors, queries = np.asarray(donor_ids), np.asarray(query_ids)
    if (d.ndim != 1 or not len(d) or any(v.shape != d.shape for v in (e, p, donors, queries))
            or any(not np.isfinite(v).all() or np.any(v < 0) for v in (d, e, p))):
        raise ValueError('aligned finite nonnegative per-case code squared errors required')

    def accuracy(mask):
        strength, error = float(d[mask].mean()), float(e[mask].mean())
        return {'D': strength, 'E': error, 'relative_rms_error': np.sqrt(error / strength) if strength > 0 else None,
                'passed': strength > 0 and error <= .25 * strength}

    def strength(mask):
        energy = float(p[mask].mean())
        return {'P': energy, 'rms_code': np.sqrt(energy), 'passed': energy >= 16.}

    accuracy_results = {'pooled': accuracy(np.ones(len(d), dtype=bool)),
                        'per_donor': {str(key): accuracy(donors == key) for key in np.unique(donors)}}
    strength_results = {'pooled': strength(np.ones(len(d), dtype=bool)),
                        'per_query': {str(key): strength(queries == key) for key in np.unique(queries)}}
    query_improvement = {str(key): float(np.mean(d[queries == key] - e[queries == key])) for key in np.unique(queries)}
    passed = (accuracy_results['pooled']['passed'] and strength_results['pooled']['passed']
              and all(r['passed'] for r in accuracy_results['per_donor'].values())
              and all(r['passed'] for r in strength_results['per_query'].values())
              and all(v > 0 for v in query_improvement.values()))
    return {'accuracy': accuracy_results, 'strength': strength_results,
            'per_query_improvement_over_identity': query_improvement, 'passed': passed,
            'scope': 'ENGINEERING_CONDITIONS_ONLY_NOT_THE_FOUR_SIGN_TESTS'}

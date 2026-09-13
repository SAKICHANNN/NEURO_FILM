import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_reference_photometry import quantize8, transform
from src.eval.fable_photometry_metrics import (
    analytic_prior, donor_error_decomposition, exact_sign_test, histogram_detail_loss,
    histogram_errors, image_histograms, parameter_errors,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def mean_render_tables(fit_tables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(fit_tables, axis=0), quantize8(np.mean(quantize8(fit_tables), axis=0))


def wrong_reference_indices(rows: list[dict]) -> np.ndarray:
    keys = [(r['cell'], r['donor_id'], r['repeat'], r['draw_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate reference identity')
    lookup = {key: i for i, key in enumerate(keys)}
    indices = []
    for row, key in zip(rows, keys, strict=True):
        other = lookup.get((*key[:3], key[3] ^ 1))
        if other is None or row['pair_id'] != row['draw_id'] // 2:
            raise ValueError('missing same-cell same-donor antithetic reference')
        if not np.array_equal(np.asarray(row['target']), -np.asarray(rows[other]['target'])):
            raise ValueError('wrong reference target is not exact antithetic alternative')
        indices.append(other)
    return np.asarray(indices)


def improvement_summary(records: list[dict], improvements: np.ndarray, alpha: float,
                        require_subgroups: bool = True) -> dict:
    values = np.asarray(improvements, dtype=np.float64)
    if values.shape != (len(records),) or not np.isfinite(values).all():
        raise ValueError('one finite improvement per record required')
    pairs = np.asarray([r['pair_id'] for r in records])
    if not np.array_equal(np.unique(pairs), np.arange(16)):
        raise ValueError('exactly the 16 frozen independent pairs required')
    sizes = [np.count_nonzero(pairs == pair) for pair in range(16)]
    if len(set(sizes)) != 1:
        raise ValueError('unbalanced pair blocks')
    blocks = np.asarray([values[pairs == pair].mean() for pair in range(16)])
    result = exact_sign_test(blocks, alpha)
    result['block_improvements'] = blocks
    result['observations_per_block'] = sizes[0]
    subgroups = {}
    for field in ('donor_id', 'repeat', 'query_id'):
        if field not in records[0]:
            continue
        identifiers = np.asarray([r[field] for r in records])
        subgroups[field] = {str(int(key)): float(values[identifiers == key].mean()) for key in np.unique(identifiers)}
    result['subgroup_means'] = subgroups
    subgroup_pass = all(value > 0 for field in ('donor_id', 'repeat')
                        for value in subgroups.get(field, {}).values())
    result['subgroups_positive'] = subgroup_pass
    result['passed'] = result['passed'] and (subgroup_pass or not require_subgroups)
    return result


def primary_decision(cases: list[dict], parameter_rows: list[dict], primary: str, alpha: float) -> dict:
    keys = [(r['draw_id'], r['donor_id'], r['repeat'], r['query_id']) for r in cases]
    expected = {(draw, donor, repeat, query) for draw in range(32) for donor in (6, 7)
                for repeat in (0, 1) for query in (8, 9, 10, 11)}
    if len(keys) != 512 or set(keys) != expected:
        raise ValueError('primary cases must contain the complete 512-case crossed design')
    parameter_keys = [(r['draw_id'], r['donor_id'], r['repeat']) for r in parameter_rows]
    if len(parameter_keys) != 128 or set(parameter_keys) != {key[:3] for key in expected}:
        raise ValueError('primary parameters must contain the complete 128-reference design')
    required = {'parameter_vs_identity': improvement_summary(parameter_rows,
        [r['errors']['identity'] - r['errors'][primary] for r in parameter_rows], alpha, False)}
    additional = {}
    for precision in ('float_mse', 'code_mse'):
        for comparator in ('identity', 'mean_render', 'fixed_contrast', 'wrong_primary', 'analytic', 'stats'):
            improvements = [r['arms'][comparator][precision] - r['arms'][primary][precision] for r in cases]
            result = improvement_summary(cases, improvements, alpha)
            destination = additional if comparator in ('analytic', 'stats') else required
            destination[f'{precision}_vs_{comparator}'] = result
    for comparator in ('analytic', 'stats'):
        additional[f'parameter_vs_{comparator}'] = improvement_summary(parameter_rows,
            [r['errors'][comparator] - r['errors'][primary] for r in parameter_rows], alpha, False)
    passed = all(result['passed'] for result in required.values())
    return {'reference_contribution': 'PASS' if passed else 'NOT_ESTABLISHED_STOP',
            'required': required, 'added_value_diagnostics_not_gates': additional,
            'claim': 'Fixed synthetic mechanism only; appearance and learned/feature value are separate.'}


def parameter_diagnostics(rows, predictions):
    targets = np.asarray([r['target'] for r in rows])
    estimates = {**{name: predictions[name] for name in ('stats', 'combined', 'analytic')},
                 'identity': np.zeros_like(targets)}
    per_row = []
    for i, row in enumerate(rows):
        per_row.append({**row, 'errors': {name: float(np.mean((values[i] - targets[i]) ** 2))
                                        for name, values in estimates.items()}})
    cells = {}
    for cell in dict.fromkeys(r['cell'] for r in rows):
        indices = [i for i, r in enumerate(rows) if r['cell'] == cell]
        donors = sorted({rows[i]['donor_id'] for i in indices})
        ordered = [[next(i for i in indices if rows[i]['donor_id'] == donor and rows[i]['draw_id'] == draw)
                    for draw in range(32)] for donor in donors]
        cells[cell] = {}
        for name, values in estimates.items():
            summary = parameter_errors(values[indices], targets[indices])
            summary['donor_decomposition'] = donor_error_decomposition(values[ordered], targets[ordered[0]])
            summary['per_donor'] = {str(d): parameter_errors(values[group], targets[group])
                                    for d, group in zip(donors, ordered, strict=True)}
            if name in ('stats', 'combined'):
                summary['clipping_frequency'] = float(predictions[name + '_clipped'][indices].mean())
            cells[cell][name] = summary
    return per_row, cells


def verify_inputs(root: Path) -> dict:
    cp = root / 'configs/fable_reference_photometry_v1.json'
    cfg = json.loads(cp.read_text())
    output = root / cfg['output']
    held = json.loads((output / 'held_features/report.json').read_text())
    receipt = json.loads((output / 'held_predictions/report.json').read_text())
    fit = json.loads((output / 'fit_features/report.json').read_text())
    model_path = output / 'fit_head/model.json'
    model = json.loads(model_path.read_text())
    if (model['status'] != 'PRIMARY_FROZEN_BEFORE_ASSESSMENT'
            or receipt['status'] != 'PREDICTIONS_FROZEN_NO_REFIT'):
        raise ValueError('expected frozen model and prediction receipts')
    pins = {
        cp: held['config_sha256'], model_path: receipt['model_sha256'],
        output / 'held_features/features.npz': receipt['held_features_sha256'],
        output / 'held_features/rows.json': receipt['rows_sha256'],
        output / 'held_predictions/predictions.npz': receipt['predictions_sha256'],
        output / 'fit_features/features.npz': fit['features_sha256'],
        output / 'draws.json': fit['draws_sha256'],
        root / cfg['sources_config']: cfg['sources_config_sha256'],
        root / 'src/eval/fable_reference_photometry.py': fit['transform_module_sha256'],
        root / 'src/eval/fable_photometry_metrics.py': receipt['metrics_module_sha256'],
        root / 'src/eval/fable_photometry_ridge.py': model['ridge_module_sha256'],
    }
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f'hash mismatch: {path}')
    if (held['config_sha256'] != fit['config_sha256'] or held['model_sha256'] != receipt['model_sha256']
            or held['features_sha256'] != receipt['held_features_sha256'] or held['rows_sha256'] != receipt['rows_sha256']
            or model['features_sha256'] != fit['features_sha256'] or model['primary'] != receipt['primary']):
        raise ValueError('inconsistent provenance receipts')
    sources = json.loads((root / cfg['sources_config']).read_text())['sources']
    for source in sources:
        path = root / source['path']
        if digest(path) != source['sha256']:
            raise ValueError(f'source hash mismatch: {path}')
        pins[path] = source['sha256']
    rows = json.loads((output / 'held_features/rows.json').read_text())
    draws = json.loads((output / 'draws.json').read_text())
    expected_rows = set()
    for cell, donors, partition, repeat in (
            ('parameter_held', range(6), 'assessment', 0), ('donor_held', (6, 7), 'fit', 0),
            ('both_held', (6, 7), 'assessment', 0), ('both_held_repeat', (6, 7), 'assessment', 1)):
        expected_rows.update((cell, d, draw, repeat, partition) for d in donors for draw in range(32))
    keys = [(r['cell'], r['donor_id'], r['draw_id'], r['repeat'], r['partition']) for r in rows]
    if len(rows) != 384 or set(keys) != expected_rows:
        raise ValueError('held reference layout mismatch')
    for row in rows:
        if not np.array_equal(row['target'], draws[row['partition']][row['draw_id']]):
            raise ValueError('reference labels differ from frozen draws')
    wrong = wrong_reference_indices(rows)
    with np.load(output / 'held_predictions/predictions.npz', allow_pickle=False) as data:
        predictions = {key: data[key].copy() for key in data.files}
    for name in ('stats', 'combined', 'analytic'):
        if (predictions[name].shape != (384, 4) or not np.isfinite(predictions[name]).all()
                or np.any(np.abs(predictions[name]) > 1)):
            raise ValueError('invalid frozen predictions')
    return {'cfg': cfg, 'output': output, 'rows': rows, 'draws': draws, 'predictions': predictions,
            'primary': model['primary'], 'sources': sources, 'wrong': wrong,
            'verified_pins': {str(path.relative_to(root)): value for path, value in pins.items()}}


def main():
    state = verify_inputs(ROOT)
    cfg, rows, predictions = state['cfg'], state['rows'], state['predictions']
    out = state['output'] / 'assessment'
    out.mkdir(exist_ok=False)
    with np.load(state['output'] / 'fit_features/features.npz', allow_pickle=False) as data:
        prior = analytic_prior(data['canonical_bases'])
    table_input = np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)

    def table(u):
        return transform(table_input, np.asarray(u), slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit'])

    fit_tables = np.stack([table(u) for u in state['draws']['fit']])
    mean_float, mean_code = mean_render_tables(fit_tables)
    fixed = table([1., 0., 0., 0.])
    primary_indices = [i for i, r in enumerate(rows) if r['cell'] in ('both_held', 'both_held_repeat')]
    parameter_rows, parameter_cells = parameter_diagnostics(rows, predictions)
    tables = {i: {name: table(predictions[name][i]) for name in ('stats', 'combined', 'analytic')}
              for i in primary_indices}
    cases = []
    for source in [s for s in state['sources'] if s['role'] == 'query']:
        with Image.open(ROOT / source['path']) as image:
            if list(image.size) != source['size']:
                raise ValueError('query size mismatch')
            histogram = image_histograms(np.asarray(image.convert('RGB'), dtype=np.uint8))
        for i in primary_indices:
            row = rows[i]
            target = table(row['target'])
            arms = {**tables[i], 'identity': table_input, 'mean_render': mean_float,
                    'fixed_contrast': fixed, 'oracle': target,
                    'wrong_primary': tables[state['wrong'][i]][state['primary']]}
            results = {}
            for name, predicted in arms.items():
                errors = histogram_errors(predicted, target, histogram['counts'])
                strength = histogram_errors(predicted, table_input, histogram['counts'])
                delivered = mean_code if name == 'mean_render' else predicted
                if name == 'mean_render':
                    errors['code_mse'] = histogram_errors(mean_code, target, histogram['counts'])['code_mse']
                    strength['code_mse'] = histogram_errors(mean_code, table_input, histogram['counts'])['code_mse']
                results[name] = {**errors, 'strength_float_mse': strength['float_mse'],
                                 'strength_code_mse': strength['code_mse'],
                                 'detail': histogram_detail_loss(delivered, histogram)}
            cases.append({**row, 'query_id': source['index'], 'arms': results,
                          'oracle_quantized_identical_to_source': results['oracle']['strength_code_mse'] == 0})
            if results['oracle']['float_mse'] != 0 or results['oracle']['code_mse'] != 0:
                raise ValueError('oracle pipeline reconstruction failed')
    primary_parameters = [parameter_rows[i] for i in primary_indices]
    decision = primary_decision(cases, primary_parameters, state['primary'], cfg['sign_test_alpha'])
    summary = {}
    for arm in cases[0]['arms']:
        summary[arm] = {}
        for metric, strength in (('float_mse', 'strength_float_mse'), ('code_mse', 'strength_code_mse')):
            error = np.mean([r['arms'][arm][metric] for r in cases])
            oracle_strength = np.mean([r['arms']['oracle'][strength] for r in cases])
            summary[arm][metric] = float(error)
            summary[arm][metric + '_relative_to_oracle_strength'] = float(error / oracle_strength) if oracle_strength else None
    for filename, contents in (('cases.json', cases), ('parameter_rows.json', parameter_rows),
                               ('parameter_cells.json', parameter_cells)):
        (out / filename).write_text(json.dumps(contents, default=json_default), encoding='utf-8')
    report = {'status': 'ASSESSED_FROZEN_HEAD_NO_REFIT_VISUAL_PENDING', 'primary': state['primary'],
              'decision': decision, 'rendered_summary': summary, 'primary_cases': len(cases),
              'diagnostic_references': len(rows), 'verified_pins': state['verified_pins'],
              'entry_sha256': digest(Path(__file__)), 'visual_assessment': 'NOT_PERFORMED',
              'analytic_prior': prior,
              'analytic_zero_denominator_references': len(rows) if prior['degenerate'] else 0,
              'oracle_quantized_identical_cases': sum(r['oracle_quantized_identical_to_source'] for r in cases),
              'artifact_sha256': {name: digest(out / name) for name in ('cases.json', 'parameter_rows.json', 'parameter_cells.json')}}
    (out / 'report.json').write_text(json.dumps(report, default=json_default, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'reference_contribution': decision['reference_contribution'],
                      'report_sha256': digest(out / 'report.json')}))


if __name__ == '__main__':
    main()

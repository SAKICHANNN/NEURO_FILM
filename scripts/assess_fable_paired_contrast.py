import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_reference_photometry import quantize8, transform
from src.eval.fable_photometry_metrics import histogram_errors, histogram_detail_loss, image_histograms, parameter_errors
from src.eval.fable_paired_contrast import joint_code_engineering_gate, strict_win_sign_test_16


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate_prediction_layout(predictions: dict, draws: np.ndarray) -> np.ndarray:
    donors, draw_ids, pair_ids = [np.asarray(predictions[key]) for key in ('donor_ids', 'draw_ids', 'pair_ids')]
    expected = {(donor, draw) for donor in (0, 1, 2, 3, 4, 5, 12, 13, 14, 15) for draw in range(32)}
    if any(v.shape != (320,) or v.dtype.kind not in 'iu' for v in (donors, draw_ids, pair_ids)):
        raise ValueError('320 integer row identifiers required')
    keys = list(zip(donors.tolist(), draw_ids.tolist(), strict=True))
    if len(set(keys)) != 320 or set(keys) != expected or not np.array_equal(pair_ids, draw_ids // 2):
        raise ValueError('prospective reference layout mismatch')
    if (draws.shape != (32, 4) or not np.isfinite(draws).all() or np.any(np.abs(draws) > 1)
            or not np.array_equal(draws[::2], -draws[1::2])
            or not np.array_equal(predictions['targets'], draws[draw_ids])):
        raise ValueError('targets do not match the frozen fresh antithetic draws')
    for arm in ('after_only', 'paired_control', 'frozen_combined'):
        u, raw, clipped = [np.asarray(predictions[arm + suffix]) for suffix in ('', '_raw', '_clipped')]
        if (u.shape != (320, 4) or raw.shape != u.shape or clipped.shape != u.shape
                or not np.isfinite(raw).all() or not np.array_equal(u, np.clip(raw, -1, 1))
                or not np.array_equal(clipped, np.abs(raw) > 1)):
            raise ValueError(f'invalid prediction or clipping data: {arm}')
    lookup = {key: row for row, key in enumerate(keys)}
    return np.asarray([lookup[(donor, draw ^ 1)] for donor, draw in keys])


def assess_cell(cases: list[dict], donor_ids: list[int], query_ids: list[int]) -> dict:
    expected = {(d, j, q) for d in donor_ids for j in range(32) for q in query_ids}
    keys = [(r['donor_id'], r['draw_id'], r['query_id']) for r in cases]
    if len(keys) != len(expected) or set(keys) != expected or any(r['pair_id'] != r['draw_id'] // 2 for r in cases):
        raise ValueError('complete balanced donor/edit/query cell required')
    donors = np.asarray([r['donor_id'] for r in cases])
    queries = np.asarray([r['query_id'] for r in cases])
    pairs = np.asarray([r['pair_id'] for r in cases])
    d = np.asarray([r['arms']['oracle']['code_P'] for r in cases])
    gates, summaries = {}, {}
    for arm in cases[0]['arms']:
        e = np.asarray([r['arms'][arm]['code_E'] for r in cases])
        p = np.asarray([r['arms'][arm]['code_P'] for r in cases])
        gates[arm] = joint_code_engineering_gate(d, e, p, donors, queries)
        summaries[arm] = {}
        for label, mask in [('pooled', np.ones(len(cases), dtype=bool)),
                            *[(f'donor_{d}', donors == d) for d in donor_ids],
                            *[(f'query_{q}', queries == q) for q in query_ids]]:
            rows = [r for r, included in zip(cases, mask, strict=True) if included]
            summary = {metric: float(np.mean([r['arms'][arm][metric] for r in rows]))
                       for metric in ('float_E', 'code_E', 'float_P', 'code_P')}
            summary['float_D'] = float(np.mean([r['arms']['oracle']['float_P'] for r in rows]))
            summary['code_D'] = float(np.mean([r['arms']['oracle']['code_P'] for r in rows]))
            summaries[arm][label] = summary
    after_error = np.asarray([r['arms']['after_only']['code_E'] for r in cases])
    margins = {'quarter_oracle_energy_minus_error': .25 * d - after_error}
    for arm in ('frozen_combined', 'mean_render', 'wrong_after_only'):
        margins[arm + '_error_minus_after_error'] = np.asarray([r['arms'][arm]['code_E'] for r in cases]) - after_error
    tests = {}
    for name, margin in margins.items():
        blocks = np.asarray([margin[pairs == pair].mean() for pair in range(16)])
        tests[name] = {**strict_win_sign_test_16(blocks), 'block_margins': blocks,
                       'cases_per_block': len(cases) // 16}
    complete = gates['after_only']['passed'] and all(t['passed'] for t in tests.values())
    return {'cases': len(cases), 'engineering_gates': gates, 'after_only_sign_tests': tests,
            'after_only_complete_gate': complete, 'summaries': summaries}


def branch_decision(primary: dict, secondary: dict) -> dict:
    if primary['after_only_complete_gate']:
        return {'action': 'ADMIT_REAL_TWO_GRADES_FOUR_FRESH_SCENES_CALIBRATION',
                'reason': 'NEW_DONOR_AFTER_ONLY_COMPLETE_GATE_PASS', 'product_promotion': False}
    if not primary['engineering_gates']['paired_control']['passed']:
        reason = 'PAIRED_CONTROL_ALSO_FAILS_ENGINEERING_GATE'
    elif secondary['after_only_complete_gate']:
        reason = 'PAIRED_AND_REUSED_AFTER_WORK_NEW_DONOR_AFTER_FAILS'
    else:
        reason = 'PAIRED_WORKS_REUSED_AFTER_COMPLETE_GATE_FAILS'
    return {'action': 'CLOSE_FIXED_DESCRIPTOR_LINEAR_DECODING_ROUTE', 'reason': reason,
            'no_rescue_refit_gain_seed_donor_change': True, 'real_calibration_admitted': False,
            'claim': 'This decoder/prior route failed; no AFTER-only impossibility claim.'}


def verify_inputs(root: Path) -> dict:
    cp = root / 'configs/fable_paired_contrast_v1.json'
    cfg = json.loads(cp.read_text())
    output = root / cfg['output']
    model_path = output / 'fit_head/model.json'
    model = json.loads(model_path.read_text())
    draws_path = output / 'fresh_draws.json'
    draw_record = json.loads(draws_path.read_text())
    feature_report = json.loads((output / 'fresh_features/report.json').read_text())
    prediction_report = json.loads((output / 'fresh_predictions/report.json').read_text())
    if (model['status'] != 'CONTRAST_DECODER_FROZEN_BEFORE_FRESH_DRAWS'
            or draw_record['status'] != 'FRESH_DRAWS_GENERATED_AFTER_MODEL_LOCK' or draw_record['redraws'] != 0):
        raise ValueError('frozen model and one prospective draw receipt required')
    pins = {root / path: expected for path, expected in cfg['frozen_inputs'].items()}
    pins.update({cp: model['config_sha256'], model_path: draw_record['model_sha256'],
                 root / cfg['source_config']: cfg['source_config_sha256'],
                 root / 'src/eval/fable_paired_contrast.py': cfg['decoder_module_sha256'],
                 draws_path: prediction_report['fresh_draws_sha256'],
                 output / 'fresh_features/features.npz': prediction_report['features_sha256'],
                 output / 'fresh_predictions/predictions.npz': prediction_report['predictions_sha256']})
    for record in (draw_record, feature_report, prediction_report):
        if record['config_sha256'] != model['config_sha256'] or record['model_sha256'] != draw_record['model_sha256']:
            raise ValueError('inconsistent frozen model/config receipt')
    if (feature_report['features_sha256'] != prediction_report['features_sha256']
            or feature_report['fresh_draws_sha256'] != prediction_report['fresh_draws_sha256']):
        raise ValueError('inconsistent feature/draw receipts')
    old_cfg = json.loads((root / 'configs/fable_reference_photometry_v1.json').read_text())
    old_sources = json.loads((root / old_cfg['sources_config']).read_text())['sources']
    new_sources = json.loads((root / cfg['source_config']).read_text())['sources']
    for source in old_sources + new_sources:
        pins[root / source['path']] = source['sha256']
        if source.get('raw_path'):
            pins[root / source['raw_path']] = source['raw_sha256']
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f'frozen artifact mismatch: {path}')
    with np.load(output / 'fresh_predictions/predictions.npz', allow_pickle=False) as data:
        predictions = {key: data[key].copy() for key in data.files}
    draws = np.asarray(draw_record['draws'], dtype=np.float64)
    wrong = validate_prediction_layout(predictions, draws)
    with np.load(output / 'fresh_features/features.npz', allow_pickle=False) as data:
        for key in ('donor_ids', 'draw_ids', 'pair_ids', 'targets'):
            if not np.array_equal(predictions[key], data[key]):
                raise ValueError('prediction identities differ from feature identities')
    queries = [s for s in old_sources if s['index'] in cfg['prospective']['queries']]
    if sorted(s['index'] for s in new_sources) != [12, 13, 14, 15] or sorted(s['index'] for s in queries) != [8, 9, 10, 11]:
        raise ValueError('frozen donor/query roles differ')
    fit_draws = np.asarray(json.loads((root / 'outputs/fable_reference_photometry_v1/draws.json').read_text())['fit'])
    return {'cfg': cfg, 'old_cfg': old_cfg, 'output': output, 'predictions': predictions, 'wrong': wrong,
            'queries': queries, 'draws': draws, 'fit_draws': fit_draws,
            'verified_pins': {str(path.relative_to(root)): value for path, value in pins.items()}}


def main():
    state = verify_inputs(ROOT)
    out = state['output'] / 'assessment'
    out.mkdir(exist_ok=False)
    predictions = state['predictions']
    source_table = np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)

    def table(u):
        return transform(source_table, u, slope_limit=state['old_cfg']['slope_limit'],
                         offset_limit=state['old_cfg']['offset_limit'])

    oracle_tables = np.stack([table(u) for u in state['draws']])
    fit_tables = np.stack([table(u) for u in state['fit_draws']])
    mean_float, mean_code = fit_tables.mean(0), quantize8(quantize8(fit_tables).mean(0))
    inferred_tables = {arm: np.stack([table(u) for u in predictions[arm]])
                       for arm in ('after_only', 'paired_control', 'frozen_combined')}
    cases = []
    for query in state['queries']:
        with Image.open(ROOT / query['path']) as image:
            if list(image.size) != query['size']:
                raise ValueError('query size mismatch')
            histogram = image_histograms(np.asarray(image.convert('RGB'), dtype=np.uint8))
        for i in range(320):
            donor, draw, pair = [int(predictions[key][i]) for key in ('donor_ids', 'draw_ids', 'pair_ids')]
            oracle = oracle_tables[draw]
            arms = {**{arm: values[i] for arm, values in inferred_tables.items()},
                    'identity': source_table, 'mean_render': mean_float, 'oracle': oracle,
                    'wrong_after_only': inferred_tables['after_only'][state['wrong'][i]]}
            metrics = {}
            for arm, values in arms.items():
                errors = histogram_errors(values, oracle, histogram['counts'])
                strength = histogram_errors(values, source_table, histogram['counts'])
                delivered = mean_code if arm == 'mean_render' else values
                if arm == 'mean_render':
                    errors['code_mse'] = histogram_errors(delivered, oracle, histogram['counts'])['code_mse']
                    strength['code_mse'] = histogram_errors(delivered, source_table, histogram['counts'])['code_mse']
                metrics[arm] = {'float_E': errors['float_mse'], 'code_E': errors['code_mse'],
                                'float_P': strength['float_mse'], 'code_P': strength['code_mse'],
                                'detail': histogram_detail_loss(delivered, histogram)}
            if metrics['oracle']['float_E'] != 0 or metrics['oracle']['code_E'] != 0:
                raise ValueError('oracle reconstruction failed')
            cases.append({'donor_id': donor, 'draw_id': draw, 'pair_id': pair, 'query_id': query['index'],
                          'cell': 'primary_new' if donor >= 12 else 'secondary_reused', 'arms': metrics})
    cells = {}
    parameter_report = {}
    for cell, donors in (('primary_new', [12, 13, 14, 15]), ('secondary_reused', list(range(6)))):
        cells[cell] = assess_cell([r for r in cases if r['cell'] == cell], donors, state['cfg']['prospective']['queries'])
        masks = {'pooled': np.isin(predictions['donor_ids'], donors),
                 **{f'donor_{d}': predictions['donor_ids'] == d for d in donors}}
        parameter_report[cell] = {}
        for arm in ('after_only', 'paired_control', 'frozen_combined'):
            parameter_report[cell][arm] = {}
            for label, mask in masks.items():
                parameter_report[cell][arm][label] = {
                    **parameter_errors(predictions[arm][mask], predictions['targets'][mask]),
                    'coordinate_clipping_frequency': float(predictions[arm + '_clipped'][mask].mean()),
                    'row_clipping_frequency': float(predictions[arm + '_clipped'][mask].any(1).mean())}
    (out / 'cases.json').write_text(json.dumps(cases, default=serial), encoding='utf-8')
    (out / 'parameters.json').write_text(json.dumps(parameter_report, default=serial), encoding='utf-8')
    report = {'status': 'ONE_FROZEN_PROSPECTIVE_ASSESSMENT_COMPLETE_VISUAL_PENDING', 'cells': cells,
              'branch': branch_decision(cells['primary_new'], cells['secondary_reused']),
              'visual_review': 'NOT_PERFORMED', 'verified_pins': state['verified_pins'],
              'entry_sha256': digest(Path(__file__)),
              'metrics_module_sha256': digest(ROOT / 'src/eval/fable_photometry_metrics.py'),
              'artifacts': {name: digest(out / name) for name in ('cases.json', 'parameters.json')}}
    (out / 'report.json').write_text(json.dumps(report, default=serial, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'branch': report['branch'], 'report_sha256': digest(out / 'report.json')}))


if __name__ == '__main__':
    main()

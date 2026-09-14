import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_canonical_prior import canonical_measure, decode_canonical_prior
from src.eval.fable_reference_photometry import dequantize8, quantize8, transform
from src.eval.fable_photometry_metrics import image_histograms, histogram_errors
from src.eval.fable_paired_contrast import joint_code_engineering_gate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp = ROOT / 'configs/fable_canonical_prior_smoke_v1.json'
    cfg = json.loads(cp.read_text())
    for path, sha in cfg['pins'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'input changed: {path}')
    out = ROOT / cfg['output']
    out.mkdir(exist_ok=False)
    with np.load(ROOT / 'outputs/fable_paired_contrast_v1/fresh_features/features.npz', allow_pickle=False) as data:
        bases, ids = data['canonical_bases'].copy(), data['identity_donor_ids'].copy()
    old_draws = np.asarray(json.loads((ROOT / 'outputs/fable_reference_photometry_v1/draws.json').read_text())['fit'])
    draws = np.sign(old_draws) * (.25 + .75 * np.abs(old_draws))
    measure_args = {k: cfg[k] for k in ['epsilon', 'scale_floor', 'tensor_limit']}
    transform_args = {k: cfg[k] for k in ['slope_limit', 'offset_limit']}
    source_table = np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)
    records, predictions, targets = [], [], []
    for donor, base in zip(ids, bases, strict=True):
        identity = canonical_measure(base, **measure_args)
        latent = dequantize8(base, seed=cfg['dequantization_seeds'][str(donor)])
        for j, u in enumerate(draws):
            after = quantize8(transform(latent, u, **transform_args))
            measured = canonical_measure(after, **measure_args)
            predicted = decode_canonical_prior(measured, identity['target'], **transform_args)
            predictions.append(predicted['u'])
            targets.append(u)
            records.append({'donor_id': int(donor), 'draw_id': j,
                'parameter_mse': float(np.mean((predicted['u'] - u) ** 2)),
                'raw_parameter_mse': float(np.mean((predicted['raw'] - u) ** 2)),
                'invariant_mse': float(np.mean((measured['tensor'] - identity['tensor']) ** 2)),
                'clamp_fraction': measured['clamp_fraction'], 'scale_floored': measured['scale_floored'],
                'tensor_clipping_fraction': measured['tensor_clipping_fraction'],
                'parameter_clipping_fraction': float(predicted['clipped'].mean())})
    np.savez(out / 'paired_diagnostic.npz', u=predictions, targets=targets, draws=draws)
    source_cfg = json.loads((ROOT / 'configs/fable_reference_photometry_sources_v1.json').read_text())
    cases = []
    for query in source_cfg['sources']:
        if query['index'] not in cfg['queries']:
            continue
        path = ROOT / query['path']
        if digest(path) != query['sha256']:
            raise ValueError('historical query changed')
        with Image.open(path) as image:
            hist = image_histograms(np.asarray(image.convert('RGB'), dtype=np.uint8))['counts']
        for row, prediction, target in zip(records, predictions, targets, strict=True):
            predicted_table = transform(source_table, prediction, **transform_args)
            oracle_table = transform(source_table, target, **transform_args)
            cases.append({'donor_id': row['donor_id'], 'draw_id': row['draw_id'], 'query_id': query['index'],
                'D': histogram_errors(oracle_table, source_table, hist)['code_mse'],
                'E': histogram_errors(predicted_table, oracle_table, hist)['code_mse'],
                'P': histogram_errors(predicted_table, source_table, hist)['code_mse']})
    gate = joint_code_engineering_gate(*[np.asarray([r[k] for r in cases]) for k in ['D', 'E', 'P', 'donor_id', 'query_id']])
    (out / 'rows.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (out / 'cases.json').write_text(json.dumps(cases), encoding='utf-8')
    report = {'status': 'HISTORICAL_ENGINEERING_SMOKE_ONLY', 'config_sha256': digest(cp),
        'entry_sha256': digest(Path(__file__)), 'pins': cfg['pins'], 'reference_rows': len(records),
        'cases': len(cases), 'gate': gate,
        'summary': {k: float(np.mean([r[k] for r in records])) for k in ['parameter_mse','raw_parameter_mse','invariant_mse','clamp_fraction','scale_floored','tensor_clipping_fraction','parameter_clipping_fraction']},
        'artifacts': {f: digest(out / f) for f in ['paired_diagnostic.npz','rows.json','cases.json']},
        'next': 'METADATA_LINEAGE_PREFLIGHT_ONLY' if gate['passed'] else 'STOP_BULK_ACQUISITION_NUMERICAL_FAILURE',
        'claim_ceiling': 'Consumed historical scenes and engineering transforms; not independent validation, learned inference, film calibration, or content proof.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'passed': gate['passed'], 'summary': report['summary'],
                      'pooled': gate['accuracy']['pooled'], 'next': report['next']}))


if __name__ == '__main__':
    main()

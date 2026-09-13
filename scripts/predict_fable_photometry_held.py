import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_photometry_ridge import predict_head
from src.eval.fable_photometry_metrics import analytic_prior, analytic_predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp = ROOT/'configs/fable_reference_photometry_v1.json'
    cfg = json.loads(cp.read_text())
    root = ROOT/cfg['output']
    report = json.loads((root/'held_features/report.json').read_text())
    mp = root/'fit_head/model.json'
    assert digest(mp) == report['model_sha256']
    assert digest(cp) == report['config_sha256']
    fp = root/'held_features/features.npz'
    assert digest(fp) == report['features_sha256']
    assert digest(root/'held_features/rows.json') == report['rows_sha256']
    fitted = json.loads(mp.read_text())
    assert digest(ROOT/'src/eval/fable_photometry_ridge.py') == fitted['ridge_module_sha256']
    fit_path = root/'fit_features/features.npz'
    assert digest(fit_path) == fitted['features_sha256']
    out = root/'held_predictions'
    out.mkdir(exist_ok=False)
    predictions = {}
    with np.load(fp, allow_pickle=False) as data, np.load(fit_path, allow_pickle=False) as fit, threadpool_limits(limits=cfg['cpu_threads']):
        for name, model in fitted['models'].items():
            model['coefficients'] = np.asarray(model['coefficients'], dtype=np.float64)
            model['intercept'] = np.asarray(model['intercept'], dtype=np.float64)
            for standard in model['standards']:
                for key in ['mean', 'scale']:
                    standard[key] = np.asarray(standard[key], dtype=np.float64)
                standard['active'] = np.asarray(standard['active'], dtype=bool)
            blocks = [data['stats']] if name == 'stats' else [data['stats'], data['deep']]
            result = predict_head(model, blocks)
            predictions[name] = result['u']
            predictions[name+'_raw'] = result['raw']
            predictions[name+'_clipped'] = result['clipped']
        prior = analytic_prior(fit['canonical_bases'])
        predictions['analytic'] = np.stack([analytic_predict(image.astype(np.float64)/255, prior,
            slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit']) for image in data['references']])
    assert predictions['analytic'].shape == predictions['combined'].shape == (384, 4)
    np.savez(out/'predictions.npz', **predictions)
    receipt = {'status': 'PREDICTIONS_FROZEN_NO_REFIT', 'primary': fitted['primary'],
        'model_sha256': digest(mp), 'held_features_sha256': digest(fp), 'rows_sha256': report['rows_sha256'],
        'predictions_sha256': digest(out/'predictions.npz'), 'entry_sha256': digest(Path(__file__)),
        'metrics_module_sha256': digest(ROOT/'src/eval/fable_photometry_metrics.py')}
    (out/'report.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

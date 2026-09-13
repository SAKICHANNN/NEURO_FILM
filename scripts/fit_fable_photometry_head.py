import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_photometry_ridge import select_heads


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_reference_photometry_v1.json'
    cfg = json.loads(cp.read_text())
    root = ROOT/cfg['output']
    feature_report = json.loads((root/'fit_features/report.json').read_text())
    assert digest(cp) == feature_report['config_sha256']
    assert digest(root/'draws.json') == feature_report['draws_sha256']
    fp = root/'fit_features/features.npz'
    assert digest(fp) == feature_report['features_sha256']
    out = root/'fit_head'
    out.mkdir(exist_ok=False)
    with np.load(fp, allow_pickle=False) as data, threadpool_limits(limits=cfg['cpu_threads']):
        fitted = select_heads(data['stats'], data['deep'], data['targets'],
            data['donor_ids'], data['pair_ids'], np.asarray(cfg['ridge_grid']))
    result = {'status': 'PRIMARY_FROZEN_BEFORE_ASSESSMENT', **fitted,
        'config_sha256': digest(cp), 'features_sha256': digest(fp),
        'entry_sha256': digest(Path(__file__)),
        'ridge_module_sha256': digest(ROOT/'src/eval/fable_photometry_ridge.py'),
        'wall_seconds': time.perf_counter()-start}
    (out/'model.json').write_text(json.dumps(result, default=json_value, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'primary': fitted['primary'], 'cv': {k: {'error': v['selected_error'],
        'lambda': v['selected_ridge']} for k, v in fitted['cv'].items()},
        'wall_seconds': result['wall_seconds'], 'model_sha256': digest(out/'model.json')}))


if __name__ == '__main__':
    main()

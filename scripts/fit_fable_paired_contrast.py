import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_paired_contrast import select_paired_contrast


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    cp = ROOT / 'configs/fable_paired_contrast_v1.json'
    cfg = json.loads(cp.read_text())
    if cfg['status'] != 'FROZEN_BEFORE_FIT':
        raise ValueError('full protocol and new donor admission must be frozen before fitting')
    for path, expected in cfg['frozen_inputs'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'frozen input mismatch: {path}')
    if not cfg['source_config'] or digest(ROOT / cfg['source_config']) != cfg['source_config_sha256']:
        raise ValueError('new donor source manifest not pinned')
    module = ROOT / 'src/eval/fable_paired_contrast.py'
    if digest(module) != cfg['decoder_module_sha256']:
        raise ValueError('decoder module mismatch')
    with np.load(ROOT / 'outputs/fable_reference_photometry_v1/fit_features/features.npz', allow_pickle=False) as data:
        fit = {key: data[key].copy() for key in data.files}
    with np.load(ROOT / 'outputs/fable_paired_contrast_v1/fit_identities/features.npz', allow_pickle=False) as data:
        identities = {key: data[key].copy() for key in data.files}
    if not np.array_equal(identities['donor_ids'], np.arange(6)):
        raise ValueError('identity rows must be ordered by original fit donor id')
    out = ROOT / cfg['output'] / 'fit_head'
    out.mkdir(exist_ok=False)
    with threadpool_limits(limits=cfg['descriptor']['cpu_threads']):
        selected = select_paired_contrast([fit['stats'], fit['deep']], [identities['stats'], identities['deep']],
            fit['targets'], fit['donor_ids'], fit['pair_ids'], cfg['training']['ridge_grid'])
    receipt = {**selected, 'status': 'CONTRAST_DECODER_FROZEN_BEFORE_FRESH_DRAWS',
        'config_sha256': digest(cp), 'module_sha256': digest(module), 'entry_sha256': digest(Path(__file__)),
        'frozen_inputs': cfg['frozen_inputs']}
    (out / 'model.json').write_text(json.dumps(receipt, default=serial, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'selected_ridge': selected['selected_ridge'],
        'paired_cv_error': float(selected['mean_errors'][selected['selected_index']]),
        'model_sha256': digest(out / 'model.json')}))


if __name__ == '__main__':
    main()

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.tst_reference_response import descriptors, load_vgg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_fit_source_features_v1.json'
    cfg = json.loads(cp.read_text())
    sources = ROOT/cfg['sources_config']
    assert digest(sources) == cfg['sources_config_sha256']
    assert digest(ROOT/'src/eval/tst_reference_response.py') == cfg['feature_entry_sha256']
    rows = [s for s in json.loads(sources.read_text())['sources'] if s['role'] == cfg['source_role']]
    assert len(rows) == 6
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    torch.set_num_threads(cfg['cpu_threads'])
    model = load_vgg(ROOT)
    stats, deep = [], []
    for row in rows:
        path = ROOT/row['path']
        assert digest(path) == row['sha256']
        with Image.open(path) as im:
            assert list(im.size) == row['size']
            values = np.asarray(im.convert('RGB'), dtype=np.float32)/255
        a, b = descriptors(values, model)
        stats.append(a)
        deep.append(b)
    np.savez(out/'features.npz', source_ids=[s['index'] for s in rows], stats=np.stack(stats), deep=np.stack(deep))
    report = {'status': 'FIT_SOURCE_FEATURES_EXTRACTED_NOT_TRAINING', 'config_sha256': digest(cp),
        'entry_sha256': digest(Path(__file__)), 'features_sha256': digest(out/'features.npz'),
        'source_ids': [s['index'] for s in rows], 'stats_shape': np.stack(stats).shape,
        'deep_shape': np.stack(deep).shape, 'wall_seconds': time.perf_counter()-start,
        'held_features_consumed': False, 'fitted_normalization_or_head': False}
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

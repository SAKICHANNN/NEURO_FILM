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
from src.eval.fable_reference_photometry import canonical_donor, dequantize8, quantize8, transform
from src.eval.tst_reference_response import descriptors, load_vgg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_reference_photometry_v1.json'
    cfg = json.loads(cp.read_text())
    sp = ROOT/cfg['sources_config']
    assert digest(sp) == cfg['sources_config_sha256']
    out = ROOT/cfg['output']/'fit_features'
    out.mkdir(exist_ok=False)
    draws_path = out.parent/'draws.json'
    draws = np.asarray(json.loads(draws_path.read_text())['fit'], dtype=np.float64)
    assert draws.shape == (32, 4) and np.array_equal(draws[::2], -draws[1::2])
    sources = [s for s in json.loads(sp.read_text())['sources'] if s['role'] == 'fit_donor']
    assert len(sources) == 6
    torch.set_num_threads(cfg['cpu_threads'])
    model = load_vgg(ROOT)
    stats, deep, ids, draw_ids, targets, bases = [], [], [], [], [], []
    for source in sources:
        path = ROOT/source['path']
        assert digest(path) == source['sha256']
        with Image.open(path) as im:
            canonical = canonical_donor(np.asarray(im.convert('RGB'), dtype=np.float32)/255, side=cfg['side'])
        bases.append(canonical)
        latent = dequantize8(canonical, seed=cfg['dequantization_seed_base']+source['index'])
        for j, u in enumerate(draws):
            after = quantize8(transform(latent, u, slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit']))
            a, b = descriptors(after, model)
            stats.append(a)
            deep.append(b)
            ids.append(source['index'])
            draw_ids.append(j)
            targets.append(u)
        print(json.dumps({'donor_done': source['index'], 'references': len(stats)}), flush=True)
    assert len(stats) == cfg['expected_reference_counts']['fit']
    np.savez(out/'features.npz', stats=stats, deep=deep, donor_ids=ids, draw_ids=draw_ids,
             pair_ids=np.asarray(draw_ids)//2, targets=targets, canonical_bases=bases)
    report = {'status': 'FIT_FEATURES_READY_NO_HEAD_OR_ASSESSMENT', 'references': len(stats),
        'config_sha256': digest(cp), 'draws_sha256': digest(draws_path), 'entry_sha256': digest(Path(__file__)),
        'transform_module_sha256': digest(ROOT/'src/eval/fable_reference_photometry.py'),
        'descriptor_module_sha256': digest(ROOT/'src/eval/tst_reference_response.py'),
        'features_sha256': digest(out/'features.npz'), 'wall_seconds': time.perf_counter()-start}
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

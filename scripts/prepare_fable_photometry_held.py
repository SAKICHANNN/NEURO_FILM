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
    root = ROOT/cfg['output']
    mp = root/'fit_head/model.json'
    evidence = json.loads((ROOT/'docs/evidence/FABLE_PHOTOMETRY_FIT_20260913.json').read_text())
    assert digest(mp) == evidence['model_sha256']
    assert digest(cp) == evidence['config_sha256']
    frozen = json.loads(mp.read_text())
    fit_report = json.loads((root/'fit_features/report.json').read_text())
    assert digest(root/'draws.json') == fit_report['draws_sha256']
    for key, path in [('transform_module_sha256', 'src/eval/fable_reference_photometry.py'),
                      ('descriptor_module_sha256', 'src/eval/tst_reference_response.py')]:
        assert digest(ROOT/path) == fit_report[key]
    sp = ROOT/cfg['sources_config']
    assert digest(sp) == cfg['sources_config_sha256']
    sources = json.loads(sp.read_text())['sources']
    draws = json.loads((root/'draws.json').read_text())
    out = root/'held_features'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(cfg['cpu_threads'])
    model = load_vgg(ROOT)
    records, refs, stats, deep = [], [], [], []
    for source in sources:
        if source['role'] == 'query':
            continue
        path = ROOT/source['path']
        assert digest(path) == source['sha256']
        with Image.open(path) as im:
            canonical = canonical_donor(np.asarray(im.convert('RGB'), dtype=np.float32)/255, side=cfg['side'])
        cells = [('parameter_held', 'assessment', 0)] if source['role'] == 'fit_donor' else [
            ('donor_held', 'fit', 0), ('both_held', 'assessment', 0), ('both_held_repeat', 'assessment', 1)]
        for cell, partition, repeat in cells:
            seed_key = 'dequantization_repeat_seed_base' if repeat else 'dequantization_seed_base'
            latent = dequantize8(canonical, seed=cfg[seed_key]+source['index'])
            for j, u in enumerate(draws[partition]):
                after = quantize8(transform(latent, np.asarray(u), slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit']))
                a, b = descriptors(after, model)
                stats.append(a)
                deep.append(b)
                refs.append(np.floor(after*255+.5).astype(np.uint8))
                records.append({'cell': cell, 'donor_id': source['index'], 'draw_id': j,
                    'pair_id': j//2, 'repeat': repeat, 'partition': partition, 'target': u})
        print(json.dumps({'donor_done': source['index'], 'references': len(records)}), flush=True)
    counts = {cell: sum(row['cell'] == cell for row in records) for cell in cfg['expected_reference_counts'] if cell != 'fit'}
    assert all(n == cfg['expected_reference_counts'][cell] for cell, n in counts.items())
    assert len(records) == 384
    np.savez(out/'features.npz', stats=stats, deep=deep, references=refs)
    (out/'rows.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    report = {'status': 'HELD_FEATURES_READY_MODEL_UNCHANGED', 'counts': counts, 'primary': frozen['primary'],
        'model_sha256': digest(mp), 'config_sha256': digest(cp), 'entry_sha256': digest(Path(__file__)),
        'rows_sha256': digest(out/'rows.json'), 'features_sha256': digest(out/'features.npz'),
        'wall_seconds': time.perf_counter()-start}
    assert report['model_sha256'] == evidence['model_sha256']
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

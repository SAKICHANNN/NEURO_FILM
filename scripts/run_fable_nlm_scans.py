import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from skimage.restoration import denoise_nl_means

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
SPEC = importlib.util.spec_from_file_location('pilot_base', ROOT/'scripts/run_fable_scan_residual_pilot.py')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_nlm_scans_v1.json'
    cfg = json.loads(cp.read_text())
    assert base.digest(Path(base.__file__)) == cfg['base_entry_sha256']
    for key in ['previous_report', 'source_inventory']:
        assert base.digest(ROOT/cfg[key]) == cfg[key+'_sha256']
    previous = json.loads((ROOT/cfg['previous_report']).read_text())
    inventory = json.loads((ROOT/cfg['source_inventory']).read_text())
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    rows = []
    for source in inventory['files']:
        path = ROOT/source['path']
        assert base.digest(path) == source['sha256']
        array = tifffile.imread(path)
        assert list(array.shape) == source['shape']
        for old in previous['scan_results']:
            if old['source'] != path.name:
                continue
            left, top, right, bottom = old['box_xyxy']
            patch = base.rgb_codes(array[top:bottom, left:right])
            assert np.array_equal(patch.mean(axis=(0, 1)), old['mean_code'])
            row = {**old, 'methods': dict(old['methods'])}
            for h in cfg['nlm_h']:
                estimate = denoise_nl_means(patch, patch_size=cfg['patch_size'],
                    patch_distance=cfg['patch_distance'], h=h, fast_mode=cfg['fast_mode'],
                    sigma=cfg['sigma'], preserve_range=True, channel_axis=-1)
                row['methods'][f'nlm_h{h}'] = base.describe(patch-estimate)
            rows.append(row)
        del array
    assert len(rows) == previous['patches'] == 72
    names = list(rows[0]['methods'])
    rms = {name: np.array([row['methods'][name]['rms'] for row in rows]) for name in names}
    ratio = rms['nlm_h0.02']/rms['nlm_h0.01']
    summary = {'channel_median_rms': {name: np.median(values, axis=0).tolist() for name, values in rms.items()},
        'h02_over_h01_rms_min_median_max': [float(ratio.min()), float(np.median(ratio)), float(ratio.max())]}
    report = {'status': 'REAL_SCAN_SENSITIVITY_NOT_GRAIN_TRUTH', 'config_sha256': base.digest(cp),
        'entry_sha256': base.digest(Path(__file__)), 'scan_results': rows, 'summary': summary,
        'patches': len(rows), 'sources': len(inventory['files']), 'wall_seconds': time.perf_counter()-start,
        'limits': previous['limits'] + ['NLM settings unchanged from known-signal controls',
            'uniform scans do not test scene leakage on natural photographs',
            'eight scans are not 72 independent specimens; no optical-density calibration']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'patches', 'wall_seconds', 'summary']}))


if __name__ == '__main__':
    main()

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from skimage.restoration import denoise_nl_means

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
SPEC = importlib.util.spec_from_file_location('pilot_base', ROOT/'scripts/run_fable_scan_residual_pilot.py')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_nlm_controls_v1.json'
    cfg = json.loads(cp.read_text())
    assert base.digest(Path(base.__file__)) == cfg['base_entry_sha256']
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    clean, correlated = base.controls(cfg['side'], cfg['seed'], cfg['noise_rms'])
    rng = np.random.default_rng(cfg['seed']+1)
    white = rng.standard_normal(correlated.shape)
    white -= white.mean(axis=(0, 1), keepdims=True)
    white *= cfg['noise_rms']/np.sqrt(np.mean(white**2, axis=(0, 1), keepdims=True))
    methods = ['gaussian_sigma1', 'gaussian_sigma2', 'median_size3'] + [f'nlm_h{h}' for h in cfg['nlm_h']]
    def extract(image, name):
        if not name.startswith('nlm_'):
            return base.residual(image, name)
        h = float(name.split('_h')[1])
        estimate = denoise_nl_means(image, patch_size=cfg['patch_size'], patch_distance=cfg['patch_distance'],
                                   h=h, fast_mode=cfg['fast_mode'], sigma=cfg['sigma'],
                                   preserve_range=True, channel_axis=-1)
        return image-estimate
    rows = []
    for scene, signal in clean.items():
        for method in methods:
            clean_residual = extract(signal, method)
            for kind, noise in [('correlated', correlated), ('white', white)]:
                estimate = extract(signal+noise, method)
                rows.append({'scene': scene, 'method': method, 'noise': kind,
                             'clean_leakage_rms': float(np.sqrt(np.mean(clean_residual**2))),
                             'noise_relative_mse': float(np.mean((estimate-noise)**2)/np.mean(noise**2)),
                             'residual_to_true_rms_ratio': float(np.sqrt(np.mean(estimate**2)/np.mean(noise**2)))})
    report = {'status': 'KNOWN_SIGNAL_DIAGNOSTIC_NOT_REAL_GRAIN_VALIDATION', 'rows': rows,
              'config_sha256': base.digest(cp), 'entry_sha256': base.digest(Path(__file__)),
              'wall_seconds': time.perf_counter()-start, 'no_clipping_or_quantization': True,
              'claim_ceiling': 'These exact clean patterns and two synthetic noise families only; no natural scan or physical grain claim.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'wall_seconds':report['wall_seconds'],'status':report['status']}))


if __name__ == '__main__':
    main()

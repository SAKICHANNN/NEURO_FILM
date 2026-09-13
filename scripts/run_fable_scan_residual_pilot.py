import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb_codes(array):
    if array.dtype != np.uint16 or array.ndim != 3 or array.shape[2] != 4:
        raise ValueError('Expected native uint16 RGB plus separate IR')
    return array[..., :3].astype(np.float64) / 65535.


def residual(image, method):
    if method == 'gaussian_sigma1':
        baseline = gaussian_filter(image, (1, 1, 0), mode='reflect')
    elif method == 'gaussian_sigma2':
        baseline = gaussian_filter(image, (2, 2, 0), mode='reflect')
    elif method == 'median_size3':
        baseline = median_filter(image, (3, 3, 1), mode='reflect')
    else:
        raise ValueError(method)
    return image - baseline


def spectrum(image):
    h, w, _ = image.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    centered = image - image.mean(axis=(0, 1), keepdims=True)
    power = abs(np.fft.fft2(centered * window[..., None], axes=(0, 1))) ** 2 / np.sum(window ** 2)
    freq = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    edges = np.linspace(0, .5, 33)
    radial = [power[(freq >= a) & (freq < b)].mean(axis=0).tolist() for a, b in zip(edges[:-1], edges[1:])]
    return radial


def describe(image):
    values = image.reshape(-1, 3)
    return {'mean': values.mean(axis=0).tolist(), 'rms': np.sqrt(np.mean(values ** 2, axis=0)).tolist(),
            'covariance': np.cov(values.T).tolist(), 'radial_psd': spectrum(image)}


def controls(side, seed, rms):
    y, x = np.mgrid[:side, :side]
    clean = {'flat': np.full((side, side), .5), 'ramp': .2 + .6*x/(side-1),
             'edge': np.where(x < side//2, .3, .7),
             'fine_lines': .5 + .1*((x//2) % 2 * 2 - 1),
             'one_pixel_lines': .5 + .1*(x % 2 * 2 - 1),
             'texture': .5 + .08*np.sin(x*.7)*np.cos(y*.5)}
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.standard_normal((side, side, 3)), (.7, .7, 0), mode='wrap')
    noise -= noise.mean(axis=(0, 1), keepdims=True)
    noise *= rms / np.sqrt(np.mean(noise ** 2, axis=(0, 1), keepdims=True))
    return {name: np.repeat(a[..., None], 3, axis=2) for name, a in clean.items()}, noise


def main():
    started = time.perf_counter()
    config_path = ROOT/'configs/fable_scan_residual_pilot_v1.json'
    cfg = json.loads(config_path.read_text())
    inventory_path = ROOT/cfg['source_inventory']
    assert digest(inventory_path) == cfg['source_inventory_sha256']
    inventory = json.loads(inventory_path.read_text())
    assert len(inventory['files']) == cfg['maximum_sources']
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    rows, examples = [], []
    for source in inventory['files']:
        path = ROOT/source['path']
        assert digest(path) == source['sha256']
        array = tifffile.imread(path)
        assert list(array.shape) == source['shape']
        h, w, _ = array.shape
        n = cfg['patch_side']
        for fy in cfg['grid_fractions']:
            for fx in cfg['grid_fractions']:
                top, left = int(h*fy)-n//2, int(w*fx)-n//2
                patch = rgb_codes(array[top:top+n, left:left+n])
                estimates = {m: residual(patch, m) for m in cfg['methods']}
                rows.append({'source': path.name, 'stock_metadata': source['stock_label_from_metadata'],
                             'box_xyxy': [left, top, left+n, top+n], 'mean_code': patch.mean(axis=(0, 1)).tolist(),
                             'methods': {m: describe(r) for m, r in estimates.items()}})
                if fy == .5 and fx == .5:
                    examples.append((path.name, patch, estimates))
        del array
    clean, noise = controls(cfg['patch_side'], cfg['seed'], cfg['synthetic_noise_rms'])
    checks = []
    for name, image in clean.items():
        for method in cfg['methods']:
            zero = residual(image, method)
            estimated = residual(image+noise, method)
            checks.append({'scene': name, 'method': method,
                           'clean_structure_leakage_rms': float(np.sqrt(np.mean(zero**2))),
                           'known_noise_relative_mse': float(np.mean((estimated-noise)**2)/np.mean(noise**2))})
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    freq = (np.linspace(0, .5, 33)[:-1]+np.linspace(0, .5, 33)[1:])/2
    for ax, (name, patch, estimates) in zip(axes.flat, examples):
        for method, values in estimates.items():
            ax.semilogy(freq, np.asarray(spectrum(values)).mean(axis=1), label=method)
        ax.set_title(name.replace('GrainScan', '\n'), fontsize=9)
        ax.set_xlabel('cycles / scanner pixel')
        ax.set_ylabel('mean RGB residual PSD')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Fixed center crops: extractor sensitivity, not physical grain calibration')
    fig.savefig(out/'extractor_sensitivity.png', dpi=140)
    plt.close(fig)
    report = {'status': 'DIAGNOSTIC_COMPLETE_NOT_TRAINING_ADMISSION', 'config_sha256': digest(config_path),
              'entry_sha256': digest(Path(__file__)), 'sources': len(inventory['files']), 'patches': len(rows),
              'scan_results': rows, 'synthetic_controls': checks, 'wall_seconds': time.perf_counter()-started,
              'limits': ['scanner code values, not calibrated optical density', 'nine patches per scan are correlated observations',
                         'simple filter sensitivity, not validation of BM3D or learned denoising', 'real residuals have no clean ground truth',
                         'PSD is second-order only; no perceptual or film stock fidelity claim']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'sources': report['sources'], 'patches': report['patches'], 'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    main()

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/fable_cdfe68_raw_v1.json'
config = json.loads(CONFIG.read_text())
os.environ['OMP_NUM_THREADS'] = str(config['threads'])
os.environ['MKL_NUM_THREADS'] = str(config['threads'])

import numpy as np
from PIL import Image

from src.preprocess.fable_canonical_raw import render_canonical_raw, linear16_to_q8


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = ROOT / 'outputs/u5_r2bq0s1_fivek_casebank_acquisition_v1/run_a/manifest.json'
    rows = json.loads(manifest.read_text())['rows']
    names = ['a0005-jn_2007_05_10__564', 'a0064-_DSC7889']
    fixtures = [next(r for r in rows if r['source_name'] == name) for name in names]
    for row in fixtures:
        if digest(row['dng_path']) != row['dng_sha256']:
            raise ValueError('consumed RAW differs from historical pinned source')
    out = ROOT / 'outputs/fable_cdfe68_renderer_v1/consumed_audit'
    out.mkdir(exist_ok=False)
    results = []
    for row in fixtures:
        path = Path(row['dng_path'])
        linear, metadata = render_canonical_raw(path, config)
        repeated, _ = render_canonical_raw(path, config)
        equal = bool(np.array_equal(linear, repeated))
        del repeated
        if not equal:
            raise ValueError('fixed RAW rendering is not reproducible')
        codes = linear16_to_q8(linear)
        file = out / (row['source_name'] + '.png')
        Image.fromarray(codes).save(file)
        result = {'identity': row['source_name'], 'source_sha256': row['dng_sha256'],
                  'metadata': metadata, 'repeat_identical': equal, 'canonical_png': str(file.relative_to(ROOT)),
                  'canonical_png_sha256': digest(file),
                  'linear_array_sha256': hashlib.sha256(linear.tobytes()).hexdigest()}
        if row['source_name'] == names[1]:
            half, _ = render_canonical_raw(path, config, diagnostic_raw_scale=.5)
            eligible = (linear >= 1000) & (linear <= 20000)
            ratio = float(np.median(half[eligible].astype(np.float64) / linear[eligible]))
            if not .48 <= ratio <= .52:
                raise ValueError(f'half-signal radiometric diagnostic failed: {ratio}')
            control = json.loads(json.dumps(config))
            control['params']['no_auto_bright'] = False
            auto, _ = render_canonical_raw(path, control)
            auto_half, _ = render_canonical_raw(path, control, diagnostic_raw_scale=.5)
            control_eligible = (auto >= 1000) & (auto <= 50000)
            auto_ratio = float(np.median(auto_half[control_eligible].astype(np.float64) / auto[control_eligible]))
            result['half_signal_diagnostic'] = {'median_ratio': ratio, 'eligible_channels': int(eligible.sum()),
                'expected': [.48, .52], 'passed': True, 'auto_bright_control_median_ratio': auto_ratio,
                'control_scope': 'Diagnostic config only; canonical config unchanged.',
                'limits': 'AHD need not be exactly exposure-equivariant; this is a median response check, not pixelwise proof.'}
        results.append(result)
        del linear, codes
    report = {'status': 'CONSUMED_RAW_ENGINEERING_ONLY', 'config_sha256': digest(CONFIG),
              'script_sha256': digest(__file__), 'module_sha256': digest(ROOT / 'src/preprocess/fable_canonical_raw.py'),
              'source_manifest_sha256': digest(manifest), 'fixtures': results, 'threads': config['threads'],
              'optimizer_steps': 0, 'new_data_acquired': 0,
              'claim_ceiling': 'Reproducible generic canonical rendering and bounded radiometric check; not physical calibration, independent validation or learned quality.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

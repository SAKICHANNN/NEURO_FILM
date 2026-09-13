import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from skimage.restoration import denoise_nl_means

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pilot_base', ROOT/'scripts/run_fable_scan_residual_pilot.py')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_blueneg_residual_v1.json'
    cfg = json.loads(cp.read_text())
    assert base.digest(Path(base.__file__)) == cfg['base_entry_sha256']
    assert base.digest(ROOT/cfg['acquisition_config']) == cfg['acquisition_config_sha256']
    acquisition = json.loads((ROOT/cfg['acquisition_config']).read_text())
    root = ROOT/acquisition['output_root']
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    rows, formats = [], []
    methods = ['gaussian_sigma1', 'gaussian_sigma2'] + [f'nlm_h{h}' for h in cfg['nlm_h']]
    side = cfg['side']
    for source in acquisition['files']:
        path = root/source['remote_path']
        if source['role'] != 'development':
            assert not path.exists()
            continue
        assert base.digest(path) == source['sha256']
        with tifffile.TiffFile(path) as tiff:
            index, series = max(((i, s) for i, s in enumerate(tiff.series)
                if len(s.shape) == 3 and s.shape[-1] == 3 and s.dtype == np.uint16),
                key=lambda item: item[1].shape[0]*item[1].shape[1])
            assert series.pages[0].is_memmappable and series.shape[0] > 6000
            formats.append({'roll_id': source['roll_id'], 'series': index, 'shape': series.shape})
        array = tifffile.memmap(path, series=index, mode='r')
        sheet = Image.new('RGB', (side*5, (side+35)*9), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, (fy, fx) in enumerate((y, x) for y in cfg['grid_fractions'] for x in cfg['grid_fractions']):
            top, left = int(array.shape[0]*fy)-side//2, int(array.shape[1]*fx)-side//2
            patch = array[top:top+side, left:left+side].astype(np.float64)/65535.
            estimates = {}
            for name in methods:
                if name.startswith('nlm_'):
                    h = float(name.split('_h')[1])
                    estimates[name] = patch-denoise_nl_means(patch, h=h, patch_size=cfg['patch_size'],
                        patch_distance=cfg['patch_distance'], sigma=cfg['sigma'], fast_mode=cfg['fast_mode'],
                        preserve_range=True, channel_axis=-1)
                else:
                    estimates[name] = base.residual(patch, name)
            rows.append({'roll_id': source['roll_id'], 'box_xyxy': [left, top, left+side, top+side],
                'mean_code': patch.mean(axis=(0, 1)).tolist(),
                'methods': {name: base.describe(value) for name, value in estimates.items()}})
            gray = patch.mean(axis=2)
            preview = 1-(gray-gray.min())/max(float(np.ptp(gray)), 1e-12)
            panels = [('SOURCE stretched/inverted', np.repeat(preview[..., None], 3, axis=2))]
            panels += [(name+' residual x10', .5+cfg['residual_display_gain']*value) for name, value in estimates.items()]
            for col, (label, values) in enumerate(panels):
                draw.text((col*side+3, j*(side+35)+3), f'{j}: {label}', fill='black')
                sheet.paste(Image.fromarray(np.round(np.clip(values, 0, 1)*255).astype(np.uint8)), (col*side, j*(side+35)+35))
        sheet.save(out/f'{source["roll_id"]}_residuals.png')
        del array
    assert len(rows) == 27
    report = {'status': 'NATURAL_SCAN_DIAGNOSTIC_NOT_GRAIN_LABELS', 'rows': rows, 'formats': formats,
        'config_sha256': base.digest(cp), 'entry_sha256': base.digest(Path(__file__)),
        'wall_seconds': time.perf_counter()-start,
        'limits': ['LINEAR_RAW scanner RGB codes, not calibrated density or display RGB',
            'No paired clean signal; residual structure cannot be causally separated from signal-dependent grain',
            'Source preview grayscale stretched/inverted solely for viewing; residuals displayed fixed x10 clipped, stats unclipped',
            '27 correlated crops from 3 development rolls; two validation scans remain absent']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    main()

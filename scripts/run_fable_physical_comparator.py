import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.film_physics.bounded_photographic_runtime import BoundedPhotographicCpuRuntime
from src.film_physics.spatial_response import apply_scanner_mtf


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_physical_comparator_v1.json'
    cfg = json.loads(cp.read_text())
    profile = ROOT/cfg['profile']['path']
    assert digest(profile) == cfg['profile']['sha256']
    runtime = BoundedPhotographicCpuRuntime.load(profile, expected_bundle_sha256=cfg['profile']['bundle_sha256'])
    out = ROOT/cfg['output']
    out.mkdir(exist_ok=False)
    side = cfg['side']
    sheet = Image.new('RGB', (side*3, (side+30)*len(cfg['sources'])), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for i, source in enumerate(cfg['sources']):
        path = ROOT/source['source_path']
        assert digest(path) == source['source_sha256']
        with Image.open(path) as im:
            assert list(im.size) == source['source_size']
            x, y, w, h = source['crop_xywh']
            x += (w-side)//2
            y += (h-side)//2
            encoded = np.asarray(im.convert('RGB').crop((x, y, x+side, y+side)), dtype=np.float32)/255
        linear = np.ascontiguousarray(np.where(encoded <= .04045, encoded/12.92, ((encoded+.055)/1.055)**2.4), dtype=np.float32)
        physical, receipt = runtime.render(linear, source_index=i)
        scanner = apply_scanner_mtf(linear.astype(np.float64), runtime.components.scanner_profile)
        values = [linear, scanner, physical]
        row = {'id': source['sample_id'], 'box_xyxy': [x, y, x+side, y+side], 'receipt': receipt, 'arms': {}}
        for col, (name, value) in enumerate(zip(cfg['arms'], values)):
            assert np.all(np.isfinite(value))
            row['arms'][name] = {'min': float(value.min()), 'max': float(value.max()),
                'linear_rmse_from_source': float(np.sqrt(np.mean((value-linear)**2)))}
            np.save(out/f'{source["sample_id"]}_{name}.npy', value)
            display = np.where(value <= .0031308, 12.92*value, 1.055*np.maximum(value, 0)**(1/2.4)-.055)
            draw.text((col*side+3, i*(side+30)+3), source['sample_id']+' '+name, fill='black')
            sheet.paste(Image.fromarray(np.round(np.clip(display, 0, 1)*255).astype(np.uint8)), (col*side, i*(side+30)+30))
        rows.append(row)
    sheet.save(out/'comparison.png')
    report = {'config_sha256': digest(cp), 'entry_sha256': digest(Path(__file__)), 'rows': rows,
        'wall_seconds': time.perf_counter()-start, 'status': 'UNPROMOTED_COMPARATOR_ONLY',
        'limits': ['previously consumed development images, not independent validation',
            'generic profile; no real-film calibration or reference conditioning',
            '256px crops do not establish full-frame texture continuity or semantics']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    main()

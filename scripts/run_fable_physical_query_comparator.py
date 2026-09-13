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
    cp = ROOT / 'configs/fable_physical_query_comparator_v1.json'
    cfg = json.loads(cp.read_text())
    sp = ROOT / cfg['sources_config']
    pp = ROOT / cfg['profile']['path']
    if digest(sp) != cfg['sources_sha256'] or digest(pp) != cfg['profile']['sha256']:
        raise ValueError('configuration provenance mismatch')
    sources = [s for s in json.loads(sp.read_text())['sources'] if s['role'] == 'query']
    if [s['index'] for s in sources] != cfg['query_indices']:
        raise ValueError('query identity mismatch')
    for source in sources:
        if digest(ROOT / source['path']) != source['sha256']:
            raise ValueError('query hash mismatch')
    runtime = BoundedPhotographicCpuRuntime.load(pp, expected_bundle_sha256=cfg['profile']['bundle_sha256'])
    out = ROOT / cfg['output']
    out.mkdir(exist_ok=False)
    rows = []
    for source in sources:
        start = time.perf_counter()
        folder = out / f"query{source['index']}"
        folder.mkdir()
        with Image.open(ROOT / source['path']) as im:
            if list(im.size) != source['size']:
                raise ValueError('query size mismatch')
            original = np.asarray(im.convert('RGB'))
        encoded = original.astype(np.float32) / 255
        linear = np.ascontiguousarray(np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4))
        physical, receipt = runtime.render(linear, source_index=source['index'])
        scanner = apply_scanner_mtf(linear.astype(np.float64), runtime.components.scanner_profile)
        overview = Image.new('RGB', (960, 350), 'white')
        native = Image.new('RGB', (768, 286), 'white')
        row = {'query': source['index'], 'source_sha256': source['sha256'], 'receipt': receipt, 'arms': {}}
        for col, (name, value) in enumerate(zip(cfg['arms'], (linear, scanner, physical), strict=True)):
            if value.shape != linear.shape or not np.isfinite(value).all():
                raise ValueError('invalid physical output')
            display = np.where(value <= .0031308, 12.92 * value, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - .055)
            codes = np.floor(np.clip(display, 0, 1) * 255 + .5).astype(np.uint8)
            if name == 'source' and not np.array_equal(codes, original):
                raise ValueError('sRGB roundtrip changed source codes')
            image = Image.fromarray(codes)
            image.save(folder / f'{name}.png')
            np.save(folder / f'{name}.npy', value)
            thumb = image.copy()
            thumb.thumbnail((320, 320))
            overview.paste(thumb, (col * 320, 30))
            ImageDraw.Draw(overview).text((col * 320 + 3, 3), name, fill='black')
            x, y = (image.width - 256) // 2, (image.height - 256) // 2
            native.paste(image.crop((x, y, x + 256, y + 256)), (col * 256, 30))
            ImageDraw.Draw(native).text((col * 256 + 3, 3), name, fill='black')
            row['arms'][name] = {'linear_rmse': float(np.sqrt(np.mean((value - linear) ** 2))),
                'code_rmse': float(np.sqrt(np.mean((codes.astype(float) - original) ** 2))),
                'png_sha256': digest(folder / f'{name}.png'), 'npy_sha256': digest(folder / f'{name}.npy')}
        overview.save(folder / 'overview.png')
        native.save(folder / 'native.png')
        row['wall_seconds'] = time.perf_counter() - start
        rows.append(row)
        (folder / 'receipt.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        print(json.dumps({'query': source['index'], 'seconds': row['wall_seconds']}), flush=True)
    report = {'status': 'FULL_FRAME_RENDERED_VISUAL_PENDING_UNPROMOTED', 'config_sha256': digest(cp),
        'entry_sha256': digest(Path(__file__)), 'rows': rows,
        'limits': ['four reused display JPEG sources, no independent validation',
                   'generic physical-inspired profile, no stock calibration or learned reference conditioning',
                   'not part of frozen synthetic photometry score; no semantic guarantee']}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

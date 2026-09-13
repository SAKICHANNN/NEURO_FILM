import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.fable_reference_photometry import quantize8, transform


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp = ROOT/'configs/fable_reference_photometry_v1.json'
    cfg = json.loads(cp.read_text())
    root = ROOT/cfg['output']
    receipt = json.loads((root/'held_predictions/report.json').read_text())
    assert digest(root/'fit_head/model.json') == receipt['model_sha256']
    assert digest(root/'held_predictions/predictions.npz') == receipt['predictions_sha256']
    assert digest(root/'held_features/rows.json') == receipt['rows_sha256']
    sp = ROOT/cfg['sources_config']
    assert digest(sp) == cfg['sources_config_sha256']
    sources = [s for s in json.loads(sp.read_text())['sources'] if s['role'] == 'query']
    rows = json.loads((root/'held_features/rows.json').read_text())
    chosen = [(i, row) for i, row in enumerate(rows) if row['cell'] == 'both_held'
              and row['pair_id'] == cfg['visual_pair_id'] and row['repeat'] == cfg['visual_dequantization_repeat']]
    assert len(chosen) == 4
    out = root/'visuals'
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(2026091701)
    key, manifest = [], []
    predictions = np.load(root/'held_predictions/predictions.npz', allow_pickle=False)
    for i, row in chosen:
        for source in sources:
            path = ROOT/source['path']
            assert digest(path) == source['sha256']
            with Image.open(path) as im:
                source_image = im.convert('RGB')
                rgb = np.asarray(source_image, dtype=np.float64)/255
            oracle = transform(rgb, np.asarray(row['target']), slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit'])
            predicted = transform(rgb, predictions[receipt['primary']][i], slope_limit=cfg['slope_limit'], offset_limit=cfg['offset_limit'])
            candidates = [('oracle', oracle), ('prediction', predicted)]
            if rng.integers(2):
                candidates.reverse()
            case = f'case{len(manifest):02d}'
            directory = out/case
            directory.mkdir()
            images = [('source', source_image)] + [(label, Image.fromarray(np.floor(quantize8(value)*255+.5).astype(np.uint8)))
                for label, (_, value) in zip(['A', 'B'], candidates)]
            overview = Image.new('RGB', (960, 350), 'white')
            native = Image.new('RGB', (768, 286), 'white')
            for col, (label, image) in enumerate(images):
                image.save(directory/f'{label}.png')
                thumb = image.copy()
                thumb.thumbnail((320, 320))
                overview.paste(thumb, (col*320, 30))
                ImageDraw.Draw(overview).text((col*320+3, 3), case+' '+label, fill='black')
                x, y = (image.width-256)//2, (image.height-256)//2
                native.paste(image.crop((x, y, x+256, y+256)), (col*256, 30))
                ImageDraw.Draw(native).text((col*256+3, 3), case+' '+label+' native center', fill='black')
            overview.save(directory/'overview.png')
            native.save(directory/'native.png')
            manifest.append({'case': case, 'source_index': source['index'], 'reference_row': i,
                'donor_id': row['donor_id'], 'draw_id': row['draw_id'],
                'hashes': {p.name: digest(p) for p in directory.glob('*.png')}})
            key.append({'case': case, 'A': candidates[0][0], 'B': candidates[1][0]})
    assert len(manifest) == 16
    (out/'manifest.json').write_text(json.dumps({'cases': manifest, 'model_sha256': receipt['model_sha256'],
        'entry_sha256': digest(Path(__file__)), 'scope': 'Fixed first held antithetic pair, both donors, four queries; labels hidden'}, indent=2), encoding='utf-8')
    (out/'key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(manifest), 'status': 'RENDERED_NOT_REVIEWED'}))


if __name__ == '__main__':
    main()

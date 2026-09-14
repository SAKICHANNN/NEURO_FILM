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
    cp = ROOT/'configs/fable_paired_contrast_v1.json'
    cfg = json.loads(cp.read_text())
    old_cfg = json.loads((ROOT/'configs/fable_reference_photometry_v1.json').read_text())
    root = ROOT/cfg['output']
    receipt = json.loads((root/'fresh_predictions/report.json').read_text())
    assert digest(root/'fit_head/model.json') == receipt['model_sha256']
    assert digest(root/'fresh_predictions/predictions.npz') == receipt['predictions_sha256']
    assert digest(cp) == receipt['config_sha256']
    sp = ROOT/old_cfg['sources_config']
    assert digest(sp) == old_cfg['sources_config_sha256']
    sources = [s for s in json.loads(sp.read_text())['sources'] if s['role'] == 'query']
    data = np.load(root/'fresh_predictions/predictions.npz', allow_pickle=False)
    chosen = [(i, {'donor_id': int(donor), 'draw_id': int(data['draw_ids'][i]), 'target': data['targets'][i]})
              for i, donor in enumerate(data['donor_ids']) if donor >= 12 and data['pair_ids'][i] == cfg['visual']['pair_id']]
    assert len(chosen) == 8
    out = root/'visuals'
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(2026091901)
    key, manifest = [], []
    predictions = np.load(root/'fresh_predictions/predictions.npz', allow_pickle=False)
    for i, row in chosen:
        for source in sources:
            path = ROOT/source['path']
            assert digest(path) == source['sha256']
            with Image.open(path) as im:
                source_image = im.convert('RGB')
                rgb = np.asarray(source_image, dtype=np.float64)/255
            oracle = transform(rgb, np.asarray(row['target']), slope_limit=old_cfg['slope_limit'], offset_limit=old_cfg['offset_limit'])
            predicted = transform(rgb, predictions['after_only'][i], slope_limit=old_cfg['slope_limit'], offset_limit=old_cfg['offset_limit'])
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
    assert len(manifest) == 32
    (out/'manifest.json').write_text(json.dumps({'cases': manifest, 'model_sha256': receipt['model_sha256'],
        'entry_sha256': digest(Path(__file__)), 'scope': 'Fixed first held antithetic pair, four new donors, four queries; labels hidden'}, indent=2), encoding='utf-8')
    (out/'key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(manifest), 'status': 'RENDERED_NOT_REVIEWED'}))


if __name__ == '__main__':
    main()

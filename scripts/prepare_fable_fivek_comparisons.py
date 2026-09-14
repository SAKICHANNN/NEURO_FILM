import csv
import hashlib
import json
import inspect
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/fable_source_population_bridge_v1'


def hashes(image):
    width, height = image.size
    cw, ch = max(1, int(.8 * width)), max(1, int(.8 * height))
    origins = [(0, 0), (width-cw, 0), (0, height-ch), (width-cw, height-ch),
               ((width-cw)//2, (height-ch)//2)]
    views = [image] + [image.crop((x, y, x+cw, y+ch)) for x, y in origins]
    result = []
    for view in views:
        values = np.asarray(view.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
        result.append(int.from_bytes(np.packbits(values[:, :-1] > values[:, 1:]).tobytes(), 'big'))
    return result


def main():
    cfg = json.loads((ROOT / 'configs/fable_fivek_preview_protocol_v1.json').read_text())
    config_sha = hashlib.sha256((ROOT / 'configs/fable_fivek_preview_protocol_v1.json').read_bytes()).hexdigest()
    if pillow_version != cfg['comparison']['pillow_version']:
        raise ValueError('Pillow version differs from frozen comparison')
    if hashlib.sha256((ROOT / cfg['cohort']).read_bytes()).hexdigest() != cfg['cohort_sha256']:
        raise ValueError('cohort changed')
    cohort = json.loads((ROOT / cfg['cohort']).read_text())
    ledger = json.loads((BASE / 'previews/ledger.json').read_text())
    if ledger['status'] != 'FETCH_COMPLETE' or ledger['protocol_sha256'] != config_sha:
        raise ValueError('fixed download must be complete')
    rows = []
    for kind in ('history_restore', 'candidates'):
        for r in cohort[kind]:
            saved = ledger['objects'][kind+'/'+r['dataset_id']]
            rows.append({'identity': 'fivek/'+r['source_name'], 'pool': 'candidate' if kind == 'candidates' else 'history',
                         'path': saved['path'], 'expected_sha256': saved['sha256']})
    freeze = ROOT / 'outputs/fivek_auto_optimize/freeze_v1/manifest.csv'
    for r in csv.DictReader(freeze.open(encoding='utf-8-sig')):
        rows.append({'identity': 'fivek/'+r['source_name'], 'pool': 'history', 'path': r['expert_preview']})
    batch = ROOT / 'outputs/u5_r2bq0s1_fivek_casebank_acquisition_v1/run_a/manifest.json'
    for r in json.loads(batch.read_text())['rows']:
        rows.append({'identity': 'fivek/'+r['source_name'], 'pool': 'history', 'path': r['expert_c_path'],
                     'expected_sha256': r['expert_c_sha256']})
    old = ROOT / 'outputs/fable_contrast_source_preparation_v1/historical_preview_paths.json'
    for r in json.loads(old.read_text())['rows']:
        ident = r['source_identity']
        key = str(ident.get('repository_id') or ident.get('raw_sha256') or ident.get('source_url'))
        if key == 'None':
            raise ValueError('unresolved historical capture identity')
        rows.append({'identity': 'rawpixls/'+key, 'pool': 'history', 'path': r['path']})
    for folder in ['outputs/fable_contrast_source_preparation_v1/decoded', 'outputs/fable_canonical_source_metadata_v1/witness']:
        for path in sorted((ROOT / folder).glob('*.jpg')):
            if path.stem.isdigit():
                rows.append({'identity': 'rawpixls/'+path.stem, 'pool': 'history', 'path': str(path)})
    output = BASE / 'comparisons'
    output.mkdir(exist_ok=True)
    descriptors = output / 'descriptors.jsonl'
    binding = {'protocol_sha256': config_sha, 'pillow_version': pillow_version,
               'hash_function_sha256': hashlib.sha256(inspect.getsource(hashes).encode()).hexdigest()}
    binding_path = output / 'cache_binding.json'
    if descriptors.exists() and (not binding_path.exists() or json.loads(binding_path.read_text()) != binding):
        raise ValueError('unbound or changed cache requires explicit verification')
    binding_path.write_text(json.dumps(binding, indent=2), encoding='utf-8')
    done = {}
    if descriptors.exists():
        for line in descriptors.read_text().splitlines():
            r = json.loads(line)
            done[r['path']] = r
    with descriptors.open('a', encoding='utf-8') as stream:
        for row in rows:
            path = ROOT / row['path']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get('expected_sha256', digest) != digest:
                raise ValueError('source changed: '+str(path))
            if row['path'] in done:
                cached = done[row['path']]
                if any(cached[k] != row[k] for k in ('identity', 'pool')) or cached['sha256'] != digest:
                    raise ValueError('cache source identity or bytes changed')
                continue
            with Image.open(path) as source:
                values = hashes(source)
                preview = source.convert('RGB')
                preview.thumbnail((300, 300), Image.Resampling.LANCZOS)
                thumb = output / (digest+'.jpg')
                preview.save(thumb, quality=92)
            record = {**row, 'sha256': digest, 'hashes': values, 'review_preview': str(thumb)}
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            done[row['path']] = record
    selected = [done[r['path']] for r in rows]
    candidates = [r for r in selected if r['pool'] == 'candidate']
    result = []
    arrays = np.asarray([r['hashes'] for r in selected], dtype=np.uint64)
    for candidate in candidates:
        own = np.asarray(candidate['hashes'], dtype=np.uint64)
        distances = np.bitwise_count(arrays[:, :, None] ^ own[None, None, :]).min(axis=(1, 2))
        full = np.bitwise_count(arrays[:, 0] ^ own[0])
        groups = {}
        for i, other in enumerate(selected):
            if other['identity'] == candidate['identity']:
                continue
            key = (int(distances[i]), int(full[i]), other['identity'], other['path'])
            if other['identity'] not in groups or key < groups[other['identity']][0]:
                groups[other['identity']] = (key, other)
        ranked = sorted(groups.values(), key=lambda v: v[0])
        history = [v for v in ranked if v[1]['pool'] == 'history']
        peers = [v for v in ranked if v[1]['pool'] == 'candidate']
        chosen = history[:5] + peers[:5]
        if len(chosen) < 10:
            chosen += [v for v in ranked if v not in chosen][:10-len(chosen)]
        result.append({'candidate': candidate, 'neighbors': [{'crop_distance': k[0], 'full_distance': k[1],
                       'review_flag': k[0] <= 4, 'source': r} for k, r in chosen], 'adjudication': 'NOT_REVIEWED'})
    (output / 'neighbors.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'candidate_count': len(candidates), 'history_capture_count':len({r['identity'] for r in selected if r['pool']=='history'}),
                      'descriptor_rows':len(selected), 'status':'COMPARISONS_PREPARED_NOT_VISUALLY_REVIEWED'}))


if __name__ == '__main__':
    main()

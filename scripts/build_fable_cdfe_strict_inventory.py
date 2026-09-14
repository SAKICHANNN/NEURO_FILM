import hashlib
import json
from pathlib import Path


def build(root: Path) -> dict:
    hashes = {}

    def read(relative):
        blob = (root / relative).read_bytes()
        hashes[relative] = hashlib.sha256(blob).hexdigest()
        return [json.loads(line) for line in blob.splitlines()] if relative.endswith('.jsonl') else json.loads(blob)

    ledger = read('outputs/fable_cdfe68_exposure_v1/ledger.json')
    prefilter = read('outputs/fable_cdfe68_exposure_v1/strict_metadata_prefilter.json')
    graph = read('outputs/fable_cdfe68_exposure_v1/expanded_metadata_graph.json')
    masks = read('outputs/fable_cdfe68_native_candidates_v1/query_masks_v1/manifest.json')
    renders = {}
    for folder, count in [('native_candidates', 114), ('native_fit', 768)]:
        base = 'outputs/fable_cdfe68_' + folder + '_v1/'
        complete = read(base + 'complete.json')
        records = read(base + 'renders.jsonl')
        frame = read(base + 'frame.json')
        if (complete['count'] != count or len(records) != count
                or hashes[base + 'renders.jsonl'] != complete['renders_sha256']
                or [r['identity'] for r in records] != frame['identities']):
            raise ValueError('incomplete or changed native rendering frame')
        for row in records:
            if row['identity'] in renders:
                raise ValueError('fit and fresh render identities overlap')
            renders[row['identity']] = row
    components = {i: c for c, members in prefilter['components'].items() for i in members}
    candidates = {r['identity']: r for r in prefilter['rows']}
    coverage = {}
    for roi in masks['rows']:
        identity = roi['identity']
        if roi['native_sha256'] != renders[identity].get('native_sha256') or not roi['reviewed']:
            raise ValueError('query evidence does not bind native render')
        blob = Path(roi['mask_path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != roi['mask_sha256']:
            raise ValueError('query mask changed')
        coverage.setdefault(identity, set()).add(roi['category'])
    rows = []
    for source in ledger['rows']:
        identity = source['identity']
        models = graph['metadata'].get(identity, {}).get('models', [])
        camera = models[0] if len(models) == 1 else 'UNRESOLVED_CAMERA'
        rendered = renders.get(identity, {})
        eligible = (rendered.get('status') == 'LOCKED_NATIVE_RENDER_PASSED_NOT_ROLE_ADMISSION'
                    and len(models) == 1)
        rows.append({'identity': identity, 'exposure': source['exposure'],
                     'component': components[identity], 'camera': camera,
                     'candidate_rank': hashlib.sha256(('CDFE68-M1-allocation-v1:' + identity).encode()).hexdigest(),
                     'technical_eligible': eligible,
                     'history_separation_established': candidates.get(identity, {}).get('metadata_prefilter_pass') is True,
                     'query_coverage_reviewed': identity in coverage,
                     'query_coverage': sorted(coverage.get(identity, [])),
                     'native_sha256': rendered.get('native_sha256')})
    return {'status': 'STRICT_INVENTORY_PENDING_ALLOCATION', 'source_sha256': hashes,
            'rank_rule': 'SHA256 UTF8 CDFE68-M1-allocation-v1: plus identity; camera uses camera: prefix. Frozen without model outcomes.',
            'rows': rows, 'smoke_ids': [],
            'camera_ranks': {c: hashlib.sha256(('CDFE68-M1-allocation-v1:camera:' + c).encode()).hexdigest()
                             for c in sorted({r['camera'] for r in rows})}}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    with (root / 'outputs/fable_cdfe68_exposure_v1/strict_role_inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2)

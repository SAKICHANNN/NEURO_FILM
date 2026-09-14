import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build(root: Path) -> dict:
    hashes = {}

    def read(relative):
        blob = (root / relative).read_bytes()
        hashes[relative] = hashlib.sha256(blob).hexdigest()
        return [json.loads(line) for line in blob.splitlines()] if relative.endswith('.jsonl') else json.loads(blob)

    def verify_file(path, expected):
        with Path(path).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'evidence digest differs: {path}')

    def verify_dependencies(document):
        for relative, expected in document.get('source_sha256', {}).items():
            verify_file(root / relative, expected)

    ledger = read('outputs/fable_cdfe68_exposure_v1/ledger.json')
    prefilter = read('outputs/fable_cdfe68_exposure_v1/strict_metadata_prefilter.json')
    graph = read('outputs/fable_cdfe68_exposure_v1/expanded_metadata_graph.json')
    masks = read('outputs/fable_cdfe68_native_candidates_v1/query_masks_v1/manifest.json')
    prefix = read('outputs/fable_cdfe68_exposure_v1/candidate_prefix.json')
    for document in [prefilter, graph, prefix]:
        verify_dependencies(document)
    if [r['identity'] for r in prefix['rows']] != [r['identity'] for r in prefilter['rows']]:
        raise ValueError('prefilter differs from frozen prefix')
    renders = {}
    for folder, count in [('native_candidates', 114), ('native_fit', 768)]:
        base = 'outputs/fable_cdfe68_' + folder + '_v1/'
        complete = read(base + 'complete.json')
        records = read(base + 'renders.jsonl')
        frame = read(base + 'frame.json')
        verify_dependencies(frame)
        if folder == 'native_candidates' and frame['identities'] != [
                r['identity'] for r in prefilter['rows'] if r['metadata_prefilter_pass'] is True]:
            raise ValueError('fresh frame differs from prefilter')
        if (complete['count'] != count or len(records) != count
                or hashes[base + 'renders.jsonl'] != complete['renders_sha256']
                or [r['identity'] for r in records] != frame['identities']):
            raise ValueError('incomplete or changed native rendering frame')
        for row in records:
            if row['identity'] in renders:
                raise ValueError('fit and fresh render identities overlap')
            renders[row['identity']] = row
            if row['status'] == 'LOCKED_NATIVE_RENDER_PASSED_NOT_ROLE_ADMISSION':
                verify_file(row['native_path'], row['native_sha256'])
    members = [i for group in prefilter['components'].values() for i in group]
    if len(members) != len(set(members)) or set(members) != {r['identity'] for r in ledger['rows']}:
        raise ValueError('components must partition the complete ledger')
    components = {i: c for c, members in prefilter['components'].items() for i in members}
    candidates = {r['identity']: r for r in prefilter['rows']}
    coverage = {}
    for roi in masks['rows']:
        identity = roi['identity']
        if roi['native_sha256'] != renders[identity].get('native_sha256') or roi['reviewed'] is not True:
            raise ValueError('query evidence does not bind native render')
        blob = Path(roi['mask_path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != roi['mask_sha256']:
            raise ValueError('query mask changed')
        verify_file(roi['crop_path'], roi['crop_sha256'])
        mask = np.load(roi['mask_path'], allow_pickle=False)
        crop = Image.open(roi['crop_path'])
        native = np.load(renders[identity]['native_path'], mmap_mode='r', allow_pickle=False)
        x0, y0, x1, y1 = roi['box_xyxy']
        if (not 0 <= x0 < x1 <= native.shape[1] or not 0 <= y0 < y1 <= native.shape[0]
                or mask.shape != (y1-y0, x1-x0) or crop.size != (x1-x0, y1-y0)
                or not np.isin(mask, [0, 1]).all() or not mask.any()
                or roi['category'] not in roi['observed_categories']):
            raise ValueError('invalid query region or category binding')
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

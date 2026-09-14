import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scripts.build_fable_cdfe_history_graph import components


def build(root):
    paths = ['outputs/fable_cdfe68_exposure_v1/history_graph.json',
             'outputs/fable_cdfe68_acquisition_v1/metadata.jsonl',
             'outputs/fable_cdfe68_exposure_v1/local_history_audit.jsonl',
             'outputs/fable_cdfe68_exposure_v1/final_local_metadata.jsonl',
             'docs/evidence/FABLE_CDFE68_RECORDED_METADATA_CHAIN_20260914.json',
             'configs/fable_fivek_camera_labels_v1.json',
             'outputs/fable_cdfe68_exposure_v1/ledger.json',
             'outputs/fable_cdfe68_exposure_v1/candidate_prefix.json']
    hashes, data = {}, []
    for path in paths:
        blob = (root / path).read_bytes()
        hashes[path] = hashlib.sha256(blob).hexdigest()
        data.append(list(map(json.loads, blob.splitlines())) if path.endswith('jsonl') else json.loads(blob))
    old, acquired, local, final, recorded, labels, ledger, prefix = data
    observations = defaultdict(list)
    for identity, value in old['metadata'].items():
        observations[identity].append({'model': value['model'], 'date': value['date'], 'source': paths[0]})

    def add(identity, fields, source):
        models = {f['fields'].get('Model') for f in fields} - {None, ''}
        dates = {f['fields'].get('DateTimeOriginal') for f in fields} - {None, ''}
        observations[identity].append({'models': sorted(models), 'dates': sorted(dates), 'source': source})

    for rows, source in [(acquired, paths[1]), (final, paths[3])]:
        for row in rows:
            add(row['identity'], row.get('ifd_metadata', []), source)
    for row in local:
        for original in row['originals']:
            add(row['identity'], original.get('metadata', []), paths[2])
    for row in recorded['rows']:
        f = row['fields']
        observations[row['identity']].append({'model': f.get('Exif.Image.Model'),
            'date': f.get('Exif.Photo.DateTimeOriginal'), 'source': paths[4]})
    aliases = {**labels['observed_labels'], **labels['additional_aliases']}
    grouped, summaries = defaultdict(list), {}
    for identity, records in sorted(observations.items()):
        models, dates = set(), set()
        for r in records:
            for model in r.get('models', [r.get('model')]):
                if model:
                    normalized = ' '.join(model.upper().split())
                    models.add(aliases.get(normalized, normalized))
            dates.update(d for d in r.get('dates', [r.get('date')]) if d)
        day = None
        if len(dates) == 1:
            try: day = datetime.strptime(next(iter(dates)), '%Y:%m:%d %H:%M:%S').date().isoformat()
            except ValueError: pass
        usable = len(models) == 1 and day is not None
        summaries[identity] = {'models': sorted(models), 'timestamps': sorted(dates), 'recorded_day': day,
            'unique_camera_day': usable, 'observations': records}
        if usable: grouped[(next(iter(models)), day)].append(identity)
    edges = list(old['edges'])
    for (camera, day), members in sorted(grouped.items()):
        edges.extend({'left': members[0], 'right': member, 'decision': 'CONSERVATIVE_CAMERA_DAY',
                      'camera': camera, 'day': day} for member in members[1:])
    ids = {r['identity'] for r in ledger['rows']}
    joined = components(ids, edges)
    anchors = {r['identity'] for r in ledger['rows'] if r['exposure'] != 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED'}
    blocked = {i for group in joined.values() if anchors.intersection(group) for i in group}
    candidate_ids = {r['identity'] for r in prefix['rows']}
    return {'status': 'EXPANDED_METADATA_GRAPH_NOT_FINAL_SEPARATION', 'source_sha256': hashes,
            'metadata': summaries, 'edges': edges, 'components': joined,
            'prefix_linked_to_history': sorted(candidate_ids & blocked),
            'prefix_missing_unique_camera_day': sorted(i for i in candidate_ids if not summaries.get(i, {}).get('unique_camera_day')),
            'limits': ['Literal normalized unknown camera labels used only for conservative exact grouping, not certified alias equivalence.',
                       'Underlying metadata warnings/evidence limitations remain; cross-date and unbounded historical relations still require disposition.',
                       'No remaining candidate automatically admitted from absence of edges.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    with (root / 'outputs/fable_cdfe68_exposure_v1/expanded_metadata_graph.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: len(result[k]) for k in ['metadata', 'components', 'prefix_linked_to_history', 'prefix_missing_unique_camera_day']}))

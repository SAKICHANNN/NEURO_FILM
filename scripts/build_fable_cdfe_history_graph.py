import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def components(identities, edges):
    parent = {identity: identity for identity in identities}

    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    for edge in edges:
        left, right = find(edge['left']), find(edge['right'])
        parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for identity in sorted(parent):
        groups[find(identity)].append(identity)
    return dict(sorted(groups.items()))


def camera_day(model, timestamp, aliases):
    camera = aliases.get(' '.join((model or '').upper().split()))
    try:
        day = datetime.strptime(timestamp or '', '%Y:%m:%d %H:%M:%S').date().isoformat()
    except ValueError:
        day = None
    return (camera, day) if camera and day else None


def build(root):
    paths = ['outputs/fable_cdfe68_exposure_v1/ledger.json',
             'outputs/fable_source_population_bridge_v1/raw_pilot_metadata_v1.json',
             'configs/fable_fivek_camera_labels_v1.json']
    blobs = [(root / path).read_bytes() for path in paths]
    ledger, pilot, labels = [json.loads(blob) for blob in blobs]
    aliases = {**labels['observed_labels'], **labels['additional_aliases']}
    metadata = {}
    for row in ledger['rows']:
        raw = row['recorded_raw_metadata']
        if raw:
            metadata[row['identity']] = {'model': raw['model'], 'date': raw['date'], 'origin': paths[0]}
    for row in pilot['rows']:
        if row['identity'] in metadata:
            raise ValueError('Unexpected overlapping history and pilot metadata')
        raw = row['metadata']
        metadata[row['identity']] = {'model': raw.get('Model'), 'date': raw.get('DateTimeOriginal'), 'origin': paths[1]}
    groups = defaultdict(list)
    unusable = []
    for identity, raw in sorted(metadata.items()):
        key = camera_day(raw['model'], raw['date'], aliases)
        if key is None:
            unusable.append(identity)
        else:
            groups[key].append(identity)
    edges = list(ledger['recorded_lineage_edges'])
    day_groups = []
    for (camera, day), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        day_groups.append({'camera': camera, 'recorded_day': day, 'members': members})
        edges.extend({'left': members[0], 'right': member, 'decision': 'CONSERVATIVE_DESIGN_UNION',
                      'camera': camera, 'recorded_day': day} for member in members[1:])
    exposures = {row['identity']: row['exposure'] for row in ledger['rows']}
    joined = components(exposures, edges)
    restricted = []
    for members in joined.values():
        anchors = [i for i in members if exposures[i] != 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED']
        if anchors:
            restricted.extend(i for i in members if exposures[i] == 'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED')
    return {'status': 'PARTIAL_HISTORY_GRAPH_NO_ADMISSION',
            'source_sha256': {p: hashlib.sha256(b).hexdigest() for p, b in zip(paths, blobs)},
            'metadata': metadata, 'unusable_camera_day': unusable, 'camera_day_groups': day_groups,
            'edges': edges, 'components': joined,
            'admission_only_candidates_linked_to_history_or_quarantine': sorted(restricted),
            'limits': ['Same camera/day is a conservative blocking rule, not evidence of same physical camera or session.',
                       'Missing metadata and unreviewed links remain unresolved; unlinked candidates are not certified fresh.',
                       'No new roles inherited from the terminated pilot; all pilot identities remain quarantine.',
                       'Recorded metadata only; no RAW asset or capture timestamp authenticity revalidation.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / 'outputs/fable_cdfe68_exposure_v1/history_graph.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: len(result[key]) for key in ['metadata', 'unusable_camera_day', 'camera_day_groups', 'edges', 'components', 'admission_only_candidates_linked_to_history_or_quarantine']}))

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


def build(root):
    paths = ['outputs/fable_cdfe68_exposure_v1/history_dispositions_draft.json',
             'outputs/fable_cdfe68_exposure_v1/local_history_audit.jsonl',
             'docs/evidence/FABLE_CDFE68_RECORDED_METADATA_CHAIN_20260914.json',
             'outputs/fable_cdfe68_exposure_v1/candidate_prefix.json',
             'outputs/fable_cdfe68_exposure_v1/history_locators_v2.json']
    blobs = [(root / path).read_bytes() for path in paths]
    draft = json.loads(blobs[0])
    local = {r['identity']: r for r in map(json.loads, blobs[1].splitlines())}
    recorded = {r['identity']: r for r in json.loads(blobs[2])['rows']}
    prefix = json.loads(blobs[3])
    locators = {r['identity']: r for r in json.loads(blobs[4])['rows']}
    rows = []
    for row in draft['rows']:
        identity = row['identity']
        attempted = bool(local[identity]['originals'])
        hashes = {e['record'].get('raw_sha256') or e['record'].get('dng_sha256') or e['record'].get('sha256')
                  for e in locators[identity]['evidence']}
        hashes.discard(None)
        urls = row['recorded_original_urls']
        if len({unquote(u) for u in urls}) != 1:
            raise ValueError(f'ambiguous original locator: {identity}')
        if identity in recorded:
            action = 'EXISTING_RECORDED_METADATA_REUSE'
        elif attempted:
            action = 'COMPLETED_LOCAL_READ_NO_FURTHER_READ'
        elif row['local_originals']:
            action = 'LOCAL_READ_ONCE_PENDING'
        else:
            action = 'REMOTE_ONCE_PENDING'
        selected = row['local_originals'][0] if action == 'LOCAL_READ_ONCE_PENDING' else None
        if selected and len(selected['expected_hashes']) != 1:
            raise ValueError(f'local identity hash unresolved: {identity}')
        if action == 'REMOTE_ONCE_PENDING' and len(hashes) > 1:
            raise ValueError(f'remote identity hash conflict: {identity}')
        rows.append({'identity': identity, 'action': action, 'selected_local': selected,
                     'url': sorted(urls, key=lambda u: (' ' in u, u))[0],
                     'expected_raw_sha256': next(iter(hashes)) if len(hashes) == 1 else None,
                     'additional_network_attempts': int(action == 'REMOTE_ONCE_PENDING'),
                     'metadata_or_relation_resolved': False,
                     'completed_read_evidence': local[identity] if attempted else None,
                     'recorded_metadata_evidence': recorded.get(identity)})
    return {'status': 'FROZEN_METADATA_ACTION_FRAME_TRANSPORT_IMPLEMENTATION_PENDING',
            'source_sha256': {p: hashlib.sha256(b).hexdigest() for p, b in zip(paths, blobs)},
            'history_rows': rows,
            'M': [r['identity'] for r in rows if r['action'] in ('REMOTE_ONCE_PENDING', 'LOCAL_READ_ONCE_PENDING')],
            'candidate_rows': prefix['rows'], 'counts': dict(Counter(r['action'] for r in rows)),
            'combined_new_transfer_ceiling_bytes': 8 * 1024 ** 3,
            'max_network_attempts_per_entry': 1, 'expand_or_replace_allowed': False,
            'limits': ['Existing metadata remains at recorded evidence level, not automatic relation resolution.',
                       'Failed/unsupported completed local reads have zero new fetch allowance.',
                       'After this bounded execution, settle history/role feasibility without further source search.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = build(root)
    with (root / 'outputs/fable_cdfe68_exposure_v1/metadata_action_frame.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report['counts']))

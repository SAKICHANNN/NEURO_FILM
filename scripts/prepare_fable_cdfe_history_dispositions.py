import hashlib
import json
from pathlib import Path
from collections import Counter


def build(root):
    paths = ['outputs/fable_cdfe68_exposure_v1/history_locators_v2.json',
             'outputs/fable_source_population_bridge_v1/index_audit_v2.json',
             'outputs/fable_cdfe68_exposure_v1/local_history_audit.jsonl']
    blobs = [(root / path).read_bytes() for path in paths]
    locators, index = map(json.loads, blobs[:2])
    audit = {r['identity']: r for r in map(json.loads, blobs[2].splitlines())}
    for path, expected in index['input_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('cached publisher index provenance drift')
    indexed = {'fivek/' + r['source_name']: r for r in index['rows']}
    rows = []
    for row in locators['rows']:
        identity = row['identity']
        urls, local = set(), {}
        for evidence in row['evidence']:
            record = evidence['record']
            for key in ('dng_url', 'source_url', 'download_url'):
                if record.get(key): urls.add(record[key])
            for key, digest in [('raw_path', 'raw_sha256'), ('dng_path', 'dng_sha256')]:
                if record.get(key) and (root / record[key]).is_file():
                    local.setdefault(str(root / record[key]), set()).add(record.get(digest) or record.get('sha256'))
        if identity in indexed:
            urls.add(indexed[identity]['dng_url'])
        old = audit[identity]['originals']
        verified = [o for o in old if o['status'].startswith('HASH_MATCH')]
        rows.append({'identity': identity, 'recorded_original_urls': sorted(urls),
                     'local_originals': [{'path': p, 'expected_hashes': sorted(h for h in hashes if h)} for p, hashes in sorted(local.items())],
                     'previous_verified_paths': [o['path'] for o in verified],
                     'transport_disposition': 'LOCAL_REUSE_NO_REFETCH' if local else 'KNOWN_ORIGINAL_REMOTE_PENDING_M',
                     'metadata_disposition': 'REQUIRES_BOUND_METADATA_DECISION',
                     'new_fetch_permitted': False})
    return {'status': 'HISTORY_DISPOSITION_DRAFT_NOT_FROZEN_M',
            'source_sha256': {p: hashlib.sha256(b).hexdigest() for p, b in zip(paths, blobs)},
            'rows': rows, 'counts': dict(Counter(r['transport_disposition'] for r in rows)),
            'limits': ['Existing local originals must not be refetched to work around unsupported metadata.',
                       'Multiple URLs or hashes require identity reconciliation before any request.',
                       'All430slots retained; transport availability does not establish historical independence.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = build(root)
    with (root / 'outputs/fable_cdfe68_exposure_v1/history_dispositions_draft.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report['counts']))

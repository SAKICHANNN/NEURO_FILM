import hashlib
import json
from pathlib import Path


def build(root):
    sources = ['outputs/fable_source_population_bridge_v1/preview_cohort_v1.json',
               'outputs/fable_cdfe68_exposure_v1/ledger.json',
               'outputs/fable_cdfe68_exposure_v1/history_graph.json']
    blobs = [(root / source).read_bytes() for source in sources]
    cohort, ledger, graph = map(json.loads, blobs)
    excluded = {r['identity'] for r in ledger['rows'] if r['exposure'] in
                {'SUBSTANTIVE_DEVELOPMENT', 'QUARANTINED_OLD_PILOT'}}
    candidates = cohort['candidates']
    if candidates != sorted(candidates, key=lambda r: r['candidate_rank']):
        raise ValueError('historical candidate order changed')
    selected = [r for r in candidates if 'fivek/' + r['source_name'] not in excluded][:256]
    if len(selected) != 256 or len({r['source_name'] for r in selected}) != 256:
        raise ValueError('candidate prefix is not256 unique identities')
    restricted = set(graph['admission_only_candidates_linked_to_history_or_quarantine'])
    rows = [{'identity': 'fivek/' + r['source_name'], 'source': r,
             'already_history_linked': 'fivek/' + r['source_name'] in restricted} for r in selected]
    return {'status': 'PREFIX_FROZEN_NOT_EXECUTABLE_ACQUISITION_MANIFEST',
            'source_sha256': {s: hashlib.sha256(b).hexdigest() for s, b in zip(sources, blobs)},
            'rows': rows, 'later_replacement_allowed': False,
            'known_history_linked_count': sum(r['already_history_linked'] for r in rows),
            'new_transfer_ceiling_bytes': 8 * 1024 ** 3, 'complete_fetch_attempts_per_entry': 1,
            'limits': ['M historical request list not frozen yet; do not execute requests from this artifact.',
                       'Retain known linked entries in prefix; exclusion must not extend prefix.',
                       'No roles, target statistics, rendered quality or comfort selection.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = build(root)
    output = root / 'outputs/fable_cdfe68_exposure_v1/candidate_prefix.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'count': len(report['rows']), 'known_linked': report['known_history_linked_count']}))

import hashlib
import json
from collections import Counter
from pathlib import Path


def build(root: Path) -> dict:
    base = root / 'outputs/fable_source_population_bridge_v1'
    provenance = {}

    def read(name):
        data = (base / name).read_bytes()
        provenance[name] = hashlib.sha256(data).hexdigest()
        if name.endswith('.jsonl'):
            return [json.loads(line) for line in data.splitlines() if line.strip()]
        return json.loads(data)

    known = {'fivek/' + name for name in read('known_history_closure.json')['excluded_source_names']}
    candidates = {'fivek/' + row['source_name']: row for row in read('preview_cohort_v1.json')['candidates']}
    pilot = {row['identity'] for row in read('raw_pilot_manifest_v1.json')['rows']}
    descriptors = read('comparisons/descriptors.jsonl')
    metadata = {row['identity']: row for row in read('raw_history_metadata_clues_v1.json')['history_rows']}
    reviews = read('comparisons/visual_reviews.jsonl')
    additional = read('comparisons/additional_lineage_edges.jsonl')
    history = {row['identity'] for row in descriptors if row['pool'] == 'history'}
    identities = known | history | set(candidates) | pilot
    if known & set(candidates) or not pilot <= set(candidates):
        raise ValueError('Unexpected history/cohort or quarantine identity relationship')
    rows = []
    for identity in sorted(identities):
        exposure = ('QUARANTINED_OLD_PILOT' if identity in pilot else
                    'SUBSTANTIVE_DEVELOPMENT' if identity in known else
                    'UNKNOWN_HISTORY_NOT_FRESH' if identity in history else
                    'ADMISSION_ONLY_NOT_YET_FRESH_CERTIFIED')
        recorded = metadata.get(identity)
        rows.append({
            'identity': identity, 'exposure': exposure,
            'candidate_rank': candidates.get(identity, {}).get('candidate_rank'),
            'recorded_raw_metadata': recorded,
            'recorded_raw_path_exists_now': Path(recorded['path']).is_file() if recorded else None,
            'representations': [d for d in descriptors if d['identity'] == identity],
            'fit_or_evaluation_admitted': False,
        })
    edges = list(additional)
    for review in reviews:
        for comparison in review['comparisons']:
            if comparison['decision'] != 'distinct_on_available_evidence':
                edges.append({'left': review['candidate'], 'right': comparison['identity'],
                              'decision': comparison['decision'], 'origin': 'visual_reviews.jsonl',
                              'recorded_comparison': comparison})
    return {
        'status': 'EXPOSURE_JOIN_ONLY_NO_SOURCE_ADMISSION',
        'source_sha256': provenance,
        'counts': dict(Counter(row['exposure'] for row in rows)),
        'identity_count': len(rows), 'recorded_metadata_count': len(metadata),
        'history_without_recorded_raw_metadata': sorted((known | history) - metadata.keys()),
        'rows': rows, 'recorded_lineage_edges': edges,
        'limits': ['Recorded representation hashes preserved, assets not rehashed by this join.',
                   'Missing metadata is not evidence of no prior exposure.',
                   'Freshness needs complete lineage and technical admission; no roles allocated.',
                   'Missing-history list is diagnostic, not the frozen acquisition list M.',
                   'Camera/day unions and pilot metadata join remain pending.'],
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = build(root)
    output = root / 'outputs/fable_cdfe68_exposure_v1/ledger.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ['status', 'counts', 'identity_count', 'recorded_metadata_count']}))

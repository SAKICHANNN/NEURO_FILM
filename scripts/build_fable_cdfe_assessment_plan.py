import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = {
        'lock': 'outputs/fable_cdfe68_exposure_v1/role_lock.json',
        'inventory': 'outputs/fable_cdfe68_exposure_v1/strict_role_inventory.json',
        'renders': 'outputs/fable_cdfe68_native_candidates_v1/renders.jsonl',
        'treatments': 'outputs/fable_cdfe68_computation_v1/treatments.json',
        'randomness': 'outputs/fable_cdfe68_computation_v1/randomness_controls_panel.json',
        'masks': 'outputs/fable_cdfe68_native_candidates_v1/query_masks_v1/manifest.json'}
    sources = {key: (root / value).read_bytes() for key, value in paths.items()}
    lock = json.loads(sources['lock'])['assignment']
    inventory = {r['identity']: r for r in json.loads(sources['inventory'])['rows']}
    renders = {r['identity']: r for r in map(json.loads, sources['renders'].splitlines())}
    random = json.loads(sources['randomness'])
    treatments = json.loads(sources['treatments'])['partitions']['assessment']
    masks = json.loads(sources['masks'])['rows']
    ids = lock['represented_assessment'] + [i for camera in lock['held_cameras']
                                          for i in lock['held_assessment'][camera]]
    if len(ids) != 32 or len(set(ids)) != 32 or len(treatments) != 32:
        raise ValueError('fixed assessment cardinalities differ')
    rows = []
    for identity in ids:
        native, info = renders[identity], inventory[identity]
        if native['native_sha256'] != info['native_sha256']:
            raise ValueError('native identity differs from admitted inventory')
        rows.append({'identity': identity, 'camera': info['camera'],
            'native_path': native['native_path'], 'native_sha256': native['native_sha256'],
            'dequantization_seed': random['dequantization']['seeds'][identity],
            'stratum': 'represented' if identity in lock['represented_assessment'] else 'held'})
    queries = [{key: renders[i][key] for key in ['identity', 'native_path', 'native_sha256']}
               for i in lock['queries']]
    if set(m['identity'] for m in masks) != set(lock['queries']):
        raise ValueError('query mask identities differ')
    plan = {'status': 'FIXED_ASSESSMENT_PLAN_NO_OUTCOMES_OPENED', 'rows': rows,
        'queries': queries, 'masks': masks, 'treatments': treatments,
        'treatment_ids': [t['id'] for t in treatments],
        'treatment_regions': [','.join('+' if v > 0 else '-' for v in t['u']) for t in treatments],
        'shuffle_mapping': random['shuffled_C']['mapping'], 'comfort': random['comfort'],
        'source_sha256': {paths[k]: hashlib.sha256(v).hexdigest() for k, v in sources.items()}}
    plan['source_sha256']['scripts/build_fable_cdfe_assessment_plan.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if len(set(plan['treatment_regions'])) != 4:
        raise ValueError('four assessment orthants required')
    path = root / 'outputs/fable_cdfe68_computation_v1/assessment_plan_v1.json'
    with path.open('x') as stream:
        json.dump(plan, stream, indent=2)
    print('Fixed assessment32 x treatment32 x query4; no native arrays or model outcomes opened.')


if __name__ == '__main__':
    main()

import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
SPEC = importlib.util.spec_from_file_location('frozen_photo_v2', ROOT / 'scripts/prepare_tst_crossed_photo_review_v2.py')
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)
read, digest, require = v2.read, v2.digest, v2.require
DEFAULT = ROOT / 'configs/tst_crossed_photo_review_v3.json'
SEED = 'tst-crossed-donor-coverage-v3-20260909'
QUOTAS = {'fit': 64, 'monitor': 16, 'constructed_test': 16}
BASE_CURSOR, BASE_AUDIT = v2.cursor, v2.audit_history


def rank(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def select_rows(rows, consumed, quotas, seed):
    require(seed == SEED, 'SELECTION_SEED_CHANGED')
    by_key = {r['operator_key']: r for r in rows}
    require(len(by_key) == len(rows) and set(consumed) <= set(by_key), 'INVALID_POPULATION_OR_CONSUMPTION')
    excluded = {by_key[k]['candidate']['before_component'] for k in consumed}
    groups = {}
    for row in rows:
        component = row['candidate']['before_component']
        if component not in excluded:
            groups.setdefault((row['split'], component), []).append(row)
    selected = []
    for split, count in quotas.items():
        components = sorted([c for s, c in groups if s == split], key=lambda c: (rank([seed, split, c]), c))
        require(len(components) >= count, 'INSUFFICIENT_UNCONSUMED_COMPONENTS:'+split)
        for component in components[:count]:
            selected.append(min(groups[split, component], key=lambda r: (rank([seed, r['operator_key']]), r['operator_key'])))
    require(len({r['candidate']['before_component'] for r in selected}) == len(selected), 'CROSS_ROLE_COMPONENT_COLLISION')
    return selected


def effective_history(lock, terminals, overrides):
    decisions = {k: v['status'] for k, v in lock['imported'].items()}
    for t in terminals:
        require(t['operator_key'] not in decisions, 'DUPLICATE_HISTORY')
        decisions[t['operator_key']] = t['status']
    seen = set()
    for a in overrides:
        key = a['operator_key']
        require(key in decisions and key not in seen, 'DUPLICATE_OR_UNKNOWN_ADJUDICATION')
        require(a['status'] in ('FAIL', 'UNCERTAIN_NOT_ADMITTED') and a['training_admitted'] is False, 'INVALID_OVERRIDE')
        decisions[key] = a['status']
        seen.add(key)
    return decisions


def prefix_audit(config):
    prior = read(ROOT / config['parent_config'])
    for path, sha in prior['pins'].items():
        require(digest(ROOT / path) == sha, 'PARENT_PIN_CHANGED:'+path)
    require(digest(ROOT / config['parent_entry']) == prior['entry_sha256'], 'PARENT_ENTRY_CHANGED')
    output = ROOT / prior['output']
    lock = read(output / 'input_lock.json')
    require(read(output / 'cursor.json')['index'] == 13, 'PARENT_PREFIX_CHANGED')
    require(not (output / 'running.lock').exists(), 'PARENT_RUNNING')
    require(v2.import_reviews(prior, set(lock['population_keys'])) == lock['imported'], 'PARENT_IMPORT_CHANGED')
    base = {k: v for k, v in lock.items() if k != 'identity'}
    require(v2.old.s3.object_hash(base) == lock['identity']['input_sha256'], 'PARENT_INPUT_IDENTITY')
    require(lock['identity']['config_sha256'] == digest(ROOT / config['parent_config']), 'PARENT_CONFIG_IDENTITY')
    checkpoint = read(output / 'accounting_checkpoint.json')
    require(checkpoint['attempts'] == 40 and abs(checkpoint['cpu']-82.828125) < 1e-8 and abs(checkpoint['wall']-81.064) < 1e-6, 'PARENT_ACCOUNTING_CHANGED')
    audit_inputs = {**base, 'queue': base['queue'][:13]}
    BASE_AUDIT(prior, audit_inputs, lock['identity'])
    last = output / 'terminal' / (base['queue'][12]+'.json')
    require(read(output / 'cursor.json')['previous_terminal_sha256'] == digest(last), 'PARENT_CURSOR_HEAD')
    terminals = [read(output / 'terminal' / (k+'.json')) for k in base['queue'][:13]]
    override_paths = sorted((output / 'adjudications').glob('*.json'))
    require([str(p.relative_to(ROOT)).replace('\\', '/') for p in override_paths] == config['adjudications'], 'ADJUDICATION_INVENTORY_CHANGED')
    overrides = []
    for p in override_paths:
        a = read(p)
        require(digest(Path(a['raw_terminal_path'])) == a['raw_terminal_sha256'], 'OVERRIDE_TERMINAL_CHANGED')
        require(digest(Path(a['independent_review_path'])) == a['independent_review_sha256'], 'OVERRIDE_REVIEW_CHANGED')
        require(read(Path(a['raw_terminal_path']))['operator_key'] == a['operator_key'] and read(Path(a['independent_review_path']))['operator_key'] == a['operator_key'], 'OVERRIDE_KEY_MISMATCH')
        overrides.append(a)
    effective = effective_history(lock, terminals, overrides)
    require(len(effective) == 25, 'HISTORY_COUNT')
    return {'status': 'PARTIAL_PREFIX_13_INTEGRITY_VERIFIED_NOT_FULL_POPULATION', 'effective_decisions': effective,
            'effective_counts': dict(Counter(effective.values())), 'checkpoint': checkpoint,
            'historical_training_admitted': False, 'unreviewed_population_not_audited': len(base['queue'])-13}


def frozen_inputs(config):
    receipt = read(ROOT / config['prefix_receipt'])
    require(receipt['status'] == 'PARTIAL_PREFIX_13_INTEGRITY_VERIFIED_NOT_FULL_POPULATION', 'PREFIX_NOT_AUDITED')
    require(config['quotas'] == QUOTAS, 'QUOTAS_CHANGED')
    lock = read(ROOT / config['parent_input'])
    selected = select_rows(lock['rows'], receipt['effective_decisions'], config['quotas'], config['selection_seed'])
    queue = v2.ordered_queue(selected, {})
    return {'population_keys': [r['operator_key'] for r in selected], 'rows': selected, 'imported': {}, 'queue': queue,
            'queue_sha256': v2.old.s3.object_hash(queue), 'stage_order': list(v2.STAGES), 'split_cycle': list(v2.SPLITS),
            'historical_counts_only': receipt['effective_counts'], 'historical_training_admitted': False,
            'endpoint': 'EXACTLY_96_NO_REPLACEMENT_NO_AUTOMATIC_TRANCHE_OR_TRAINING'}


def cursor(config, inputs, identity):
    state, item, probe = BASE_CURSOR(config, inputs, identity)
    state['terminal_count'] -= 12
    state['historical_terminal_count'] = 25
    state['scope'] = 'FINITE_96_NOT_FULL_3456_POPULATION'
    return state, item, probe


def audit_history(config, inputs, identity, active=None):
    prefix_audit(config)
    result = BASE_AUDIT(config, inputs, identity, active)
    result.update(status='FINITE_96_HISTORY_VERIFIED_NOT_TRAINING', full_population_audited=False,
                  historical_terminal_count=25, automatic_next_tranche=False)
    return result


def install():
    v2.__file__ = __file__
    v2.frozen_inputs = frozen_inputs
    v2.cursor = cursor
    v2.audit_history = audit_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--ingest', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    require(sum((args.prepare, args.run, args.audit, args.ingest is not None, args.worker is not None)) <= 1, 'ONE_ACTION_ONLY')
    if args.prepare:
        start_cpu, start_wall = time.process_time(), time.monotonic()
        cfg = read(args.config)
        result = prefix_audit(cfg)
        result.update(preparation_cpu_seconds=time.process_time()-start_cpu, preparation_wall_seconds=time.monotonic()-start_wall)
        v2.create_json(ROOT / cfg['prefix_receipt'], result)
        print(json.dumps(result, indent=2))
    else:
        install()
        if args.worker:
            try:
                v2.worker(args.config, args.worker)
            finally:
                v2.create_json(args.worker / 'worker_final.json', {'worker_cpu_seconds': time.process_time()})
        elif args.run or args.ingest or args.audit:
            raise SystemExit(v2.launch(args.config, args.ingest, args.audit))
        else:
            print(json.dumps(v2.preflight(args.config)[2], indent=2))

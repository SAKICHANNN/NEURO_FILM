import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
ENTRY = ROOT / 'scripts/prepare_tst_crossed_photo_review_v3.py'
CONFIG = ROOT / 'configs/tst_crossed_photo_review_v3.json'
OUTPUT = ROOT / 'outputs/tst_crossed_photo_review_v3'
RECEIPT = OUTPUT / 'recovery/attempt0230_adjudication.json'
ATTEMPT_SHA = 'e79947212b32f65bac0476a460e6fd4a812e34b9c68e125ce0fad37ceb8e97c7'
IDENTITY_SHA = 'eb8066c634d0bad12a9635f0fe1336e61f706d2c81cddf1f6999d0879ac3bc17'
KEY = '61d3db30d324a931e57d75bef32b6120d7f490a1dec7cda3a8594a3165edc4ed'
PINS = {
    'scripts/prepare_tst_crossed_photo_review_v2.py': '3fa51ab01e93d5d53bef8c88611b4ab704242becf3e57cec601e8b9d07f5d60b',
    'scripts/prepare_tst_crossed_photo_review_v3.py': 'ec301a2691fd7085ef35693bf30e2918b807109fbce1d8b1bf4fe34dd96fae20',
    'configs/tst_crossed_photo_review_v3.json': 'fe3051479474ae32d52f11214179db93e4e4ef6054e32b59e90acb47d23e65ad',
}
SPEC = importlib.util.spec_from_file_location('recovery_frozen_v3', ENTRY)
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)
v2 = v3.v2
read, digest, require = v2.read, v2.digest, v2.require
BASE_COSTS = v2.costs


def verify_failed_attempt(directory):
    require(directory.name == '0230', 'RECOVERY_WRONG_ATTEMPT')
    require({p.name for p in directory.iterdir()} == {'identity.json', 'accounting.json', 'failure.json'}, 'RECOVERY_WORKER_OR_UNKNOWN_FILE')
    require(digest(directory / 'accounting.json') == ATTEMPT_SHA, 'RECOVERY_FAILED_ACCOUNTING_CHANGED')
    require(digest(directory / 'failure.json') == ATTEMPT_SHA, 'RECOVERY_FAILURE_CHANGED')
    require(digest(directory / 'identity.json') == IDENTITY_SHA, 'RECOVERY_IDENTITY_CHANGED')
    value = read(directory / 'accounting.json')
    require(value['live'] is False and value['exit_code'] == -1 and value['peak_tree_rss_bytes'] == 0 and value['added_candidate_bytes'] == 0, 'RECOVERY_NOT_PRE_WORKER_FAILURE')
    require(value['failure'] == 'WORKER_EXCEPTION_OR_INTERRUPTION', 'RECOVERY_FAILURE_KIND_CHANGED')
    return value


def accepted_failure(path, value):
    if value['failure'] is None:
        return True
    require(path == OUTPUT / 'attempts/0230/accounting.json', 'UNADJUDICATED_FAILURE')
    require(value == verify_failed_attempt(path.parent), 'RECOVERY_ACCOUNTING_VALUE_CHANGED')
    return True


def verify_accounting(output, checkpoint):
    previous, totals = None, {'cpu': 0., 'wall': 0., 'bytes': 0}
    for n in range(1, checkpoint['attempts'] + 1):
        path = output / 'attempts' / f'{n:04d}' / 'accounting.json'
        value = read(path)
        require(value['previous_accounting_sha256'] == previous, 'ACCOUNTING_CHAIN_FAILURE')
        require(accepted_failure(path, value), 'ACCOUNTING_FAILURE_UNADJUDICATED')
        totals['cpu'] += value['aggregate_cpu_seconds']
        totals['wall'] += value['active_wall_seconds']
        totals['bytes'] += value['added_candidate_bytes']
        previous = digest(path)
    require(previous == checkpoint['last_sha256'], 'ACCOUNTING_HEAD_CHANGED')
    require(all(abs(totals[k] - checkpoint[k]) < 1e-8 for k in totals), 'ACCOUNTING_TOTAL_CHANGED')


def proof():
    for path, sha in PINS.items():
        require(digest(ROOT / path) == sha, 'RECOVERY_FROZEN_PIN_CHANGED:' + path)
    failure = verify_failed_attempt(OUTPUT / 'attempts/0230')
    return {
        'status': 'ADJUDICATED_PRE_WORKER_PATH_VALIDATION_FAILURE_ONLY',
        'attempt': '0230', 'accounting_sha256': ATTEMPT_SHA,
        'identity_sha256': IDENTITY_SHA, 'pins': PINS,
        'adapter_sha256': digest(Path(__file__)),
        'failure_retained': failure,
        'original_failure_cleared': False, 'training_admitted': False,
        'permitted_change': 'Resume unchanged finite96 queue; retain failed cost and raw history; no other failure exception.',
    }


def adjudicate():
    value = proof()
    require(not (OUTPUT / 'running.lock').exists(), 'RECOVERY_RUNNING_LOCK')
    require(not (OUTPUT / 'attempts/0231').exists(), 'RECOVERY_ALREADY_ADVANCED')
    checkpoint = read(OUTPUT / 'accounting_checkpoint.json')
    require(checkpoint['attempts'] == 230 and checkpoint['last_sha256'] == ATTEMPT_SHA, 'RECOVERY_WRONG_HEAD')
    verify_accounting(OUTPUT, checkpoint)
    require(checkpoint['bytes'] == 83490877, 'RECOVERY_PREFIX_BYTES_CHANGED')
    require(sum(p.stat().st_size for p in (OUTPUT / 'candidate').rglob('*') if p.is_file()) == checkpoint['bytes'], 'RECOVERY_CANDIDATE_BYTES_CHANGED')
    require(read(OUTPUT / 'cursor.json') == {'index': 66, 'previous_terminal_sha256': 'f97412d99dce3df40c16ad3caac3ef68f0778a3166ee723821b9e3d3829647f4'}, 'RECOVERY_CURSOR_CHANGED')
    require(not (OUTPUT / 'candidate' / KEY / 'lowkey_environment/review.json').exists(), 'RECOVERY_REVIEW_ALREADY_INGESTED')
    require(not (OUTPUT / 'terminal' / (KEY + '.json')).exists(), 'RECOVERY_TERMINAL_ALREADY_PRESENT')
    v2.create_json(RECEIPT, value)
    return value


def costs(config, active=None):
    if config['output'] != 'outputs/tst_crossed_photo_review_v3':
        return BASE_COSTS(config, active)
    require(read(RECEIPT) == proof(), 'RECOVERY_RECEIPT_CHANGED')
    ledger = v2.old.s3.ledger({**config, 'output': config['output'] + '/unused_ledger_namespace'})
    checkpoint = read(OUTPUT / 'accounting_checkpoint.json')
    require(checkpoint['attempts'] >= 230, 'RECOVERY_CHECKPOINT_REWOUND')
    verify_accounting(OUTPUT, checkpoint)
    next_attempt = OUTPUT / 'attempts' / f"{checkpoint['attempts'] + 1:04d}"
    require(not next_attempt.exists() or next_attempt == active, 'UNACCOUNTED_ATTEMPT_REQUIRES_ADJUDICATION')
    legacy = v2.old.s3.attempts_accounting(ROOT / config['legacy_output'] / 'attempts')
    extra_cpu, extra_wall = config['external_photo_cpu_seconds'], config['external_photo_wall_reserve_seconds']
    ledger.update(stage_cpu=checkpoint['cpu'], stage_wall=checkpoint['wall'],
                  photo_cpu=checkpoint['cpu'] + sum(r['aggregate_cpu_seconds'] for r in legacy) + extra_cpu,
                  photo_wall=checkpoint['wall'] + sum(r['active_wall_seconds'] for r in legacy) + extra_wall,
                  retained_bytes=config['legacy_candidate_bytes'] + checkpoint['bytes'], checkpoint=checkpoint)
    ledger['study_cpu'] += checkpoint['cpu'] + extra_cpu
    ledger['study_wall'] += checkpoint['wall'] + extra_wall
    return ledger


def audit_history(config, inputs, identity, active=None):
    v3.prefix_audit(config)
    checkpoint = costs(config, active)['checkpoint']
    previous, seen = None, set()
    population = {r['operator_key']: r for r in inputs['rows']}
    metadata = read(ROOT / config['probe_manifest'])['rows']
    for index, key in enumerate(inputs['queue']):
        path = OUTPUT / 'terminal' / (key + '.json')
        require(path.exists(), 'POPULATION_REVIEW_INCOMPLETE')
        terminal = read(path)
        require(terminal['identity'] == identity and terminal['operator_key'] == key and terminal['next_index'] == index + 1 and terminal['previous_terminal_sha256'] == previous, 'TERMINAL_CHAIN_FAILURE')
        reviews = []
        for receipt in terminal['receipts']:
            stage = receipt['stage']
            directory = OUTPUT / 'candidate' / key / stage
            pid = None if stage == 'donor' else next(r['component'] for r in metadata if r['split'] == population[key]['split'] and r['category'] == stage)
            record, rh = v2.verify_record(directory, identity, population[key], stage, pid)
            require(rh == receipt['record_sha256'] and digest(directory / 'review.json') == receipt['review_sha256'], 'HISTORICAL_REVIEW_CHANGED')
            reviews.append(v2.validate_observation(read(directory / 'review.json'), record, rh))
            seen.add(str(directory))
        require(v2.outcome(reviews) == terminal['status'] and terminal['unreviewed_stages'] == list(v2.STAGES[len(reviews):]), 'TERMINAL_OUTCOME_CHANGED')
        previous = digest(path)
    require({str(p.parent) for p in (OUTPUT / 'candidate').rglob('record.json')} == seen, 'UNEXPECTED_RENDERED_STAGE')
    require(sum(p.stat().st_size for p in (OUTPUT / 'candidate').rglob('*') if p.is_file()) == checkpoint['bytes'], 'RETAINED_STORAGE_CHANGED')
    return {'status': 'FINITE_96_HISTORY_VERIFIED_WITH_ADJUDICATED_PRE_WORKER_FAILURE_NOT_TRAINING',
            'population': len(inputs['population_keys']), 'imported': len(inputs['imported']),
            'new_terminal': len(inputs['queue']), 'training_admitted': False,
            'full_population_audited': False, 'historical_terminal_count': 25,
            'automatic_next_tranche': False, 'accepted_failed_attempt': '0230',
            'accepted_failed_accounting_sha256': ATTEMPT_SHA,
            'adjudication_sha256': digest(RECEIPT), 'failure_cost_included': True}


def route_worker(argv, **kwargs):
    require(len(argv) == 6 and Path(argv[1]) == ENTRY and argv[2] == '--config' and Path(argv[3]) == CONFIG and argv[4] == '--worker', 'RECOVERY_UNEXPECTED_WORKER_COMMAND')
    return subprocess.Popen([argv[0], str(Path(__file__)), *argv[2:]], **kwargs)


def install():
    require(read(RECEIPT) == proof(), 'RECOVERY_RECEIPT_CHANGED')
    v3.install()
    v2.costs = costs
    v2.audit_history = audit_history
    v2.subprocess = SimpleNamespace(Popen=route_worker, STDOUT=subprocess.STDOUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--adjudicate', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--audit', action='store_true')
    action.add_argument('--ingest', type=Path)
    action.add_argument('--worker', type=Path)
    args = parser.parse_args()
    require(args.config == CONFIG, 'RECOVERY_CONFIG_OVERRIDE_FORBIDDEN')
    if args.adjudicate:
        print(json.dumps(adjudicate(), indent=2))
    else:
        install()
        if args.worker:
            try:
                v2.worker(CONFIG, args.worker)
            finally:
                v2.create_json(args.worker / 'worker_final.json', {'worker_cpu_seconds': time.process_time()})
        elif args.run or args.audit or args.ingest:
            if args.ingest:
                require(args.ingest.is_absolute() and args.ingest.is_relative_to(ROOT), 'REVIEW_MUST_BE_LOCAL_CHECKOUT_FILE')
            raise SystemExit(v2.launch(CONFIG, args.ingest, args.audit))
        else:
            print(json.dumps(v2.preflight(CONFIG)[2], indent=2))

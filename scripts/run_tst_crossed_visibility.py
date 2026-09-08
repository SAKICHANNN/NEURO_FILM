import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from src.eval import tst_operator_visibility as core

spec = importlib.util.spec_from_file_location('visibility_frozen_s2', ROOT / 'scripts/run_tst_crossed_operators.py')
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)
read, save, digest, require = s2.read, s2.save, s2.digest, s2.require
DEFAULT = ROOT / 'configs/tst_crossed_visibility_v1.json'
SOLVED = 'PROVISIONAL_NUMERICAL_SOLVE_ONLY'
S2_STATUSES = {SOLVED, 'NUMERICAL_FAILURE_NOT_ADMITTED', 'DIMENSION_MISMATCH_NO_PIXEL_PAIRING', 'ITEM_TECHNICAL_FAILURE'}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def probe_metadata(config):
    manifest = read(ROOT / config['probe_manifest'])
    require(manifest['status'] == 'FROZEN_12_SOURCE_ADMISSION_PROBES_ROOT_ACCEPTED', 'PROBES_NOT_FROZEN')
    rows = manifest['rows']
    require(len(rows) == 12 and len({r['component'] for r in rows}) == 12, 'PROBE_COUNT_OR_COMPONENT_REUSE')
    require(Counter(r['split'] for r in rows) == {'fit': 4, 'monitor': 4, 'constructed_test': 4}, 'PARTITION_PROBE_COUNT')
    files = {r['path']: r for r in read(ROOT / config['acquisition_config'])['files']}
    source_identity = read(ROOT / config['source_report'])['identity']
    result = []
    for row in rows:
        name = row['representative']
        token = hashlib.sha256(name.encode()).hexdigest()
        header_path = ROOT / config['headers'] / (token + '.json')
        source_path = ROOT / config['sources'] / (token + '.json')
        require(digest(header_path) == row['header_record_sha256'], 'PROBE_HEADER_CHANGED')
        header, source = read(header_path), read(source_path)
        require(header['identity'] == source_identity and source['identity'] == source_identity, 'PROBE_SOURCE_IDENTITY_CHANGED')
        require(source['status'] == 'SOURCE_DECODED' and source['split'] == row['split'], 'PROBE_SOURCE_STATE')
        require(source['canonical_pixel_sha256'] == row['source_canonical_srgb16_sha256'], 'PROBE_CANONICAL_METADATA_CHANGED')
        require(source['shape'] == row['shape'] and source['color_policy'] == row['color_policy'], 'PROBE_SHAPE_OR_COLOR_POLICY_CHANGED')
        require(source['encoded_sha256'] == header['encoded_sha256'] == files[name]['sha256'] == row['source_encoded_sha256'], 'PROBE_ENCODED_METADATA_CHANGED')
        result.append({'probe_id': row['component'], 'split': row['split'], 'source': name,
                       'shape': row['shape'], 'canonical_sha256': row['source_canonical_srgb16_sha256'],
                       'encoded': files[name], 'header_path': str(header_path), 'header_sha256': digest(header_path),
                       'source_record_sha256': digest(source_path), 'color_policy': row['color_policy']})
    for split in ('fit', 'monitor', 'constructed_test'):
        cache = sum(int(np.prod(r['shape'])) * 2 for r in result if r['split'] == split)
        require(cache <= config['probe_cache_bytes'], 'PROBE_CACHE_CAPACITY')
    return result


def completed_s2(config):
    directory = ROOT / config['s2_output']
    report_path = directory / 'report.json'
    if (directory / 'running.lock').exists() or not report_path.exists():
        return None
    report = read(report_path)
    if report.get('status') != 'COMPLETE_PROVISIONAL_NOT_ADMISSION':
        return None
    attempts = sorted((directory / 'attempts').glob('*'))
    require(bool(attempts), 'S2_ACCOUNTING_MISSING')
    accounting = read(attempts[-1] / 'accounting.json')
    require(not accounting.get('live') and accounting.get('failure') is None and accounting['exit_code'] == 0, 'S2_LAST_ATTEMPT_NOT_COMPLETE')
    require(report['last_attempt_accounting'] == accounting, 'S2_COMPLETION_ACCOUNTING_CHANGED')
    queue_path = directory / 'queue_lock.json'
    queue = read(queue_path)
    identity = queue['identity']
    require(identity['config_sha256'] == digest(ROOT / config['s2_config']), 'S2_CONFIG_IDENTITY')
    require(identity['entry_sha256'] == digest(ROOT / config['s2_entry']), 'S2_ENTRY_IDENTITY')
    require(identity['queue_sha256'] == object_hash({k: v for k, v in queue.items() if k != 'identity'}), 'S2_QUEUE_CONTENT_CHANGED')
    require(report['identity'] == identity and report['processed'] == len(queue['rows']) == config['expected_candidates'], 'S2_INCOMPLETE_CENSUS')
    require(len({r['operator_key'] for r in queue['rows']}) == len(queue['rows']), 'S2_DUPLICATE_CANDIDATE')
    rows, counts, split_counts = [], Counter(), {}
    for candidate in queue['rows']:
        path = directory / 'candidate' / candidate['operator_key']
        record = s2.reusable(path, identity, candidate)
        require(record['status'] in S2_STATUSES, 'INVALID_S2_TERMINAL_STATUS')
        counts[record['status']] += 1
        split_counts.setdefault(candidate['split'], Counter())[record['status']] += 1
        item = {'operator_key': candidate['operator_key'], 'split': candidate['split'], 's2_status': record['status'],
                'record_path': str(path / 'record.json'), 'record_sha256': digest(path / 'record.json'),
                'candidate_sha256': object_hash(candidate), 'grid': None}
        if record['status'] == SOLVED:
            require(record['solver']['solved'] and record['training_admitted'] is False, 'SOLVED_RECORD_CONTRACT')
            require(record['canonical_before'] == candidate['before_canonical_sha256'], 'S2_CANONICAL_P_IDENTITY')
            grids = [a for a in record['artifacts'] if a['path'] == 'absolute_grid.npy']
            require(len(grids) == 1, 'S2_GRID_MISSING_OR_REPEATED')
            item['grid'] = {**grids[0], 'path': str(path / 'absolute_grid.npy')}
        rows.append(item)
    require(dict(counts) == report['counts'] and split_counts == report['counts_by_split'], 'S2_REPORT_COUNTS_DISAGREE')
    return {'s2_report_sha256': digest(report_path), 's2_queue_sha256': digest(queue_path),
            's2_identity': identity, 'rows': rows, 's2_counts': dict(counts)}


def attempts_accounting(directory, active_attempt=None):
    rows = []
    for attempt in sorted(directory.glob('*')):
        require(attempt.is_dir() and (attempt / 'accounting.json').exists(), 'UNACCOUNTED_ATTEMPT')
        record = read(attempt / 'accounting.json')
        require(not record.get('live') or attempt == active_attempt, 'UNACCOUNTED_LIVE_ATTEMPT')
        require(record['aggregate_cpu_seconds'] >= 0 and record['active_wall_seconds'] >= 0, 'INVALID_ACCOUNTING')
        rows.append({'path': str(attempt / 'accounting.json'), 'sha256': digest(attempt / 'accounting.json'), **record})
    return rows


def ledger(config, active_attempt=None):
    prior = []
    for directory in config['prior_attempt_directories']:
        prior.extend(attempts_accounting(ROOT / directory))
    current = attempts_accounting(ROOT / config['output'] / 'attempts', active_attempt)
    download = read(ROOT / config['source_report'])['identity']['download_accounting']
    cpu = sum(r['aggregate_cpu_seconds'] for r in current)
    wall = sum(r['active_wall_seconds'] for r in current)
    return {'stage_cpu': cpu, 'stage_wall': wall,
            'study_cpu': config['prior_reserve_cpu_seconds'] + download['cpu_seconds'] + sum(r['aggregate_cpu_seconds'] for r in prior) + cpu,
            'study_wall': config['prior_reserve_wall_seconds'] + download['wall_seconds'] + sum(r['active_wall_seconds'] for r in prior) + wall,
            'prior_attempt_pins': {r['path']: r['sha256'] for r in prior}}


def preflight(config_path, active_attempt=None):
    config = read(config_path)
    require(digest(Path(__file__)) == config['entry_sha256'], 'ENTRY_CHANGED')
    for path, expected in config['pins'].items():
        require(digest(ROOT / path) == expected, 'PIN_CHANGED:' + path)
    require(config['strength'] == 1.0 and config['chunk_pixels'] == core.CHUNK_PIXELS and config['threshold'] == core.VISIBILITY_THRESHOLD, 'VISIBILITY_CONTRACT_CHANGED')
    require(config['bound_tol'] == read(ROOT / config['s2_config'])['solver']['bound_tol'], 'FROZEN_SOLVER_BOUND_TOL_CHANGED')
    probes = probe_metadata(config)
    receipt = completed_s2(config)
    if receipt is None:
        return config, None, {'status': 'WAIT_S2_COMPLETE', 'photo_reads': 0, 'partial_S2_records_read': 0}
    costs = ledger(config, active_attempt)
    inputs = {**receipt, 'probes': probes, 'prior_attempt_pins': costs['prior_attempt_pins']}
    identity = {'config_sha256': digest(config_path), 'entry_sha256': config['entry_sha256'], 'input_sha256': object_hash(inputs)}
    output = ROOT / config['output']
    if (output / 'input_lock.json').exists():
        require(read(output / 'input_lock.json') == {'identity': identity, **inputs}, 'INPUT_LOCK_CHANGED')
    for attempt in (output / 'attempts').glob('*'):
        require(read(attempt / 'identity.json') == identity, 'ATTEMPT_IDENTITY_CHANGED')
    require(costs['stage_cpu'] < config['cpu_seconds'] and costs['stage_wall'] < config['wall_seconds'], 'S3_BUDGET_EXHAUSTED')
    require(costs['study_cpu'] + config['cpu_seconds'] - costs['stage_cpu'] <= config['study_cpu_seconds'], 'CUMULATIVE_STUDY_CPU_INSUFFICIENT')
    require(costs['study_wall'] + config['wall_seconds'] - costs['stage_wall'] <= config['study_wall_seconds'], 'CUMULATIVE_STUDY_WALL_INSUFFICIENT')
    return config, inputs, {'status': 'READY', 'photo_reads': 0, 'identity': identity, 'accounting': costs,
                            'candidate_count': len(receipt['rows']), 'solved_count': sum(r['s2_status'] == SOLVED for r in receipt['rows'])}


def load_probes(config, rows):
    result = []
    for row in rows:
        path = s2.corpus.confined(ROOT / config['originals'], row['source'])
        s2.verify_encoded(path, row['encoded'])
        header = s2.verified_header(Path(row['header_path']), row['header_sha256'])
        values, policy = s2.corpus.decode(path, header)
        require(list(values.shape) == row['shape'], 'PROBE_NATIVE_SHAPE_CHANGED')
        require(s2.corpus.canonical_hash(values) == row['canonical_sha256'], 'PROBE_CANONICAL_CHANGED')
        require(policy == row['color_policy'], 'PROBE_COLOR_POLICY_CHANGED')
        require(values.flags.c_contiguous, 'PROBE_NOT_NATIVE_CONTIGUOUS')
        result.append(core.VisibilityProbe(row['probe_id'], row['split'], values))
    return result


def reusable(path, identity, item):
    record = read(path)
    require(record['identity'] == identity and record['input'] == item, 'RESUME_INPUT_CHANGED')
    require(record['status'] != 'FATAL_IDENTITY_FAILURE', 'PRIOR_FATAL_FAILURE')
    require(record['training_admitted'] is False, 'INVALID_ADMISSION_CLAIM')
    return record


def measure_candidate(item, probes, *, bound_tol):
    require(digest(Path(item['record_path'])) == item['record_sha256'], 'S2_RECORD_CHANGED_AFTER_LOCK')
    grid = item['grid']
    require(digest(Path(grid['path'])) == grid['sha256'], 'S2_GRID_CHANGED_AFTER_LOCK')
    values = np.load(grid['path'], allow_pickle=False)
    require(values.dtype == np.float64 and values.shape == (343, 3) and np.isfinite(values).all(), 'S2_GRID_ARRAY_CONTRACT')
    require(np.min(values) >= -bound_tol and np.max(values) <= 1 + bound_tol, 'S2_GRID_NOT_BOUNDED_ABSOLUTE')
    return core.screen_visibility(probes, values, candidate_split=item['split'])


def worker(config_path, attempt):
    config, inputs, checks = preflight(config_path, attempt)
    require(checks['status'] == 'READY' and checks['identity'] == read(attempt / 'identity.json'), 'WORKER_BINDING_CHANGED')
    threadpool_limits(config['threads'])
    output = ROOT / config['output']
    counts, split_counts, processed = Counter(), {}, 0
    started, cpu_started = time.monotonic(), time.process_time()
    probes = []
    for split in ('fit', 'monitor', 'constructed_test'):
        probes = []
        rows = [r for r in inputs['rows'] if r['split'] == split]
        probe_rows = [r for r in inputs['probes'] if r['split'] == split]
        for item in rows:
            directory = output / 'candidate' / item['operator_key']
            path = directory / 'record.json'
            if path.exists():
                record = reusable(path, checks['identity'], item)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                s2.begin_candidate(directory, checks['identity'], attempt.name)
                record = {'identity': checks['identity'], 'input': item, 'attempt': attempt.name,
                          'training_admitted': False, 'photographic_admission': 'NOT_REVIEWED', 'human_preference': 'UNMEASURED'}
                start, cpu = time.monotonic(), time.process_time()
                fatal = None
                try:
                    if item['s2_status'] != SOLVED:
                        record.update({'status': 'NOT_ELIGIBLE_S2_OUTCOME_RETAINED', 's2_status': item['s2_status'], 'probes': []})
                    else:
                        if not probes:
                            try:
                                probes = load_probes(config, probe_rows)
                            except Exception as exc:
                                raise s2.IdentityFailure('FROZEN_PROBE_UNAVAILABLE:' + str(exc)) from exc
                        record.update(measure_candidate(item, probes, bound_tol=config['bound_tol']))
                except Exception as exc:
                    fatal = exc if isinstance(exc, (s2.IdentityFailure, OSError)) else None
                    record.update({'status': 'FATAL_IDENTITY_FAILURE' if fatal else 'VISIBILITY_TECHNICAL_FAILURE',
                                   'failure': {'type': type(exc).__name__, 'message': str(exc)}})
                record.update({'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - start})
                save(path, record)
                if fatal:
                    raise fatal
            counts[record['status']] += 1
            split_counts.setdefault(split, Counter())[record['status']] += 1
            processed += 1
            save(attempt / 'progress.json', {'processed': processed, 'expected': len(inputs['rows']), 'counts': dict(counts), 'worker_cpu_seconds': time.process_time()})
            if processed % 128 == 0:
                save(attempt / f'projection_{processed:04d}.json', {'processed': processed, 'remaining': len(inputs['rows']) - processed,
                     'elapsed_cpu_seconds': time.process_time() - cpu_started, 'elapsed_wall_seconds': time.monotonic() - started,
                     'estimated_remaining_cpu_seconds': (time.process_time() - cpu_started) / processed * (len(inputs['rows']) - processed),
                     'claim': 'Projection includes resume and numeric exclusions; no coverage or threshold changes.'})
    save(attempt / 'worker_report.json', {'status': 'COMPLETE_VISIBILITY_SCREEN_NOT_ADMISSION', 'identity': checks['identity'],
         'processed': processed, 'counts': dict(counts), 'counts_by_split': split_counts, 'worker_cpu_seconds': time.process_time(),
         'claim': 'Full-native necessary visibility screen only. Pair correctness, comfort and content preservation remain unreviewed; no learning admission.'})


def launch(config_path):
    config, inputs, checks = preflight(config_path)
    require(checks['status'] == 'READY', 'S3_NOT_READY')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'running.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    attempt, process, registry = None, None, {}
    started, parent_cpu = time.monotonic(), time.process_time()
    cpu, peak, reason, code = 0., 0, None, -1
    try:
        save(output / 'input_lock.json', {'identity': checks['identity'], **inputs})
        attempt = output / 'attempts' / f'{len(list((output / "attempts").glob("*"))) + 1:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        save(attempt / 'identity.json', checks['identity'])
        save(attempt / 'accounting.json', {'aggregate_cpu_seconds': 0., 'active_wall_seconds': 0., 'live': True})
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--config', str(config_path), '--worker', str(attempt)], stdout=log, stderr=subprocess.STDOUT)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                cpu, rss = s2.corpus.sample_tree(owned, registry)
                peak = max(peak, rss + psutil.Process().memory_info().rss)
                elapsed = time.monotonic() - started
                aggregate = cpu + time.process_time() - parent_cpu
                costs = checks['accounting']
                if costs['stage_cpu'] + aggregate >= config['cpu_seconds'] or costs['study_cpu'] + aggregate >= config['study_cpu_seconds']:
                    reason = 'CPU_LIMIT'
                elif costs['stage_wall'] + elapsed >= config['wall_seconds'] or costs['study_wall'] + elapsed >= config['study_wall_seconds']:
                    reason = 'WALL_LIMIT'
                elif peak > config['rss_bytes']:
                    reason = 'RSS_LIMIT'
                if reason:
                    s2.corpus.stop_tree(registry)
                    break
                time.sleep(.1)
            code = process.wait()
            final_cpu, final_rss = s2.corpus.sample_tree(owned, registry)
            cpu, peak = max(cpu, final_cpu), max(peak, final_rss + psutil.Process().memory_info().rss)
    finally:
        s2.corpus.stop_tree(registry)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if attempt is not None:
            for filename in ('progress.json', 'worker_report.json'):
                path = attempt / filename
                if path.exists():
                    cpu = max(cpu, read(path)['worker_cpu_seconds'])
            accounting = {'aggregate_cpu_seconds': cpu + time.process_time() - parent_cpu + .2,
                          'active_wall_seconds': time.monotonic() - started, 'peak_tree_rss_bytes': peak, 'live': False,
                          'exit_code': code, 'failure': reason if reason else (None if code == 0 else 'WORKER_EXCEPTION_OR_INTERRUPTION'),
                          'owned_processes': [{'pid': pid, 'create_time': created, 'cpu_seconds': row['cpu_seconds'], 'peak_rss_bytes': row['peak_rss_bytes']} for (pid, created), row in registry.items()]}
            save(attempt / 'accounting.json', accounting)
            if code == 0 and reason is None and (attempt / 'worker_report.json').exists():
                save(output / 'report.json', {**read(attempt / 'worker_report.json'), 'last_attempt_accounting': accounting})
            else:
                save(attempt / 'failure.json', accounting)
        lock.unlink()
    return int(code != 0 or reason is not None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.config, args.worker)
        finally:
            path = args.worker / 'progress.json'
            save(path, {**(read(path) if path.exists() else {}), 'worker_cpu_seconds': time.process_time()})
    elif args.run:
        raise SystemExit(launch(args.config))
    else:
        print(json.dumps(preflight(args.config)[2], indent=2))

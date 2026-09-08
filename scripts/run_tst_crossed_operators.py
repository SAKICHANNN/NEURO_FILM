import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits
from src.eval import tst_correspondence_operator as core
from src.eval.tst_reference_response import scoring_folds


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


corpus = module('crossed_frozen_corpus', 'scripts/prepare_tst_crossed_corpus.py')
geometry = module('crossed_frozen_geometry', 'scripts/prepare_tst_reference_pairs.py').geometry
read, save, digest = corpus.read, corpus.save, corpus.digest
DEFAULT = ROOT / 'configs/tst_crossed_operators_v1.json'


class IdentityFailure(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise IdentityFailure(reason)


def verify_encoded(path, row):
    try:
        corpus.verify_input(path, row)
    except Exception as exc:
        raise IdentityFailure('ENCODED_INPUT_CHANGED:' + str(path)) from exc


def verified_header(path, expected):
    try:
        require(digest(path) == expected, 'HEADER_CHANGED_AFTER_LOCK')
        return read(path)['header']
    except Exception as exc:
        raise IdentityFailure('HEADER_IDENTITY_UNAVAILABLE:' + str(path)) from exc


def freeze_queue(candidates, probes, components):
    require(probes['status'] == 'FROZEN_12_SOURCE_ADMISSION_PROBES_ROOT_ACCEPTED', 'PROBES_NOT_FROZEN')
    require(len(probes['rows']) == 12, 'PROBE_COUNT')
    reserved = {r['component'] for r in probes['rows']}
    require(len(reserved) == 12, 'PROBE_COMPONENT_REUSE')
    for row in probes['rows']:
        require(components['owner_to_component'][row['representative']] == row['component'], 'STALE_PROBE_COMPONENT')
        actual = {p for p, c in components['owner_to_component'].items() if c == row['component']}
        require(actual == set(row['all_component_aliases']), 'INCOMPLETE_PROBE_ALIAS_RESERVATION')
    quarantine = set(components['quarantined_owners'])
    queue, excluded = [], []
    for row in sorted(candidates, key=lambda r: (r['split'], r['before'], r['after'])):
        require(row['before_component'] == components['owner_to_component'][row['before']], 'STALE_CANDIDATE_COMPONENT')
        require(row['before'] not in quarantine, 'QUARANTINED_CANDIDATE')
        target = excluded if row['before_component'] in reserved else queue
        target.append({**row, 'probe_alias_reservation': 'EXCLUDED' if target is excluded else 'CLEARED'})
    require(len({r['operator_key'] for r in queue}) == len(queue), 'DUPLICATE_OPERATOR_KEY')
    return {'rows': queue, 'excluded_probe_donors': excluded, 'reserved_components': sorted(reserved)}


def ledger(config, output, active_attempt=None):
    prior = []
    for pattern in config['prior_accounting_globs']:
        prior.extend(read(p) for p in ROOT.glob(pattern))
    attempts = sorted((output / 'attempts').glob('*'))
    for attempt in attempts:
        require((attempt / 'accounting.json').exists(), 'UNACCOUNTED_ATTEMPT_REQUIRES_COST_RECONCILIATION')
    current = [read(p / 'accounting.json') for p in attempts]
    for path, record in zip(attempts, current):
        require(not record.get('live') or path == active_attempt, 'UNACCOUNTED_LIVE_ATTEMPT_REQUIRES_RECONCILIATION')
    stage_cpu = sum(r['aggregate_cpu_seconds'] for r in current)
    stage_wall = sum(r['active_wall_seconds'] for r in current)
    download = read(ROOT / config['source_report'])['identity']['download_accounting']
    return {'stage_cpu': stage_cpu, 'stage_wall': stage_wall,
            'study_cpu': config['prior_reserve_cpu_seconds'] + download['cpu_seconds'] + sum(r['aggregate_cpu_seconds'] for r in prior) + stage_cpu,
            'study_wall': config['prior_reserve_wall_seconds'] + download['wall_seconds'] + sum(r['active_wall_seconds'] for r in prior) + stage_wall}


def preflight(config_path, active_attempt=None):
    config = read(config_path)
    require(digest(Path(__file__)) == config['entry_sha256'], 'ENTRY_CHANGED')
    for path, expected in config['pins'].items():
        require(digest(ROOT / path) == expected, 'PIN_CHANGED:' + path)
    old = read(ROOT / config['solver_config'])
    for key in ('dimension', 'smoothness', 'tie', 'seed', 'pixels_per_fold', 'solver', 'threads'):
        require(config[key] == old[key], 'FROZEN_SOLVER_CHANGED:' + key)
    probes = read(ROOT / config['probe_manifest'])
    queue = freeze_queue(read(ROOT / config['candidates']), probes, read(ROOT / config['components']))
    files = {r['path']: r for r in read(ROOT / config['acquisition_config'])['files']}
    source_report = read(ROOT / config['source_report'])
    require(source_report['counts'] == {'SOURCE_DECODED': 4264}, 'INCOMPLETE_S1')
    require(source_report['identity']['receipt_sha256'] == digest(ROOT / config['receipt']), 'RECEIPT_CHANGED')
    for row in queue['rows']:
        for role in ('before', 'after'):
            require(files[row[role]]['sha256'] == row[role + '_encoded_sha256'], 'INPUT_MANIFEST_CHANGED')
            token = hashlib.sha256(row[role].encode()).hexdigest()
            path = ROOT / config['headers'] / (token + '.json')
            record = read(path)
            require(record['identity'] == source_report['identity'] and record['encoded_sha256'] == files[row[role]]['sha256'], 'HEADER_IDENTITY_CHANGED')
            require(record['header']['eligible_decode'], 'INELIGIBLE_HEADER')
            row[role + '_header_sha256'] = digest(path)
        source_path = ROOT / config['sources'] / (hashlib.sha256(row['before'].encode()).hexdigest() + '.json')
        source = read(source_path)
        require(source['identity'] == source_report['identity'] and source['status'] == 'SOURCE_DECODED', 'SOURCE_IDENTITY_CHANGED')
        row['before_canonical_sha256'] = source['canonical_pixel_sha256']
        row['source_record_sha256'] = digest(source_path)
    identity = {'config_sha256': digest(config_path), 'entry_sha256': config['entry_sha256'],
                'queue_sha256': hashlib.sha256(json.dumps(queue, sort_keys=True).encode()).hexdigest()}
    output = ROOT / config['output']
    if (output / 'queue_lock.json').exists():
        require(read(output / 'queue_lock.json') == {'identity': identity, **queue}, 'QUEUE_LOCK_CHANGED')
    costs = ledger(config, output, active_attempt)
    require(costs['stage_cpu'] < config['cpu_seconds'] and costs['stage_wall'] < config['wall_seconds'], 'STAGE_BUDGET_EXHAUSTED')
    require(costs['study_cpu'] + config['cpu_seconds'] - costs['stage_cpu'] <= config['study_cpu_seconds'], 'STUDY_BUDGET_INSUFFICIENT')
    return config, queue, {'status': 'READY', 'photo_reads': 0, 'identity': identity, 'accounting': costs,
                           'candidates': len(queue['rows']), 'excluded_probe_donors': len(queue['excluded_probe_donors'])}


def reusable(directory, identity, row):
    record = read(directory / 'record.json')
    require(record['identity'] == identity and record['candidate'] == row, 'RESUME_IDENTITY_CHANGED')
    require(record['status'] != 'FATAL_IDENTITY_FAILURE', 'PRIOR_FATAL_IDENTITY_FAILURE_REQUIRES_ADJUDICATION')
    for artifact in record['artifacts']:
        require(digest(directory / artifact['path']) == artifact['sha256'], 'RESUME_ARTIFACT_CHANGED')
    return record


def begin_candidate(directory, identity, attempt_name):
    starts = sorted(directory.glob('started_*.json'))
    require(len(starts) < 2, 'INTERRUPTED_CANDIDATE_REPLAY_LIMIT')
    for path in starts:
        require(read(path)['identity'] == identity, 'INTERRUPTED_CANDIDATE_IDENTITY_CHANGED')
    if starts:
        archive = directory / ('interrupted_' + read(starts[0])['attempt'])
        archive.mkdir(exist_ok=False)
        for path in directory.iterdir():
            if path.is_file() and not path.name.startswith('started_'):
                path.rename(archive / path.name)
    save(directory / ('started_' + attempt_name + '.json'), {'identity': identity, 'attempt': attempt_name, 'prior_starts': len(starts)})


def fit_arrays(before, after, canonical, config, directory):
    record = {'geometry': geometry(before, after), 'artifacts': []}
    if before.shape != after.shape:
        return {**record, 'status': 'DIMENSION_MISMATCH_NO_PIXEL_PAIRING'}
    folds = scoring_folds(*before.shape[:2], canonical, config['seed'], config['pixels_per_fold'])
    x = before.reshape(-1, 3)
    y = after.reshape(-1, 3)
    fit_x, fit_y = x[folds[0]].astype(float) / 65535, y[folds[0]].astype(float) / 65535
    values, state = core.fit_operator(fit_x, fit_y, config)
    check_x, check_y = x[folds[1]].astype(float) / 65535, y[folds[1]].astype(float) / 65535
    prediction = core.render_absolute(check_x, values, config['dimension'])
    error = deltaE_ciede2000(rgb2lab(prediction), rgb2lab(check_y))
    identity_error = deltaE_ciede2000(rgb2lab(check_x), rgb2lab(check_y))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / 'absolute_grid.npy', values, allow_pickle=False)
    np.savez_compressed(directory / 'coordinates.npz', fit_flat_indices=folds[0], score_flat_indices=folds[1], shape=np.array(before.shape))
    for filename in ('absolute_grid.npy', 'coordinates.npz'):
        path = directory / filename
        record['artifacts'].append({'path': filename, 'sha256': digest(path), 'bytes': path.stat().st_size})
    record.update({'status': 'PROVISIONAL_NUMERICAL_SOLVE_ONLY' if state['solved'] else 'NUMERICAL_FAILURE_NOT_ADMITTED',
                   'solver': state, 'support': core.support_summary(fit_x, check_x, config['dimension']),
                   'score': {'mean_deltaE2000': float(error.mean()), 'p95_deltaE2000': float(np.quantile(error, .95)),
                             'identity_mean_deltaE2000': float(identity_error.mean()), 'pixels': len(folds[1])},
                   'training_admitted': False, 'photographic_admission': 'NOT_REVIEWED', 'human_preference': 'UNMEASURED'})
    return record


def worker(config_path, attempt):
    config, queue, checks = preflight(config_path, attempt)
    require(read(attempt / 'identity.json') == checks['identity'], 'WORKER_IDENTITY_CHANGED')
    threadpool_limits(config['threads'])
    output = ROOT / config['output']
    files = {r['path']: r for r in read(ROOT / config['acquisition_config'])['files']}
    counts, previous_before, before, policy = {}, None, None, None
    split_counts = {}
    work_started, work_cpu = time.monotonic(), time.process_time()
    cache_bytes = sum(p.stat().st_size for p in (output / 'candidate').rglob('*') if p.is_file())
    for index, row in enumerate(queue['rows']):
        directory = output / 'candidate' / row['operator_key']
        if (directory / 'record.json').exists():
            result = reusable(directory, checks['identity'], row)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            begin_candidate(directory, checks['identity'], attempt.name)
            started, cpu = time.monotonic(), time.process_time()
            result = {'artifacts': []}
            fatal = None
            arrays, after, values = [], None, None
            try:
                policies = []
                for role in ('before', 'after'):
                    path = corpus.confined(ROOT / config['originals'], row[role])
                    verify_encoded(path, files[row[role]])
                    token = hashlib.sha256(row[role].encode()).hexdigest()
                    header_path = ROOT / config['headers'] / (token + '.json')
                    header = verified_header(header_path, row[role + '_header_sha256'])
                    if role == 'before' and previous_before == row['before']:
                        values, color = before, policy
                    else:
                        values, color = corpus.decode(path, header)
                    arrays.append(values)
                    policies.append(color)
                before, after = arrays
                policy, previous_before = policies[0], row['before']
                canonical = corpus.canonical_hash(before)
                require(canonical == row['before_canonical_sha256'], 'CANONICAL_P_CHANGED')
                result = fit_arrays(before, after, canonical, config, directory)
                result.update({'canonical_before': canonical, 'canonical_after': corpus.canonical_hash(after), 'color_policies': policies})
            except Exception as exc:
                fatal = exc if isinstance(exc, IdentityFailure) else None
                result.update({'status': 'FATAL_IDENTITY_FAILURE' if fatal else 'ITEM_TECHNICAL_FAILURE', 'failure': {'type': type(exc).__name__, 'message': str(exc)}})
                before, previous_before = None, None
            finally:
                arrays, after, values = [], None, None
            result.update({'candidate': row, 'identity': checks['identity'], 'attempt': attempt.name,
                           'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started})
            save(directory / 'record.json', result)
            if fatal:
                raise fatal
            cache_bytes += sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
            require(cache_bytes <= config['artifact_bytes'], 'ARTIFACT_LIMIT')
        counts[result['status']] = counts.get(result['status'], 0) + 1
        split_counts.setdefault(row['split'], {})
        split_counts[row['split']][result['status']] = split_counts[row['split']].get(result['status'], 0) + 1
        save(attempt / 'progress.json', {'processed': index + 1, 'expected': len(queue['rows']), 'counts': counts, 'worker_cpu_seconds': time.process_time()})
        if (index + 1) % 128 == 0:
            save(attempt / f'projection_{index + 1:04d}.json', {'processed': index + 1, 'remaining': len(queue['rows']) - index - 1,
                 'elapsed_wall_seconds': time.monotonic() - work_started, 'elapsed_cpu_seconds': time.process_time() - work_cpu,
                 'estimated_remaining_cpu_seconds': (time.process_time() - work_cpu) / (index + 1) * (len(queue['rows']) - index - 1),
                 'claim': 'Observed average including resume verification; projection only, no threshold or coverage change.'})
    save(attempt / 'worker_report.json', {'status': 'COMPLETE_PROVISIONAL_NOT_ADMISSION', 'counts': counts,
                                         'counts_by_split': split_counts, 'processed': len(queue['rows']), 'identity': checks['identity'], 'worker_cpu_seconds': time.process_time()})


def launch(config_path):
    config, queue, checks = preflight(config_path)
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'running.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    attempt, process, registry = None, None, {}
    started, parent_cpu = time.monotonic(), time.process_time()
    cpu, rss, peak, reason, code = 0., 0, 0, None, -1
    try:
        save(output / 'queue_lock.json', {'identity': checks['identity'], **queue})
        attempt = output / 'attempts' / f'{len(list((output / "attempts").glob("*"))) + 1:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        save(attempt / 'identity.json', checks['identity'])
        # Worker preflight excludes its own live attempt from completed cost accounting.
        save(attempt / 'accounting.json', {'aggregate_cpu_seconds': 0., 'active_wall_seconds': 0., 'live': True})
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--config', str(config_path), '--worker', str(attempt)], stdout=log, stderr=subprocess.STDOUT)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                cpu, rss = corpus.sample_tree(owned, registry)
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
                    corpus.stop_tree(registry)
                    break
                time.sleep(.1)
            code = process.wait()
            final_cpu, final_rss = corpus.sample_tree(owned, registry)
            cpu = max(cpu, final_cpu)
            peak = max(peak, final_rss + psutil.Process().memory_info().rss)
    finally:
        corpus.stop_tree(registry)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if attempt is not None:
            progress = attempt / 'progress.json'
            if progress.exists():
                cpu = max(cpu, read(progress)['worker_cpu_seconds'])
            accounting = {'aggregate_cpu_seconds': cpu + time.process_time() - parent_cpu + .2,
                          'active_wall_seconds': time.monotonic() - started, 'peak_tree_rss_bytes': peak,
                          'owned_processes': [{'pid': pid, 'create_time': created, 'cpu_seconds': entry['cpu_seconds'], 'peak_rss_bytes': entry['peak_rss_bytes']} for (pid, created), entry in registry.items()],
                          'exit_code': code, 'failure': reason if reason else (None if code == 0 else 'WORKER_EXCEPTION_OR_INTERRUPTION'), 'live': False}
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

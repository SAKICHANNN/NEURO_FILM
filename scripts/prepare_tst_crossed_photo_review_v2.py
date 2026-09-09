import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('frozen_photo_v1', ROOT / 'scripts/prepare_tst_crossed_photo_review.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, digest, require = old.read, old.digest, old.require
STAGES = ('portrait_face', 'donor', 'neutral_text_built_scene', 'chromatic_nature_or_objects', 'lowkey_environment')
SPLITS = ('fit', 'monitor', 'constructed_test')
DEFAULT = ROOT / 'configs/tst_crossed_photo_review_v2.json'


def create_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def ordered_queue(rows, imported):
    require(len({r['operator_key'] for r in rows}) == len(rows), 'DUPLICATE_POPULATION_KEY')
    require(set(imported) <= {r['operator_key'] for r in rows}, 'IMPORT_OUTSIDE_POPULATION')
    queues = {s: deque(r['operator_key'] for r in rows if r['split'] == s and r['operator_key'] not in imported) for s in SPLITS}
    require(sum(map(len, queues.values()))+len(imported) == len(rows), 'UNKNOWN_SPLIT')
    result = []
    while any(queues.values()):
        for split in SPLITS:
            if queues[split]:
                result.append(queues[split].popleft())
    return result


def field_bool(value):
    return value is None or type(value) is bool


def validate_observation(review, record, record_hash):
    require(review.get('operator_key') == record['operator_key'] and review.get('stage') == record['stage'], 'WRONG_REVIEW_STAGE_OR_OPERATOR')
    require(review.get('stage_record_sha256') == record_hash, 'WRONG_REVIEW_RECORD_HASH')
    require(review.get('probe_id') == record['probe_id'], 'WRONG_PROBE_ID')
    require(review.get('reviewed_artifact_hashes') == [record['artifact']['sha256']], 'MISSING_DUPLICATE_OR_FABRICATED_ARTIFACT_HASH')
    require(review.get('actual_viewed') is True and isinstance(review.get('reviewer_kind'), str) and bool(review['reviewer_kind'].strip()), 'VIEW_ATTESTATION_REQUIRED')
    require(isinstance(review.get('reason'), str) and bool(review['reason'].strip()), 'REVIEW_REASON_REQUIRED')
    require(review.get('decision') in ('PASS', 'FAIL', 'UNCERTAIN'), 'INVALID_REVIEW_DECISION')
    require(review.get('training_admitted', False) is False, 'TRAINING_ADMISSION_FORBIDDEN')
    if record['stage'] == 'donor':
        fields = ('pair_content_geometry_compatible', 'donor_reconstruction_appearance_compatible')
        require(all(k in review and field_bool(review[k]) for k in fields), 'DONOR_LABELS_REQUIRED')
        failed = any(review[k] is False for k in fields)
        passed = all(review[k] is True for k in fields)
    else:
        fields = ('comfortable', 'visibly_changed_comfortably_rich', 'objectionable_new_content_or_detail_loss')
        require(all(k in review and field_bool(review[k]) for k in fields), 'PROBE_LABELS_REQUIRED')
        failed = review['comfortable'] is False or review['objectionable_new_content_or_detail_loss'] is True
        passed = review['comfortable'] is True and review['objectionable_new_content_or_detail_loss'] is False and type(review['visibly_changed_comfortably_rich']) is bool
        require(review['visibly_changed_comfortably_rich'] is not True or passed, 'RICH_LABEL_CONFLICT')
    if review['decision'] == 'FAIL':
        require(failed, 'FAIL_REQUIRES_UNEQUIVOCAL_MANDATORY_FAILURE')
    elif review['decision'] == 'PASS':
        require(passed and not failed, 'PASS_REQUIRES_COMPLETE_POSITIVE_MANDATORY_LABELS')
    else:
        require(not failed, 'KNOWN_FAILURE_MUST_BE_FAIL_NOT_UNCERTAIN')
    return review


def outcome(observations):
    require(len(observations) <= 5, 'TOO_MANY_REVIEWS')
    for i, review in enumerate(observations):
        require(review['stage'] == STAGES[i], 'REVIEW_ORDER_OR_DUPLICATE')
        require(i == 0 or observations[i-1]['decision'] == 'PASS', 'REVIEW_AFTER_TERMINAL')
    if observations and observations[-1]['decision'] == 'FAIL':
        return 'FAIL'
    if observations and observations[-1]['decision'] == 'UNCERTAIN':
        return 'UNCERTAIN_NOT_ADMITTED'
    if len(observations) < 5:
        return 'PENDING'
    return 'PASS_CURATOR_ONLY' if sum(r['visibly_changed_comfortably_rich'] is True for r in observations if r['stage'] != 'donor') >= 3 else 'FAIL_INSUFFICIENT_COMFORTABLE_RICH_CHANGE'


def verify_artifact(directory, artifact):
    require(Path(artifact['path']).name == artifact['path'], 'ARTIFACT_PATH_ESCAPE')
    p = directory / artifact['path']
    require(p.stat().st_size == artifact['bytes'] and digest(p) == artifact['sha256'], 'ARTIFACT_CHANGED')


def import_reviews(config, population):
    batch = read(ROOT / config['legacy_batch'])
    by_key = {r['operator_key']: r for r in batch['records']}
    all_rows = []
    for path in config['legacy_reviews']:
        data = read(ROOT / path)
        all_rows.extend(data.get('rows', data.get('reviews', [])))
    require(len(all_rows) == len(by_key) == 12, 'LEGACY_REVIEW_COUNT')
    imported = {}
    for review in all_rows:
        key = review['operator_key']
        require(key in by_key and key not in imported and key in population, 'LEGACY_DUPLICATE_OR_UNKNOWN')
        item = by_key[key]
        rp = Path(item['record_path'])
        require(digest(rp) == item['record_sha256'], 'LEGACY_RECORD_CHANGED')
        record = read(rp)
        for artifact in record['artifacts']:
            verify_artifact(rp.parent, artifact)
        sheets = [a for a in record['artifacts'] if a['path'].endswith('.png')]
        expected = [a['sha256'] for a in sheets]
        actual = review['reviewed_artifact_hashes']
        require(len(expected) == len(actual) == len(set(actual)) == 5 and set(expected) == set(actual), 'LEGACY_VIEW_HASH_MISMATCH')
        require([p['probe_id'] for p in review['probes']] == [a['probe_id'] for a in sheets if 'probe_id' in a], 'LEGACY_PROBE_ID_MISMATCH')
        require(review['pair_content_geometry_compatible'] is True and review['donor_reconstruction_appearance_compatible'] is True, 'LEGACY_PAIR_NOT_CONFIRMED')
        probes = review['probes']
        require(all(type(p[k]) is bool for p in probes for k in ('comfortable', 'visibly_changed_comfortably_rich', 'objectionable_new_content_or_detail_loss')), 'LEGACY_LABELS_INVALID')
        passed = all(p['comfortable'] and not p['objectionable_new_content_or_detail_loss'] for p in probes) and sum(p['visibly_changed_comfortably_rich'] for p in probes) >= 3
        require(str(review['final_decision']).startswith('PASS' if passed else 'FAIL'), 'LEGACY_DECISION_CONFLICT')
        imported[key] = {'status': 'PASS_CURATOR_ONLY' if passed else 'FAIL', 'record_sha256': item['record_sha256'], 'review_sha256': old.s3.object_hash(review), 'reviewed_artifact_hashes': actual, 'training_admitted': False}
    return imported


def frozen_inputs(config):
    legacy = read(ROOT / config['legacy_input'])
    rows = legacy['survivors']
    require(len(rows) == config['population'] == 3456, 'POPULATION_CHANGED')
    population = {r['operator_key']: r for r in rows}
    imported = import_reviews(config, population)
    ordered = ordered_queue(rows, imported)
    return {'population_keys': [r['operator_key'] for r in rows], 'rows': rows, 'imported': imported, 'queue': ordered,
            'queue_sha256': old.s3.object_hash(ordered), 'stage_order': list(STAGES), 'split_cycle': list(SPLITS)}


def costs(config, active=None):
    ledger_config = {**config, 'output': config['output']+'/unused_ledger_namespace'}
    ledger = old.s3.ledger(ledger_config)
    output = ROOT / config['output']
    checkpoint = read(output / 'accounting_checkpoint.json') if (output / 'accounting_checkpoint.json').exists() else {'attempts': 0, 'cpu': 0., 'wall': 0., 'bytes': 0}
    if checkpoint['attempts']:
        ap = output / 'attempts' / f"{checkpoint['attempts']:04d}" / 'accounting.json'
        require(digest(ap) == checkpoint['last_sha256'], 'ACCOUNTING_HEAD_CHANGED')
        require(read(ap)['failure'] is None, 'PRIOR_FAILED_ATTEMPT_REQUIRES_ADJUDICATION')
    next_attempt = output / 'attempts' / f"{checkpoint['attempts']+1:04d}"
    require(not next_attempt.exists() or next_attempt == active, 'UNACCOUNTED_ATTEMPT_REQUIRES_ADJUDICATION')
    legacy = old.s3.attempts_accounting(ROOT / config['legacy_output'] / 'attempts')
    extra_cpu, extra_wall = config['external_photo_cpu_seconds'], config['external_photo_wall_reserve_seconds']
    ledger.update(stage_cpu=checkpoint['cpu'], stage_wall=checkpoint['wall'],
                  photo_cpu=checkpoint['cpu']+sum(r['aggregate_cpu_seconds'] for r in legacy)+extra_cpu,
                  photo_wall=checkpoint['wall']+sum(r['active_wall_seconds'] for r in legacy)+extra_wall,
                  retained_bytes=config['legacy_candidate_bytes']+checkpoint['bytes'], checkpoint=checkpoint)
    ledger['study_cpu'] += checkpoint['cpu']+extra_cpu
    ledger['study_wall'] += checkpoint['wall']+extra_wall
    return ledger


def verify_record(directory, identity, item, stage, probe_id):
    rp = directory / 'record.json'
    record = read(rp)
    require(record['identity'] == identity and record['operator_key'] == item['operator_key'] and record['stage'] == stage and record['probe_id'] == probe_id, 'STAGE_IDENTITY_CHANGED')
    verify_artifact(directory, record['artifact'])
    return record, digest(rp)


def cursor(config, inputs, identity):
    output = ROOT / config['output']
    pointer = read(output / 'cursor.json') if (output / 'cursor.json').exists() else {'index': 0, 'previous_terminal_sha256': None}
    index = pointer['index']
    require(type(index) is int and 0 <= index <= len(inputs['queue']), 'INVALID_CURSOR_INDEX')
    if index:
        tp = output / 'terminal' / (inputs['queue'][index-1]+'.json')
        require(digest(tp) == pointer['previous_terminal_sha256'], 'TERMINAL_HEAD_CHANGED')
        require(read(tp)['identity'] == identity and read(tp)['next_index'] == index, 'TERMINAL_HEAD_IDENTITY')
    if index == len(inputs['queue']):
        return {'status': 'ALL_CANDIDATES_RECORDED_FINAL_AUDIT_REQUIRED', 'terminal_count': index+12}, None, None
    key = inputs['queue'][index]
    item = next(r for r in inputs['rows'] if r['operator_key'] == key)
    metadata = read(ROOT / config['probe_manifest'])['rows']
    observations = []
    for stage in STAGES:
        probe = None if stage == 'donor' else next(r for r in metadata if r['split'] == item['split'] and r['category'] == stage)
        pid = None if probe is None else probe['component']
        directory = output / 'candidate' / key / stage
        rp, reviewp = directory / 'record.json', directory / 'review.json'
        state = {'operator_key': key, 'stage': stage, 'probe_id': pid, 'terminal_count': index+12, 'queue_index': index}
        if not rp.exists():
            require(not directory.exists(), 'PARTIAL_STAGE_REQUIRES_ADJUDICATION')
            return {'status': 'READY_TO_RENDER', **state}, item, probe
        record, rh = verify_record(directory, identity, item, stage, pid)
        if not reviewp.exists():
            return {'status': 'AWAITING_EXTERNAL_REVIEW', **state, 'record_path': str(rp), 'record_sha256': rh, 'artifact': record['artifact']}, item, probe
        observations.append(validate_observation(read(reviewp), record, rh))
        require(outcome(observations) == 'PENDING', 'UNCOMMITTED_TERMINAL_REQUIRES_ADJUDICATION')
    raise RuntimeError('INVALID_CURSOR')


def commit_terminal_if_needed(config, identity, key, queue_index):
    output = ROOT / config['output']
    observations, receipts = [], []
    for stage in STAGES:
        directory = output / 'candidate' / key / stage
        if not (directory / 'review.json').exists():
            break
        review = read(directory / 'review.json')
        observations.append(review)
        receipts.append({'stage': stage, 'record_sha256': digest(directory / 'record.json'), 'review_sha256': digest(directory / 'review.json')})
    state = outcome(observations)
    if state == 'PENDING':
        return
    pointer = read(output / 'cursor.json') if (output / 'cursor.json').exists() else {'index': 0, 'previous_terminal_sha256': None}
    require(pointer['index'] == queue_index, 'CURSOR_COMMIT_CONFLICT')
    tp = output / 'terminal' / (key+'.json')
    create_json(tp, {'identity': identity, 'operator_key': key, 'next_index': queue_index+1, 'previous_terminal_sha256': pointer['previous_terminal_sha256'], 'status': state, 'receipts': receipts, 'unreviewed_stages': list(STAGES[len(observations):]), 'unreviewed_status': 'NOT_REVIEWED_AFTER_NECESSARY_FAILURE' if state == 'FAIL' else ('NOT_REVIEWED_AFTER_UNCERTAIN_EXCLUSION' if state == 'UNCERTAIN_NOT_ADMITTED' else None), 'training_admitted': False})
    temporary = output / 'cursor.pending.json'
    create_json(temporary, {'index': queue_index+1, 'previous_terminal_sha256': digest(tp)})
    temporary.replace(output / 'cursor.json')


def audit_history(config, inputs, identity, active=None):
    output = ROOT / config['output']
    previous, totals = None, {'cpu': 0., 'wall': 0., 'bytes': 0}
    checkpoint = costs(config, active)['checkpoint']
    for n in range(1, checkpoint['attempts']+1):
        ap = output / 'attempts' / f'{n:04d}' / 'accounting.json'
        a = read(ap)
        require(a['previous_accounting_sha256'] == previous and a['failure'] is None, 'ACCOUNTING_CHAIN_FAILURE')
        totals['cpu'] += a['aggregate_cpu_seconds']; totals['wall'] += a['active_wall_seconds']; totals['bytes'] += a['added_candidate_bytes']
        previous = digest(ap)
    require(all(abs(totals[k]-checkpoint[k]) < 1e-8 for k in totals), 'ACCOUNTING_TOTAL_CHANGED')
    previous, seen = None, set()
    population = {r['operator_key']: r for r in inputs['rows']}
    metadata = read(ROOT / config['probe_manifest'])['rows']
    for index, key in enumerate(inputs['queue']):
        tp = output / 'terminal' / (key+'.json')
        require(tp.exists(), 'POPULATION_REVIEW_INCOMPLETE')
        terminal = read(tp)
        require(terminal['identity'] == identity and terminal['operator_key'] == key and terminal['next_index'] == index+1 and terminal['previous_terminal_sha256'] == previous, 'TERMINAL_CHAIN_FAILURE')
        reviews = []
        for receipt in terminal['receipts']:
            stage = receipt['stage']; directory = output / 'candidate' / key / stage
            pid = None if stage == 'donor' else next(r['component'] for r in metadata if r['split'] == population[key]['split'] and r['category'] == stage)
            record, rh = verify_record(directory, identity, population[key], stage, pid)
            require(rh == receipt['record_sha256'] and digest(directory / 'review.json') == receipt['review_sha256'], 'HISTORICAL_REVIEW_CHANGED')
            reviews.append(validate_observation(read(directory / 'review.json'), record, rh)); seen.add(str(directory))
        require(outcome(reviews) == terminal['status'] and terminal['unreviewed_stages'] == list(STAGES[len(reviews):]), 'TERMINAL_OUTCOME_CHANGED')
        previous = digest(tp)
    actual = {str(p.parent) for p in (output / 'candidate').rglob('record.json')}
    require(actual == seen, 'UNEXPECTED_RENDERED_STAGE')
    require(sum(p.stat().st_size for p in (output / 'candidate').rglob('*') if p.is_file()) == checkpoint['bytes'], 'RETAINED_STORAGE_CHANGED')
    return {'status': 'FULL_HISTORY_INTEGRITY_VERIFIED_NOT_TRAINING', 'population': len(inputs['population_keys']), 'imported': len(inputs['imported']), 'new_terminal': len(inputs['queue']), 'training_admitted': False}


def preflight(config_path, active=None):
    config = read(config_path)
    require(digest(Path(__file__)) == config['entry_sha256'], 'ENTRY_CHANGED')
    for path, sha in config['pins'].items():
        require(digest(ROOT / path) == sha, 'PIN_CHANGED:'+path)
    require(config['stage_order'] == list(STAGES) and config['split_cycle'] == list(SPLITS), 'ORDER_PROTOCOL_CHANGED')
    require(config['threads'] == 2 and config['strength'] == 1 and config['chunk_pixels'] == 32768, 'FROZEN_RENDER_PROTOCOL_CHANGED')
    inputs = frozen_inputs(config)
    identity = {'config_sha256': digest(config_path), 'entry_sha256': config['entry_sha256'], 'input_sha256': old.s3.object_hash(inputs)}
    output = ROOT / config['output']
    if (output / 'input_lock.json').exists():
        require(read(output / 'input_lock.json') == {'identity': identity, **inputs}, 'INPUT_LOCK_CHANGED')
    ledger = costs(config, active)
    require(ledger['photo_cpu'] < config['cpu_seconds'] and ledger['photo_wall'] < config['wall_seconds'], 'PHOTO_BUDGET_EXHAUSTED')
    require(ledger['study_cpu']+config['cpu_seconds']-ledger['photo_cpu'] <= config['study_cpu_seconds'], 'STUDY_CPU_CAPACITY')
    require(ledger['study_wall']+config['wall_seconds']-ledger['photo_wall'] <= config['study_wall_seconds'], 'STUDY_WALL_CAPACITY')
    require(ledger['retained_bytes'] < config['cache_bytes'], 'PHOTO_STORAGE_EXHAUSTED')
    state, item, probe = cursor(config, inputs, identity)
    return config, inputs, {'identity': identity, 'accounting': ledger, 'state': state, 'photo_reads': 0}, item, probe


def load_grid(config, item):
    require(digest(Path(item['s2_record_path'])) == item['s2_record_sha256'] and digest(Path(item['s3_record_path'])) == item['s3_record_sha256'], 'PARENT_RECORD_CHANGED')
    parent = Path(item['s2_record_path']).parent
    expected = next(a for a in item['artifacts'] if a['path'] == 'absolute_grid.npy')
    verify_artifact(parent, expected)
    grid = np.load(parent / 'absolute_grid.npy', allow_pickle=False)
    require(grid.dtype == np.float64 and grid.shape == (343, 3) and np.isfinite(grid).all(), 'GRID_INVALID')
    require(grid.min() >= -config['bound_tol'] and grid.max() <= 1+config['bound_tol'], 'GRID_OUT_OF_BOUNDS')
    return grid


def stage_panel(config, item, stage, probe, grid):
    if stage != 'donor':
        expanded = next(r for r in old.s3.probe_metadata(config) if r['probe_id'] == probe['component'])
        source = old.s3.load_probes(config, [expanded])[0].source
        rendered = old.render_native(source, grid, config['chunk_pixels'])
        boxes = [c['box_xyxy_exclusive'] for c in probe['crops']]
        board, layout = old.panel([source, rendered], ['Original source', 'F(source) strength1'], boxes, probe['category']+' | '+probe['representative'], config['overview_side'])
        return board, layout, boxes
    files = {r['path']: r for r in read(ROOT / config['acquisition_config'])['files']}
    arrays = []
    for role in ('before', 'after'):
        candidate = item['candidate']
        name = candidate[role]
        require(files[name]['sha256'] == candidate[role+'_encoded_sha256'], 'DONOR_HASH_METADATA_CHANGED')
        path = old.s2.corpus.confined(ROOT / config['originals'], name)
        old.s2.verify_encoded(path, files[name])
        hp = ROOT / config['headers'] / (hashlib.sha256(name.encode()).hexdigest()+'.json')
        header = old.s2.verified_header(hp, candidate[role+'_header_sha256'])
        source, _ = old.s2.corpus.decode(path, header)
        expected = candidate['before_canonical_sha256'] if role == 'before' else item['canonical_after']
        require(old.s2.corpus.canonical_hash(source) == expected, 'DONOR_CANONICAL_CHANGED')
        arrays.append(source)
    require(arrays[0].shape == arrays[1].shape, 'DONOR_SHAPE_MISMATCH')
    rendered = old.render_native(arrays[0], grid, config['chunk_pixels'])
    boxes = old.donor_boxes(arrays[0].shape, config['donor_patch_side'])
    board, layout = old.panel([*arrays, rendered], ['P original', 'R original', 'F(P) strength1'], boxes, 'Donor compatibility / reconstructed appearance', config['overview_side'])
    return board, layout, boxes


def worker(config_path, attempt):
    config, inputs, checks, item, probe = preflight(config_path, attempt)
    require(read(attempt / 'identity.json') == checks['identity'], 'WORKER_IDENTITY')
    request = read(attempt / 'request.json')
    require(request['state'] == checks['state'], 'WORKER_CURSOR_CHANGED')
    if request['action'] == 'audit':
        require(checks['state']['status'] == 'ALL_CANDIDATES_RECORDED_FINAL_AUDIT_REQUIRED', 'AUDIT_BEFORE_COMPLETE')
        result = audit_history(config, inputs, checks['identity'], attempt)
        create_json(attempt / 'worker_report.json', {**result, 'worker_cpu_seconds': time.process_time()})
        return
    stage = checks['state']['stage']
    directory = ROOT / config['output'] / 'candidate' / item['operator_key'] / stage
    if request['action'] == 'ingest':
        require(checks['state']['status'] == 'AWAITING_EXTERNAL_REVIEW', 'NOT_WAITING_FOR_REVIEW')
        path = Path(request['review_path'])
        require(path.stat().st_size <= config['metadata_reserve_bytes']//2, 'REVIEW_METADATA_TOO_LARGE')
        require(checks['accounting']['retained_bytes']+config['metadata_reserve_bytes'] <= config['cache_bytes'], 'REVIEW_STORAGE_CAP')
        require(digest(path) == request['review_sha256'], 'REVIEW_SUBMISSION_CHANGED')
        record, rh = verify_record(directory, checks['identity'], item, stage, checks['state']['probe_id'])
        review = validate_observation(read(path), record, rh)
        require(not (directory / 'review.json').exists(), 'DUPLICATE_OR_CONFLICTING_REVIEW')
        create_json(directory / 'review.json', review)
        commit_terminal_if_needed(config, checks['identity'], item['operator_key'], checks['state']['queue_index'])
    else:
        require(request['action'] == 'render' and checks['state']['status'] == 'READY_TO_RENDER', 'NOT_READY_TO_RENDER')
        directory.mkdir(parents=True, exist_ok=False)
        create_json(directory / 'started.json', {'identity': checks['identity'], 'attempt': attempt.name})
        threadpool_limits(config['threads'])
        grid = load_grid(config, item)
        board, layout, boxes = stage_panel(config, item, stage, probe, grid)
        budget = {'bytes': checks['accounting']['retained_bytes'], 'limit': min(config['cache_bytes'], checks['accounting']['retained_bytes']+config['dispatch_bytes'])-config['metadata_reserve_bytes']}
        artifact = old.put_artifact(directory, 'sheet.png', board, budget)
        create_json(directory / 'record.json', {'identity': checks['identity'], 'operator_key': item['operator_key'], 'split': item['split'], 'stage': stage, 'probe_id': checks['state']['probe_id'], 'artifact': artifact, 'native_layout': layout, 'source_boxes': boxes, 'strength': 1.0, 'training_admitted': False, 'human_preference': 'UNMEASURED', 'claim': 'Overview and listed native patches only; external review required. Non-gating v1 full-color support diagnostic is not recomputed in this rendering-only stage.'})
    create_json(attempt / 'worker_report.json', {'status': 'COMPLETE_STAGE_NOT_TRAINING', 'worker_cpu_seconds': time.process_time()})


def launch(config_path, review_path=None, audit=False):
    initial_cpu, initial_wall = time.process_time(), time.monotonic()
    config, inputs, checks, _, _ = preflight(config_path)
    action = 'audit' if audit else ('ingest' if review_path is not None else 'render')
    expected = 'ALL_CANDIDATES_RECORDED_FINAL_AUDIT_REQUIRED' if audit else ('AWAITING_EXTERNAL_REVIEW' if review_path is not None else 'READY_TO_RENDER')
    require(checks['state']['status'] == expected, 'CURSOR_DOES_NOT_ALLOW_ACTION')
    output = ROOT / config['output']; output.mkdir(parents=True, exist_ok=True)
    lock = output / 'running.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    target = output / 'candidate' / checks['state'].get('operator_key', '__audit__') / checks['state'].get('stage', '__none__')
    before_bytes = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    attempt, process, registry = None, None, {}
    cpu, peak, reason, code = 0., 0, None, -1
    try:
        if not (output / 'input_lock.json').exists():
            create_json(output / 'input_lock.json', {'identity': checks['identity'], **inputs})
        attempt = output / 'attempts' / f"{checks['accounting']['checkpoint']['attempts']+1:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        create_json(attempt / 'identity.json', checks['identity'])
        request = {'action': action, 'state': checks['state']}
        if review_path is not None:
            require(review_path.is_absolute() and review_path.is_relative_to(ROOT), 'REVIEW_MUST_BE_LOCAL_CHECKOUT_FILE')
            request.update(review_path=str(review_path), review_sha256=digest(review_path))
        create_json(attempt / 'request.json', request)
        old.save(attempt / 'accounting.json', {'aggregate_cpu_seconds': 0., 'active_wall_seconds': 0., 'live': True})
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--config', str(config_path), '--worker', str(attempt)], stdout=log, stderr=subprocess.STDOUT)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                cpu, rss = old.s2.corpus.sample_tree(owned, registry)
                peak = max(peak, rss+psutil.Process().memory_info().rss)
                elapsed = time.monotonic()-initial_wall
                aggregate = cpu+time.process_time()-initial_cpu
                prior = checks['accounting']
                if aggregate >= min(config['audit_dispatch_seconds'] if audit else config['dispatch_cpu_seconds'], config['cpu_seconds']-prior['photo_cpu'], config['study_cpu_seconds']-prior['study_cpu']):
                    reason = 'CPU_LIMIT'
                elif elapsed >= min(config['audit_dispatch_seconds'] if audit else config['dispatch_wall_seconds'], config['wall_seconds']-prior['photo_wall'], config['study_wall_seconds']-prior['study_wall']):
                    reason = 'WALL_LIMIT'
                elif peak > config['rss_bytes']:
                    reason = 'RSS_LIMIT'
                if reason:
                    old.s2.corpus.stop_tree(registry)
                    break
                time.sleep(.1)
            code = process.wait()
            last, _ = old.s2.corpus.sample_tree(owned, registry)
            cpu = max(cpu, last)
    finally:
        old.s2.corpus.stop_tree(registry)
        if process is not None and process.poll() is None:
            process.kill(); process.wait()
        if attempt is not None:
            for filename in ('worker_report.json', 'worker_final.json'):
                if (attempt / filename).exists():
                    cpu = max(cpu, read(attempt / filename)['worker_cpu_seconds'])
            accounting = {'aggregate_cpu_seconds': cpu+time.process_time()-initial_cpu+.2, 'active_wall_seconds': time.monotonic()-initial_wall, 'peak_tree_rss_bytes': peak, 'live': False, 'exit_code': code, 'failure': reason or (None if code == 0 else 'WORKER_EXCEPTION_OR_INTERRUPTION'), 'previous_accounting_sha256': checks['accounting']['checkpoint'].get('last_sha256')}
            target = output / 'candidate' / checks['state'].get('operator_key', '__audit__') / checks['state'].get('stage', '__none__')
            after_bytes = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
            accounting['added_candidate_bytes'] = after_bytes-before_bytes
            old.save(attempt / 'accounting.json', accounting)
            previous = checks['accounting']['checkpoint']
            snapshot = {'attempts': previous['attempts']+1, 'cpu': previous['cpu']+accounting['aggregate_cpu_seconds'], 'wall': previous['wall']+accounting['active_wall_seconds'], 'bytes': previous['bytes']+accounting['added_candidate_bytes'], 'last_sha256': digest(attempt / 'accounting.json')}
            temporary = output / 'accounting_checkpoint.pending.json'
            create_json(temporary, snapshot)
            temporary.replace(output / 'accounting_checkpoint.json')
            if code != 0 or reason:
                create_json(attempt / 'failure.json', accounting)
        lock.unlink()
    return int(code != 0 or reason is not None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--ingest', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.config, args.worker)
        finally:
            create_json(args.worker / 'worker_final.json', {'worker_cpu_seconds': time.process_time()})
    elif args.run or args.ingest or args.audit:
        require(int(args.run)+int(args.ingest is not None)+int(args.audit) == 1, 'ONE_ACTION_ONLY')
        raise SystemExit(launch(args.config, args.ingest, args.audit))
    else:
        print(json.dumps(preflight(args.config)[2], indent=2))

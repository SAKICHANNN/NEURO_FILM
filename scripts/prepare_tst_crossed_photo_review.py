import argparse
from collections import Counter
import hashlib
import importlib.util
import io
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
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('photo_frozen_s3', ROOT / 'scripts/run_tst_crossed_visibility.py')
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
s2 = s3.s2
read, save, digest, require = s3.read, s3.save, s3.digest, s3.require
DEFAULT = ROOT / 'configs/tst_crossed_photo_review_v1.json'
SURVIVOR = 'VISIBILITY_POSSIBLE_AWAITING_AGENT_REVIEW'


def completed_s3(config):
    output = ROOT / config['s3_output']
    if (output / 'running.lock').exists() or not (output / 'report.json').exists():
        return None
    report = read(output / 'report.json')
    if report.get('status') != 'COMPLETE_VISIBILITY_SCREEN_NOT_ADMISSION':
        return None
    attempts = sorted((output / 'attempts').glob('*'))
    require(bool(attempts), 'S3_ACCOUNTING_MISSING')
    accounting = read(attempts[-1] / 'accounting.json')
    require(not accounting.get('live') and accounting.get('failure') is None and accounting['exit_code'] == 0, 'S3_LATEST_ATTEMPT_INCOMPLETE')
    require(report['last_attempt_accounting'] == accounting, 'S3_COMPLETION_ACCOUNTING_CHANGED')
    lock = read(output / 'input_lock.json')
    identity = lock['identity']
    require(identity['config_sha256'] == digest(ROOT / config['s3_config']) and identity['entry_sha256'] == digest(ROOT / config['s3_entry']), 'S3_CODE_IDENTITY_CHANGED')
    require(identity['input_sha256'] == s3.object_hash({k: v for k, v in lock.items() if k != 'identity'}), 'S3_INPUT_LOCK_CHANGED')
    require(report['identity'] == identity and report['processed'] == len(lock['rows']) == config['expected_candidates'], 'INCOMPLETE_S3_CENSUS')
    counts, partitions, survivors, outcomes = Counter(), {}, [], []
    for item in lock['rows']:
        path = output / 'candidate' / item['operator_key'] / 'record.json'
        record = s3.reusable(path, identity, item)
        require(record['status'] in {SURVIVOR, 'PROVISIONAL_OPERATOR_NOT_VISIBLE_PAIR_UNREVIEWED', 'NOT_ELIGIBLE_S2_OUTCOME_RETAINED', 'VISIBILITY_TECHNICAL_FAILURE'}, 'INVALID_S3_TERMINAL_STATUS')
        counts[record['status']] += 1
        partitions.setdefault(item['split'], Counter())[record['status']] += 1
        outcomes.append({'operator_key': item['operator_key'], 'status': record['status'], 'record_sha256': digest(path)})
        if record['status'] != SURVIVOR:
            continue
        require(item['s2_status'] == s3.SOLVED and record['strength'] == 1.0 and record['threshold_mean_delta_e2000'] == 4.0, 'SURVIVOR_CONTRACT')
        require(record['computed_visible_count'] >= 3 and record['uncomputed_count'] == 0 and record['visibility_possible'], 'SURVIVOR_VISIBILITY_CONTRACT')
        s2_record_path = Path(item['record_path'])
        require(digest(s2_record_path) == item['record_sha256'], 'INHERITED_S2_RECORD_CHANGED')
        parent = read(s2_record_path)
        s2.reusable(s2_record_path.parent, lock['s2_identity'], parent['candidate'])
        survivors.append({'operator_key': item['operator_key'], 'split': item['split'], 's3_record_path': str(path),
                           's3_record_sha256': digest(path), 's2_record_path': str(s2_record_path),
                           's2_record_sha256': item['record_sha256'], 'candidate': parent['candidate'],
                           'canonical_after': parent['canonical_after'], 'artifacts': parent['artifacts']})
    require(dict(counts) == report['counts'] and partitions == report['counts_by_split'], 'S3_REPORT_COUNTS_CHANGED')
    return {'s3_report_sha256': digest(output / 'report.json'), 's3_input_lock_sha256': digest(output / 'input_lock.json'),
            's3_identity': identity, 'outcomes': outcomes, 'survivors': survivors, 'counts': dict(counts)}


def preflight(config_path, batch, active_attempt=None):
    config = read(config_path)
    require(digest(Path(__file__)) == config['entry_sha256'], 'ENTRY_CHANGED')
    for path, expected in config['pins'].items():
        require(digest(ROOT / path) == expected, 'PIN_CHANGED:' + path)
    require(config['batch_size'] == 12 and config['strength'] == 1.0, 'REVIEW_PROTOCOL_CHANGED')
    receipt = completed_s3(config)
    if receipt is None:
        return config, None, {'status': 'WAIT_S3_COMPLETE', 'photo_reads': 0, 'partial_S3_records_read': 0}
    costs = s3.ledger(config, active_attempt)
    inputs = {**receipt, 'prior_attempt_pins': costs['prior_attempt_pins']}
    identity = {'config_sha256': digest(config_path), 'entry_sha256': config['entry_sha256'], 'input_sha256': s3.object_hash(inputs)}
    output = ROOT / config['output']
    if (output / 'input_lock.json').exists():
        require(read(output / 'input_lock.json') == {'identity': identity, **inputs}, 'INPUT_LOCK_CHANGED')
    for path in (output / 'attempts').glob('*'):
        require(read(path / 'identity.json') == identity, 'ATTEMPT_IDENTITY_CHANGED')
    total = (len(inputs['survivors']) + config['batch_size'] - 1) // config['batch_size']
    require(batch >= 0 and (batch < total or total == batch == 0), 'BATCH_OUTSIDE_FROZEN_POPULATION')
    if batch > 0:
        require((output / 'batches' / f'{batch - 1:04d}.json').exists(), 'PREVIOUS_BATCH_NOT_PREPARED')
    require(costs['stage_cpu'] < config['cpu_seconds'] and costs['stage_wall'] < config['wall_seconds'], 'PHOTO_STAGE_BUDGET_EXHAUSTED')
    require(costs['study_cpu'] + config['cpu_seconds'] - costs['stage_cpu'] <= config['study_cpu_seconds'], 'STUDY_CPU_CAPACITY')
    require(costs['study_wall'] + config['wall_seconds'] - costs['stage_wall'] <= config['study_wall_seconds'], 'STUDY_WALL_CAPACITY')
    return config, inputs, {'status': 'READY', 'photo_reads': 0, 'identity': identity, 'accounting': costs,
                            'survivors': len(inputs['survivors']), 'batch': batch, 'batches': total}


def donor_boxes(shape, side):
    height, width = shape[:2]
    w, h = min(side, width), min(side, height)
    boxes = []
    for cy, cx in ((height//4, width//4), (height//4, 3*width//4), (3*height//4, width//4), (3*height//4, 3*width//4)):
        x, y = max(0, min(width-w, cx-w//2)), max(0, min(height-h, cy-h//2))
        boxes.append([x, y, x+w, y+h])
    return boxes


def display(values):
    return Image.fromarray(((values.astype(np.uint32) + 128)//257).astype(np.uint8))


def png_bytes(image):
    meta = PngImagePlugin.PngInfo()
    meta.add(b'sRGB', b'\x00')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', pnginfo=meta)
    return buffer.getvalue()


def panel(images, names, boxes, title, overview_side):
    require(len(images) == len(names), 'PANEL_COLUMNS')
    source_shape = images[0].shape
    require(all(a.shape == source_shape for a in images), 'MATCHED_NATIVE_SHAPES')
    for x0, y0, x1, y1 in boxes:
        require(0 <= x0 < x1 <= source_shape[1] and 0 <= y0 < y1 <= source_shape[0], 'NATIVE_CROP_OUTSIDE_SOURCE')
    gap, label = 8, 24
    widths = [x1-x0 for x0, y0, x1, y1 in boxes]
    heights = [y1-y0 for x0, y0, x1, y1 in boxes]
    column = max([384, overview_side, *widths])
    board = Image.new('RGB', (len(images)*(column+gap)+gap, label*3+overview_side+sum(h+label for h in heights)+gap), (25, 25, 25))
    draw, font = ImageDraw.Draw(board), ImageFont.load_default(size=16)
    draw.text((gap, 2), title, fill='white', font=font)
    layout = []
    for index, (values, name) in enumerate(zip(images, names)):
        x = gap + index*(column+gap)
        draw.text((x, label), name + ' | overview only', fill='white', font=font)
        view = display(values)
        view.thumbnail((overview_side, overview_side), Image.Resampling.LANCZOS)
        board.paste(view, (x, label*2))
        y = label*2 + overview_side
        for number, (box, height) in enumerate(zip(boxes, heights)):
            draw.text((x, y), f'native 1:1 crop {number}: {box}', fill='white', font=font)
            x0, y0, x1, y1 = box
            crop = display(values[y0:y1, x0:x1])
            board.paste(crop, (x, y+label))
            layout.append({'column': index, 'crop': number, 'source_xyxy': box, 'board_xyxy': [x, y+label, x+x1-x0, y+label+y1-y0]})
            y += label+height
    return board, layout


def render_native(source, grid, chunk):
    output = np.empty_like(source)
    flat, destination = source.reshape(-1, 3), output.reshape(-1, 3)
    for start in range(0, len(flat), chunk):
        prediction = s2.core.render_absolute(flat[start:start+chunk].astype(float)/65535, grid, 7)
        require(np.isfinite(prediction).all(), 'NONFINITE_RENDER')
        destination[start:start+chunk] = np.rint(np.clip(prediction, 0, 1)*65535).astype(np.uint16)
    return output


def probe_support(fit_colors, probe, chunk):
    count, fraction, weight, observed = 0, 0., 0., None
    flat = probe.reshape(-1, 3)
    for start in range(0, len(flat), chunk):
        colors = flat[start:start+chunk].astype(float)/65535
        row = s2.core.support_summary(fit_colors, colors, 7)
        count += len(colors)
        fraction += len(colors)*row['scoring_pixels_touching_unobserved_nodes_fraction']
        weight += len(colors)*row['mean_unobserved_node_interpolation_weight']
        observed = row['observed_nodes']
    return {'observed_nodes': observed, 'nodes': 343, 'native_probe_pixels': count,
            'pixels_touching_unobserved_nodes_fraction': fraction/count, 'mean_unobserved_node_interpolation_weight': weight/count,
            'claim': 'Support diagnostic only; extrapolated colors remain regularized extension. No numeric admission cutoff.'}


def put_artifact(directory, filename, image, budget):
    data = png_bytes(image)
    require(budget['bytes'] + len(data) <= budget['limit'], 'REVIEW_CACHE_LIMIT_NO_DELETION')
    path = directory / filename
    require(not path.exists(), 'ARTIFACT_ALREADY_EXISTS')
    path.write_bytes(data)
    budget['bytes'] += len(data)
    return {'path': filename, 'sha256': digest(path), 'bytes': len(data), 'shape': [image.height, image.width, 3]}


def put_json(directory, filename, value, budget):
    data = (json.dumps(value, indent=2) + '\n').encode('utf-8')
    require(budget['bytes'] + len(data) <= budget['limit'], 'REVIEW_CACHE_LIMIT_NO_DELETION')
    path = directory / filename
    require(not path.exists(), 'ARTIFACT_ALREADY_EXISTS')
    path.write_bytes(data)
    budget['bytes'] += len(data)
    return {'path': filename, 'sha256': digest(path), 'bytes': len(data)}


def review_template(item, artifacts, probes):
    return {'operator_key': item['operator_key'], 'status': 'PENDING_EXTERNAL_REVIEW', 'training_admitted': False,
            'reviewer_kind': None, 'human_preference': 'UNMEASURED', 'reviewed_artifact_hashes': [],
            'required_artifact_hashes': [a['sha256'] for a in artifacts], 'pair_content_geometry_compatible': None,
            'donor_reconstruction_appearance_compatible': None,
            'probes': [{'probe_id': p['component'], 'comfortable': None, 'visibly_changed_comfortably_rich': None,
                        'objectionable_new_content_or_detail_loss': None, 'reason': None} for p in probes],
            'supplementary_native_regions_required': [], 'final_decision': None, 'reason': None,
            'review_rule': 'Uncertain/mismatched P/R or incompatible reconstruction excludes. All4 probes comfortable with no objectionable new content/detail loss; >=3 visibly changed and comfortably rich. Numeric S3 does not replace review. Record actual viewed hashes and reasons externally; renderer never admits.',
            'claim': 'Overviews do not certify whole-native content. Native evidence covers listed rectangles only; request recorded supplementary regions when insufficient.'}


def prepare_candidate(config, item, probe_metadata, directory, budget):
    require(digest(Path(item['s3_record_path'])) == item['s3_record_sha256'], 'S3_RECORD_CHANGED')
    require(digest(Path(item['s2_record_path'])) == item['s2_record_sha256'], 'S2_RECORD_CHANGED')
    parent = Path(item['s2_record_path']).parent
    artifacts_by_name = {a['path']: a for a in item['artifacts']}
    for name in ('absolute_grid.npy', 'coordinates.npz'):
        require(digest(parent / name) == artifacts_by_name[name]['sha256'], 'INHERITED_ARTIFACT_CHANGED')
    grid = np.load(parent / 'absolute_grid.npy', allow_pickle=False)
    require(grid.dtype == np.float64 and grid.shape == (343, 3) and np.isfinite(grid).all(), 'GRID_CONTRACT')
    require(grid.min() >= -config['bound_tol'] and grid.max() <= 1+config['bound_tol'], 'GRID_BOUND_TOLERANCE')
    candidate = item['candidate']
    files = {r['path']: r for r in read(ROOT / config['acquisition_config'])['files']}
    arrays, color_policies = [], []
    for role in ('before', 'after'):
        name = candidate[role]
        require(files[name]['sha256'] == candidate[role+'_encoded_sha256'], 'DONOR_ENCODED_METADATA_CHANGED')
        path = s2.corpus.confined(ROOT / config['originals'], name)
        s2.verify_encoded(path, files[name])
        header_path = ROOT / config['headers'] / (hashlib.sha256(name.encode()).hexdigest()+'.json')
        header = s2.verified_header(header_path, candidate[role+'_header_sha256'])
        values, color_policy = s2.corpus.decode(path, header)
        canonical = candidate['before_canonical_sha256'] if role == 'before' else item['canonical_after']
        require(s2.corpus.canonical_hash(values) == canonical, 'DONOR_CANONICAL_CHANGED')
        arrays.append(values)
        color_policies.append({'role': role, 'policy': color_policy})
    before, after = arrays
    require(before.shape == after.shape, 'DONOR_GEOMETRY_CHANGED')
    with np.load(parent / 'coordinates.npz', allow_pickle=False) as coordinates:
        indices = coordinates['fit_flat_indices']
        require(np.array_equal(coordinates['shape'], before.shape) and indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), 'FIT_COORDINATE_CONTRACT')
        require(len(indices) > 0 and indices.min() >= 0 and indices.max() < before.shape[0]*before.shape[1], 'FIT_COORDINATE_BOUNDS')
        fit_colors = before.reshape(-1, 3)[indices].astype(float)/65535
    rendered = render_native(before, grid, config['chunk_pixels'])
    boxes = donor_boxes(before.shape, config['donor_patch_side'])
    board, layout = panel([before, after, rendered], ['P original', 'R original', 'F(P) strength1'], boxes, 'Donor compatibility / reconstructed appearance', config['overview_side'])
    artifacts = [{**put_artifact(directory, '00_donor.png', board, budget), 'native_layout': layout, 'source_boxes': boxes}]
    del rendered, board, arrays, before, after, values
    support = []
    frozen = [r for r in read(ROOT / config['probe_manifest'])['rows'] if r['split'] == item['split']]
    for number, (probe, metadata) in enumerate(zip(probe_metadata, frozen)):
        require(probe.probe_id == metadata['component'] and probe.split == item['split'], 'PROBE_PANEL_PARTITION_OR_ORDER')
        rendered = render_native(probe.source, grid, config['chunk_pixels'])
        boxes = [c['box_xyxy_exclusive'] for c in metadata['crops']]
        board, layout = panel([probe.source, rendered], ['Original source', 'F(source) strength1'], boxes,
                              metadata['category'] + ' | ' + metadata['representative'], config['overview_side'])
        artifacts.append({**put_artifact(directory, f'{number+1:02d}_probe.png', board, budget), 'native_layout': layout, 'source_boxes': boxes, 'probe_id': probe.probe_id})
        del rendered, board
        support.append({'probe_id': probe.probe_id, **probe_support(fit_colors, probe.source, config['chunk_pixels'])})
    template = review_template(item, artifacts, frozen)
    artifacts.append(put_json(directory, 'review_template.json', template, budget))
    return {'status': 'PREPARED_AWAITING_EXTERNAL_REVIEW', 'training_admitted': False, 'artifacts': artifacts,
            'support': support, 'donor_color_policies': color_policies, 'photographic_admission': 'NOT_REVIEWED', 'human_preference': 'UNMEASURED',
            'render_contract': 'Unchanged absolute grid at1.0; native interpolation then clip/round uint16; sRGB8 display; overview Lanczos only; patches native1:1.',
            'claim': 'Five sheets of bounded visual evidence, no automatic quality decision. Four fixed donor patches are source-only quadrant-centred samples, not whole-native inspection.'}


def reusable(directory, identity, item):
    record = read(directory / 'record.json')
    require(record['identity'] == identity and record['input'] == item, 'RESUME_IDENTITY_CHANGED')
    require(record['status'] == 'PREPARED_AWAITING_EXTERNAL_REVIEW', 'PREVIOUS_FAILURE_REQUIRES_ADJUDICATION')
    for artifact in record['artifacts']:
        require(digest(directory / artifact['path']) == artifact['sha256'], 'REVIEW_ARTIFACT_CHANGED')
    return record


def worker(config_path, batch, attempt):
    config, inputs, checks = preflight(config_path, batch, attempt)
    require(checks['status'] == 'READY' and read(attempt / 'identity.json') == checks['identity'], 'WORKER_BINDING_CHANGED')
    require(read(attempt / 'request.json') == {'batch': batch}, 'WORKER_BATCH_REQUEST_CHANGED')
    threadpool_limits(config['threads'])
    output = ROOT / config['output']
    rows = inputs['survivors'][batch*config['batch_size']:(batch+1)*config['batch_size']]
    budget = {'bytes': sum(p.stat().st_size for p in (output / 'candidate').rglob('*') if p.is_file()), 'limit': config['cache_bytes']}
    probes, split, records = [], None, []
    for item in rows:
        directory = output / 'candidate' / item['operator_key']
        if (directory / 'record.json').exists():
            record = reusable(directory, checks['identity'], item)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            require(budget['bytes'] + config['metadata_reserve_bytes'] <= budget['limit'], 'REVIEW_CACHE_LIMIT_BEFORE_CANDIDATE')
            s2.begin_candidate(directory, checks['identity'], attempt.name)
            budget['bytes'] += sum(p.stat().st_size for p in directory.glob('started_*.json'))
            started, cpu = time.monotonic(), time.process_time()
            try:
                if split != item['split']:
                    probes = []
                    split = item['split']
                    probe_rows = [r for r in s3.probe_metadata(config) if r['split'] == split]
                    probes = s3.load_probes(config, probe_rows)
                record = prepare_candidate(config, item, probes, directory, budget)
            except Exception as exc:
                save(directory / 'record.json', {'status': 'REVIEW_PREPARATION_FAILED', 'identity': checks['identity'], 'input': item,
                     'failure': {'type': type(exc).__name__, 'message': str(exc)}, 'training_admitted': False,
                     'cpu_seconds': time.process_time()-cpu, 'wall_seconds': time.monotonic()-started})
                raise
            record.update({'identity': checks['identity'], 'input': item, 'cpu_seconds': time.process_time()-cpu, 'wall_seconds': time.monotonic()-started})
            put_json(directory, 'record.json', record, budget)
        records.append({'operator_key': item['operator_key'], 'record_path': str(directory / 'record.json'), 'record_sha256': digest(directory / 'record.json'),
                        'review_order': [str(directory / a['path']) for a in record['artifacts'] if a['path'].endswith('.png')]})
        save(attempt / 'progress.json', {'processed': len(records), 'expected': len(rows), 'worker_cpu_seconds': time.process_time(), 'cache_bytes': budget['bytes']})
    save(attempt / 'worker_report.json', {'status': 'BATCH_PREPARED_NOT_REVIEWED', 'batch': batch, 'identity': checks['identity'],
         'population_survivors': len(inputs['survivors']), 'batch_size': len(rows), 'records': records,
         'worker_cpu_seconds': time.process_time(), 'review_instructions': 'View five sheets per candidate at original pixels; record viewed artifact hashes and external decisions. Supplement insufficient native evidence; uncertainty is not admission.'})


def launch(config_path, batch):
    config, inputs, checks = preflight(config_path, batch)
    require(checks['status'] == 'READY', 'PHOTO_REVIEW_NOT_READY')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    batch_path = output / 'batches' / f'{batch:04d}.json'
    require(not batch_path.exists(), 'BATCH_ALREADY_PREPARED')
    lock = output / 'running.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    attempt, process, registry = None, None, {}
    started, parent_cpu = time.monotonic(), time.process_time()
    cpu, peak, reason, code = 0., 0, None, -1
    try:
        save(output / 'input_lock.json', {'identity': checks['identity'], **inputs})
        attempt = output / 'attempts' / f'{len(list((output / "attempts").glob("*")))+1:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        save(attempt / 'identity.json', checks['identity'])
        save(attempt / 'request.json', {'batch': batch})
        save(attempt / 'accounting.json', {'aggregate_cpu_seconds': 0., 'active_wall_seconds': 0., 'live': True})
        with (attempt / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--config', str(config_path), '--batch', str(batch), '--worker', str(attempt)], stdout=log, stderr=subprocess.STDOUT)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                cpu, rss = s2.corpus.sample_tree(owned, registry)
                peak = max(peak, rss+psutil.Process().memory_info().rss)
                elapsed, costs = time.monotonic()-started, checks['accounting']
                aggregate = cpu+time.process_time()-parent_cpu
                if costs['stage_cpu']+aggregate >= config['cpu_seconds'] or costs['study_cpu']+aggregate >= config['study_cpu_seconds']:
                    reason = 'CPU_LIMIT'
                elif costs['stage_wall']+elapsed >= config['wall_seconds'] or costs['study_wall']+elapsed >= config['study_wall_seconds']:
                    reason = 'WALL_LIMIT'
                elif peak > config['rss_bytes']:
                    reason = 'RSS_LIMIT'
                if reason:
                    s2.corpus.stop_tree(registry)
                    break
                time.sleep(.1)
            code = process.wait()
            last_cpu, last_rss = s2.corpus.sample_tree(owned, registry)
            cpu, peak = max(cpu, last_cpu), max(peak, last_rss+psutil.Process().memory_info().rss)
    finally:
        s2.corpus.stop_tree(registry)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if attempt is not None:
            for name in ('progress.json', 'worker_report.json'):
                path = attempt / name
                if path.exists():
                    cpu = max(cpu, read(path)['worker_cpu_seconds'])
            accounting = {'aggregate_cpu_seconds': cpu+time.process_time()-parent_cpu+.2, 'active_wall_seconds': time.monotonic()-started,
                 'peak_tree_rss_bytes': peak, 'batch': batch, 'live': False, 'exit_code': code,
                 'failure': reason if reason else (None if code == 0 else 'WORKER_EXCEPTION_OR_INTERRUPTION'),
                 'owned_processes': [{'pid': pid, 'create_time': created, 'cpu_seconds': row['cpu_seconds'], 'peak_rss_bytes': row['peak_rss_bytes']} for (pid, created), row in registry.items()]}
            save(attempt / 'accounting.json', accounting)
            if code == 0 and reason is None and (attempt / 'worker_report.json').exists():
                save(batch_path, {**read(attempt / 'worker_report.json'), 'last_attempt_accounting': accounting})
            else:
                save(attempt / 'failure.json', accounting)
        lock.unlink()
    return int(code != 0 or reason is not None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.config, args.batch, args.worker)
        finally:
            path = args.worker / 'progress.json'
            save(path, {**(read(path) if path.exists() else {}), 'worker_cpu_seconds': time.process_time()})
    elif args.run:
        raise SystemExit(launch(args.config, args.batch))
    else:
        print(json.dumps(preflight(args.config, args.batch)[2], indent=2))

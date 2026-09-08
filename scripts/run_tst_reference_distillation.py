import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROCESS_STARTED_WALL = time.monotonic()

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from src.eval import tst_reference_distillation as net
from src.eval import tst_reference_kernel as kernel
from src.eval import tst_reference_response as frozen

CONFIG = ROOT/'configs/tst_reference_distillation_v1.json'
STAGES = ('fit_features', 'train', 'predict', 'diagnostic_features', 'score')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def legacy_module():
    spec = importlib.util.spec_from_file_location('frozen_check', ROOT/'scripts/run_tst_reference_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_keys(cfg, stage):
    rows = cfg['fit_rows'] if stage in ('fit_features', 'train') else cfg['development_rows']
    roles = ('P', 'Y') if stage in ('fit_features', 'diagnostic_features') else ('X', 'R') if stage == 'predict' else ('X', 'R', 'P', 'Y')
    return sorted({key(row, role) for row in rows for role in roles})


def key(row, role):
    return f"{row['group']:02d}_{role}{row['target'] if role != 'X' else ''}"


def remaining_budget(cfg, reports):
    cpu = cfg['remaining_cpu_seconds']-sum(r['accounted_cpu_seconds'] for r in reports)
    wall = cfg['remaining_wall_seconds']-sum(r['supervisor']['wall_seconds'] for r in reports)
    return cpu, wall


def preflight(stage):
    cfg = load_json(CONFIG)
    assert stage in STAGES and cfg['threads'] == 2
    assert cfg['entry_sha256'] == frozen.digest(Path(__file__))
    for name, sha in cfg['pins'].items():
        assert frozen.digest(ROOT/name) == sha, name
    assert cfg['architecture'] == {'feature': 3998, 'hidden': 64, 'code': 16, 'source': 6, 'grid': 7, 'epsilon': 1e-12}
    assert cfg['updates'] == {'T': 1000, 'S': 1000, 'S0': 1000, 'D': 2000}
    ledger = load_json(ROOT/cfg['pair_report'])['new_study_budget_ledger']
    assert cfg['remaining_cpu_seconds'] == ledger['remaining_feature_training_scoring_cpu_seconds']
    assert cfg['remaining_wall_seconds'] == ledger['remaining_active_wall_seconds']
    assert cfg['pair_processing_cpu'] == ledger['processing_cpu_seconds']
    assert len(cfg['fit_rows']) == 43 and len(cfg['development_rows']) == 16
    assert set(cfg['images']) == {name for s in STAGES for name in stage_keys(cfg, s)}
    admission = load_json(ROOT/cfg['pair_admission'])
    assert admission['status'] == 'ADMITTED_FOR_PAIRED_MECHANISM_RESEARCH'
    assert admission['report_sha256'] == cfg['pins'][cfg['pair_report']]
    expected = {(r['group'], r['target']) for r in cfg['fit_rows']+cfg['development_rows']}
    assert len(admission['pairs']) == 59
    assert {(r['group'], r['target']) for r in admission['pairs'] if r['same_content']} == expected
    for name in stage_keys(cfg, stage):
        meta = cfg['images'][name]
        assert frozen.digest(ROOT/meta['path']) == meta['sha256'], name
    prior, reports = {}, []
    status = 'READY'
    for earlier in STAGES[:STAGES.index(stage)]:
        directory = ROOT/cfg['output']/earlier
        if not (directory/'report.json').exists():
            status = f'WAIT_{earlier.upper()}'
            break
        report = load_json(directory/'report.json')
        assert report['status'] == 'COMPLETE_PHASE', earlier
        assert report['checks']['config_sha256'] == frozen.digest(CONFIG)
        for artifact in report['artifacts']:
            assert frozen.digest(directory/artifact['path']) == artifact['sha256']
        reports.append(report)
        prior[earlier] = frozen.digest(directory/'report.json')
    cpu, wall = remaining_budget(cfg, reports)
    if stage in ('diagnostic_features', 'score') and status == 'READY':
        lock = load_json(ROOT/cfg['output']/'predict/prediction_lock.json')
        assert lock['development_P_Y_decodes'] == 0 and lock['teacher_weights_loaded'] is False
        assert len(lock['native_outputs']) == 96
    if min(cpu, wall) <= 0:
        status = 'RESOURCE_INCOMPLETE'
    return cfg, {'status': status, 'stage': stage, 'config_sha256': frozen.digest(CONFIG),
                 'remaining_cpu_seconds': {'learning': max(0., cpu)},
                 'remaining_wall_seconds': max(0., wall), 'prior_report_sha256': prior}, legacy_module()


def read_image(cfg, image_key):
    value = tifffile.imread(ROOT/cfg['images'][image_key]['path'])
    assert value.dtype == np.uint16 and value.ndim == 3 and value.shape[-1] == 3
    return value.astype(np.float64)/65535


def load_features(paths):
    cache = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            assert not set(cache).intersection(data.files)
            cache.update({k: data[k] for k in data.files})
    return cache


def transformed(old, cache, keys):
    blocks = [np.stack([cache[f'{k}_{suffix}'] for k in keys]) for suffix in ('stats', 'vgg')]
    phi = kernel.pre_pca(old['projections']['image']['standards'], blocks)
    source = frozen.project(old['projections']['source'], blocks)
    assert phi.shape == (len(keys), 3998) and source.shape == (len(keys), 6)
    assert np.isfinite(phi).all() and np.isfinite(source).all()
    return torch.from_numpy(phi.astype(np.float32)), torch.from_numpy(source.astype(np.float32))


def extract(cfg, directory, stage):
    model = frozen.load_vgg(ROOT)
    cache = {}
    keys = stage_keys(cfg, stage)
    assert len(keys) == (86 if stage == 'fit_features' else 32)
    for i, name in enumerate(keys):
        stats, learned = frozen.descriptors(read_image(cfg, name), model)
        cache[f'{name}_stats'], cache[f'{name}_vgg'] = stats, learned
        if (i+1) % 10 == 0:
            print('feature', i+1, len(keys), flush=True)
    np.savez(directory/'features.npz', **cache)
    return {'feature_forwards': len(keys), 'keys': keys, 'training_updates': 0,
            'privileged_development': stage == 'diagnostic_features'}


def data_bundle(cfg, rows, cache):
    old = legacy_module().load_model(ROOT/cfg['old_model'])
    data = {}
    for role in ('X', 'R', 'P', 'Y'):
        data[role], data[f's{role}'] = transformed(old, cache, [key(r, role) for r in rows])
    return data


def sample(cfg, name, fold):
    whole = read_image(cfg, name)
    indices = frozen.scoring_folds(*whole.shape[:2], cfg['images'][name]['sampling_sha256'],
                                  cfg['seed'], cfg['pixels_per_fold'])[fold]
    return whole.reshape(-1, 3)[indices], indices, whole.shape


def quadratics(cfg, rows):
    xs, ys, ps, rs, cached_x = [], [], [], [], {}
    for row in rows:
        xkey = key(row, 'X')
        if xkey not in cached_x:
            cached_x[xkey] = sample(cfg, xkey, 0)
        x, indices, shape = cached_x[xkey]
        y = read_image(cfg, key(row, 'Y'))
        assert y.shape == shape
        p, pi, pshape = sample(cfg, key(row, 'P'), 0)
        r = read_image(cfg, key(row, 'R'))
        assert r.shape == pshape
        xs.append(x)
        ys.append(y.reshape(-1, 3)[indices])
        ps.append(p)
        rs.append(r.reshape(-1, 3)[pi])
    ids = [r['group'] for r in rows]
    return (net.PixelQuadratic(xs, ys, ids, cfg['architecture']['grid'], True),
            net.PixelQuadratic(ps, rs, ids, cfg['architecture']['grid'], False))


def state_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def train(cfg, directory, legacy):
    cache = load_features([ROOT/cfg['feature_caches'][0], ROOT/cfg['output']/'fit_features/features.npz'])
    data = data_bundle(cfg, cfg['fit_rows'], cache)
    query, reference = quadratics(cfg, cfg['fit_rows'])
    weights = torch.tensor(net.row_weights([r['group'] for r in cfg['fit_rows']]), dtype=torch.float64)
    teacher, decoder, initial_student, initial_decoder, student_state = net.initialize(cfg)
    initial = {'student': state_digest(student_state), 'decoder': state_digest(initial_decoder)}
    history, teacher_codes, teacher_decoder = {}, None, None
    for arm in ('T', 'S', 'S0', 'D'):
        if arm == 'T':
            encoder = teacher
        else:
            encoder = copy.deepcopy(initial_student)
            encoder.load_state_dict(student_state, strict=True)
            decoder = net.Decoder(cfg['architecture']['source'], cfg['architecture']['code'], cfg['architecture']['grid'])
            decoder.load_state_dict(initial_decoder if arm == 'D' else teacher_decoder, strict=True)
            decoder.requires_grad_(arm == 'D')
        parameters = [p for module in (encoder, decoder) for p in module.parameters() if p.requires_grad]
        optim = cfg['optimizer']
        optimizer = torch.optim.Adam(parameters, lr=optim['lr'], betas=tuple(optim['betas']), eps=optim['eps'], weight_decay=optim['weight_decay'])
        local = {**cfg, 'kd_weight': cfg['kd_weight'] if arm == 'S' else 0.}
        history[arm] = []
        for step in range(1, cfg['updates'][arm]+1):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = net.objective(encoder, decoder, data, query, reference, weights, local,
                                        paired=arm == 'T', teacher_codes=teacher_codes if arm == 'S' else None)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss {arm} {step}')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f'Nonfinite gradient {arm} {step}')
            norm = torch.nn.utils.clip_grad_norm_(parameters, optim['gradient_norm_clip'], error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError(f'Nonfinite weights {arm} {step}')
            if step == 1 or step % 100 == 0 or step == cfg['updates'][arm]:
                history[arm].append({'step': step, 'loss_before_update': float(loss.detach()),
                    'gradient_norm_before_clip': float(norm), **{k: float(v) for k, v in parts.items()}})
                legacy.save(directory/'progress.json', history)
                print(arm, step, float(loss.detach()), flush=True)
        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, directory/f'{arm}.pt')
        if arm == 'T':
            teacher.eval().requires_grad_(False)
            decoder.eval().requires_grad_(False)
            teacher_decoder = copy.deepcopy(decoder.state_dict())
            with torch.no_grad():
                teacher_codes = (net.paired_code(teacher, data['P'], data['R']).detach(),
                                 net.paired_code(teacher, data['X'], data['Y']).detach())
    return {'training_updates': cfg['updates'], 'total_updates': sum(cfg['updates'].values()),
            'initialization_sha256': initial, 'history': history,
            'final_checkpoint_only': True, 'development_features_loaded': False,
            'claim': 'Scheduled updates completed; no global optimizer convergence certificate or scientific promotion.'}


def load_arm(cfg, arm):
    a = cfg['architecture']
    encoder = net.Encoder(a['feature']*(2 if arm == 'T' else 1), a['hidden'], a['code'], a['epsilon'])
    decoder = net.Decoder(a['source'], a['code'], a['grid'])
    state = torch.load(ROOT/cfg['output']/'train'/f'{arm}.pt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(state['encoder'], strict=True)
    decoder.load_state_dict(state['decoder'], strict=True)
    return encoder.eval().requires_grad_(False), decoder.eval().requires_grad_(False)


def infer_after_only(encoder, decoder, source_scores, reference_features):
    with torch.no_grad():
        return net.after_only(encoder, decoder, source_scores, reference_features).double().numpy()


def render_native(cfg, directory, image_key, values, output_name, strengths, legacy, crop_boxes=None):
    source = read_image(cfg, image_key)
    with tifffile.TiffFile(ROOT/cfg['images'][image_key]['path']) as handle:
        tag = handle.pages[0].tags.get(34675)
        assert tag is not None
        icc = tag.value
    flat = source.reshape(-1, 3)
    prediction = np.empty_like(flat)
    for offset in range(0, len(flat), cfg['render_chunk_pixels']):
        prediction[offset:offset+cfg['render_chunk_pixels']] = frozen.render(flat[offset:offset+cfg['render_chunk_pixels']], values)
    prediction = prediction.reshape(source.shape)
    for strength in strengths:
        result = source*(1-strength)+prediction*strength
        label = 'native' if strength == 1 else 'strength08'
        path = directory/f'{output_name}_{label}.tif'
        legacy.write16(path, result, icc)
        if crop_boxes:
            for name, box in crop_boxes.items():
                if box:
                    x, y, w, h = box
                    Image.fromarray(np.rint(result[y:y+h, x:x+w]*255).astype(np.uint8)).save(
                        directory/f'{output_name}_{label}_{name}.png', icc_profile=icc)


def predict(cfg, directory, legacy):
    cache = load_features([ROOT/cfg['feature_caches'][1]])
    assert all('_X_' in k or '_R1_' in k or '_R2_' in k for k in cache)
    old = legacy.load_model(ROOT/cfg['old_model'])
    rows = cfg['development_rows']
    _, source = transformed(old, cache, [key(r, 'X') for r in rows])
    reference, _ = transformed(old, cache, [key(r, 'R') for r in rows])
    crops = load_json(ROOT/cfg['source_crops'])
    operators = {}
    output = directory/'predictions'
    output.mkdir()
    for arm in ('S', 'S0', 'D'):
        encoder, decoder = load_arm(cfg, arm)
        values = infer_after_only(encoder, decoder, source, reference)
        swapped = infer_after_only(encoder, decoder, source, reference.reshape(8, 2, -1).flip(1).reshape(16, -1))
        assert np.array_equal(swapped, values.reshape(8, 2, 343, 3)[:, ::-1].reshape(16, 343, 3))
        for row, u in zip(rows, values, strict=True):
            name = f"{key(row, 'Y')}_{arm}"
            operators[name] = u
            boxes = crops['groups'][row['group']-24]['regions'] if arm == 'S' else None
            render_native(cfg, output, key(row, 'X'), u, name, cfg['preview_strengths'], legacy, boxes)
    np.savez(directory/'operators.npz', **operators)
    native = [{'path': str(p.relative_to(directory)), 'sha256': frozen.digest(p)} for p in sorted(output.glob('*.tif'))]
    assert len(native) == 96
    lock = {'native_outputs': native, 'operators_sha256': frozen.digest(directory/'operators.npz'),
            'teacher_weights_loaded': False, 'development_P_Y_decodes': 0,
            'source_crops_sha256': frozen.digest(ROOT/cfg['source_crops']),
            'model_sha256': {a: frozen.digest(ROOT/cfg['output']/'train'/f'{a}.pt') for a in ('S', 'S0', 'D')},
            'claim': 'After-only locked before privileged development features. All development owners already consumed.'}
    legacy.save(directory/'prediction_lock.json', lock)
    return lock


def metric_panel(cfg, rows, operators, legacy):
    groups = []
    for g in dict.fromkeys(r['group'] for r in rows):
        selected = [r for r in rows if r['group'] == g]
        x, indices, shape = sample(cfg, f'{g:02d}_X', 1)
        lab_x = rgb2lab(x)
        targets, predictions, records = [], {a: [] for a in operators}, []
        for r in selected:
            whole = read_image(cfg, key(r, 'Y'))
            assert whole.shape == shape
            y = whole.reshape(-1, 3)[indices]
            targets.append(y)
            lab_y = rgb2lab(y)
            identity = float(deltaE_ciede2000(lab_x, lab_y).mean())
            record = {'target': r['target'], 'identity_error': identity, 'arms': {}}
            for arm, values in operators.items():
                p = frozen.render(x, values[key(r, 'Y')])
                predictions[arm].append(p)
                err = deltaE_ciede2000(rgb2lab(p), lab_y)
                record['arms'][arm] = {'error_mean': float(err.mean()), 'error_p95': float(np.quantile(err, .95)),
                    'appearance_pass': bool(err.mean() <= max(cfg['gates']['appearance_absolute'], cfg['gates']['appearance_fraction']*identity)),
                    'change_native_mean': float(deltaE_ciede2000(rgb2lab(p), lab_x).mean()),
                    'new_boundary_vs_X': float(np.any(((p <= 0)|(p >= 1)) & (x > 0) & (x < 1), axis=1).mean()),
                    'new_boundary_vs_Y': float(np.any(((p <= 0)|(p >= 1)) & (y > 0) & (y < 1), axis=1).mean())}
            records.append(record)
        groups.append({'group': g, 'targets': records, 'switches':
                       {a: legacy.switching(v, targets) for a, v in predictions.items()} if len(selected) == 2 else {}})
    group_q = {a: np.array([np.mean([t['arms'][a]['error_mean'] for t in r['targets']])/
                               max(1., np.mean([t['identity_error'] for t in r['targets']])) for r in groups]) for a in operators}
    summary = {'groups': len(groups), 'Q': {a: float(v.mean()) for a, v in group_q.items()},
        'group_Q': {a: v.tolist() for a, v in group_q.items()},
        'joint': {a: sum(bool(r['switches']) and r['switches'][a]['pass'] and all(t['arms'][a]['appearance_pass'] for t in r['targets']) for r in groups) for a in operators},
        'singletons': {a: sum(len(r['targets']) == 1 and r['targets'][0]['arms'][a]['appearance_pass'] for r in groups) for a in operators}}
    return {'groups': groups, 'summary': summary}


def score(cfg, directory, legacy):
    paths = [ROOT/p for p in cfg['feature_caches']]+[ROOT/cfg['output']/stage/'features.npz' for stage in ('fit_features', 'diagnostic_features')]
    cache = load_features(paths)
    fit = data_bundle(cfg, cfg['fit_rows'], cache)
    dev = data_bundle(cfg, cfg['development_rows'], cache)
    fit_ops, dev_ops, own_ops, reference_ops = {}, {}, {}, {}
    with np.load(ROOT/cfg['output']/'predict/operators.npz', allow_pickle=False) as lock:
        for arm in ('S', 'S0', 'D'):
            dev_ops[arm] = {key(r, 'Y'): lock[f"{key(r, 'Y')}_{arm}"] for r in cfg['development_rows']}
    with torch.no_grad():
        for arm in ('T', 'S', 'S0', 'D'):
            encoder, decoder = load_arm(cfg, arm)
            for label, data, rows in (('fit', fit, cfg['fit_rows']), ('development', dev, cfg['development_rows'])):
                zr = net.paired_code(encoder, data['P'], data['R']) if arm == 'T' else encoder(data['R'])
                zy = net.paired_code(encoder, data['X'], data['Y']) if arm == 'T' else encoder(data['Y'])
                cross = decoder(data['sX'], zr).double().numpy()
                own = decoder(data['sX'], zy).double().numpy()
                ref = decoder(data['sP'], zr).double().numpy()
                values = {key(r, 'Y'): u for r, u in zip(rows, cross, strict=True)}
                if label == 'fit':
                    fit_ops[arm] = values
                elif arm == 'T':
                    dev_ops['T'] = values
                else:
                    assert all(np.array_equal(values[k], dev_ops[arm][k]) for k in values)
                own_ops[label, arm] = {key(r, 'Y'): u for r, u in zip(rows, own, strict=True)}
                reference_ops[label, arm] = ref
    fit_metrics = metric_panel(cfg, cfg['fit_rows'], fit_ops, legacy)
    dev_metrics = metric_panel(cfg, cfg['development_rows'], dev_ops, legacy)
    legacy.save(directory/'fit_metrics.json', fit_metrics)
    legacy.save(directory/'development_metrics.json', dev_metrics)
    auxiliary = {}
    diagnostic = directory/'privileged_predictions'
    diagnostic.mkdir()
    for label, rows in (('fit', cfg['fit_rows']), ('development', cfg['development_rows'])):
        aux_query = metric_panel(cfg, rows, {a: own_ops[label, a] for a in ('T', 'S', 'S0', 'D')}, legacy)
        reference_records = []
        for i, r in enumerate(rows):
            p, indices, shape = sample(cfg, key(r, 'P'), 1)
            target = read_image(cfg, key(r, 'R'))
            assert target.shape == shape
            y = target.reshape(-1, 3)[indices]
            record = {'group': r['group'], 'target': r['target'], 'identity_error': float(deltaE_ciede2000(rgb2lab(p), rgb2lab(y)).mean()), 'arms': {}}
            for a in ('T', 'S', 'S0', 'D'):
                pred = frozen.render(p, reference_ops[label, a][i])
                record['arms'][a] = float(deltaE_ciede2000(rgb2lab(pred), rgb2lab(y)).mean())
            reference_records.append(record)
            if label == 'development':
                for suffix, src, u in (('T_cross', key(r, 'X'), dev_ops['T'][key(r, 'Y')]),
                    ('T_query_own', key(r, 'X'), own_ops[label, 'T'][key(r, 'Y')]),
                    ('T_reference_own', key(r, 'P'), reference_ops[label, 'T'][i])):
                    render_native(cfg, diagnostic, src, u, f"{key(r, 'Y')}_{suffix}", [1.], legacy)
        auxiliary[label] = {'query_own': aux_query, 'reference_own_rows': reference_records,
                            'claim': 'Separate self-reconstruction diagnostics; never included in query Q.'}
    legacy.save(directory/'auxiliary_metrics.json', auxiliary)
    np.savez(directory/'diagnostic_operators.npz', **{
        f'{label}_{a}_{k}': v for label, ops in (('fit', fit_ops), ('development', dev_ops)) for a, values in ops.items() for k, v in values.items()},
        **{f'{label}_{a}_query_own_{k}': v for (label, a), values in own_ops.items() for k, v in values.items()},
        **{f'{label}_{a}_reference_own': v for (label, a), v in reference_ops.items()})
    f, d, gates = fit_metrics['summary'], dev_metrics['summary'], cfg['gates']
    competence = {a: f['joint'][a] >= gates['fit_joint'] and f['singletons'][a] == gates['fit_singletons'] and f['Q'][a] <= gates['fit_Q'] for a in ('T', 'S', 'S0', 'D')}
    response = {a: d['joint'][a] >= gates['development_joint'] and d['Q'][a] <= gates['development_Q'] for a in ('T', 'S')}
    benefit = {a: d['Q']['S'] <= gates['student_Q_ratio']*d['Q'][a] and
                  int(np.sum(np.array(d['group_Q']['S']) < d['group_Q'][a])) >= gates['student_group_wins'] for a in ('S0', 'D')}
    return {'fit_summary': f, 'development_summary': d, 'fit_competence': competence,
            'development_response': response, 'student_benefit': benefit,
            'mechanism_pass': all(competence.values()) and all(response.values()) and all(benefit.values()),
            'strict_8_of_8_milestone': d['joint']['S'] == 8,
            'photographic_review': 'PENDING all16 S native1 and48 protected regions; >=6/8 comfortable-rich-changed-distinct pairs and0severe defects. Numeric result is not promotion.',
            'claim': 'Repeatedly consumed owner-separated development, not unseen or independent human preference validation.'}


def worker(stage, directory):
    cfg, checks, legacy = preflight(stage)
    locked = load_json(directory/'lock.json')
    assert all(locked[k] == v for k, v in checks.items() if k != 'remaining_cpu_seconds')
    assert 0 < locked['remaining_cpu_seconds']['learning'] <= checks['remaining_cpu_seconds']['learning']
    assert checks['status'] == 'READY'
    checks = locked
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=cfg['threads'])
    legacy.save(directory/'report.json', {'status': 'RUNNING', 'checks': checks})
    try:
        if stage in ('fit_features', 'diagnostic_features'):
            result = extract(cfg, directory, stage)
        else:
            result = {'train': train, 'predict': predict, 'score': score}[stage](cfg, directory, legacy)
        result['status'] = 'COMPLETE_PHASE'
    except FloatingPointError as exc:
        result = {'status': 'NUMERICAL_INCOMPLETE', 'exception': str(exc), 'exception_type': type(exc).__name__}
    except MemoryError as exc:
        result = {'status': 'RESOURCE_INCOMPLETE', 'exception': str(exc), 'exception_type': type(exc).__name__}
    result['checks'] = checks
    result['artifacts'] = [{'path': str(p.relative_to(directory)), 'sha256': frozen.digest(p)} for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ('report.json', 'report.pending', 'worker.log')]
    result.update(worker_pid=os.getpid(), worker_cpu_seconds=time.process_time())
    legacy.save(directory/'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.stage, args.worker)
    else:
        config, state, legacy = preflight(args.stage)
        if args.run:
            assert state['status'] == 'READY'
            legacy.__file__, legacy.CONFIG = __file__, CONFIG
            legacy.category = lambda _: 'learning'
            runtime = {**config, 'wall_seconds_per_stage': max(0., state['remaining_wall_seconds']-(time.monotonic()-PROCESS_STARTED_WALL))}
            launch_state = copy.deepcopy(state)
            launch_state['remaining_cpu_seconds']['learning'] -= time.process_time()
            assert launch_state['remaining_cpu_seconds']['learning'] > 0 and runtime['wall_seconds_per_stage'] > 0
            code = legacy.launch(args.stage, runtime, launch_state)
            path = ROOT/config['output']/args.stage/'report.json'
            report = load_json(path)
            launcher_cpu = time.process_time()
            report['accounted_cpu_seconds'] += launcher_cpu
            report['supervisor']['worker_tree_wall_seconds'] = report['supervisor']['wall_seconds']
            report['supervisor']['wall_seconds'] = time.monotonic()-PROCESS_STARTED_WALL
            report['supervisor']['launcher_cpu_seconds'] = launcher_cpu
            report['supervisor']['worker_cpu_limit'] = launch_state['remaining_cpu_seconds']['learning']
            report['supervisor']['remaining_category_cpu_seconds'] = max(0., state['remaining_cpu_seconds']['learning']-report['accounted_cpu_seconds'])
            if report['accounted_cpu_seconds'] > state['remaining_cpu_seconds']['learning']:
                report['supervisor']['reason'] = 'CPU_FINAL_ACCOUNTING_LIMIT'
            if report['supervisor']['wall_seconds'] > state['remaining_wall_seconds']:
                report['supervisor']['reason'] = 'WALL_LIMIT'
            if report['supervisor']['reason'] in ('RSS_LIMIT', 'CPU_LIMIT', 'CPU_FINAL_ACCOUNTING_LIMIT', 'WALL_LIMIT'):
                report['status'] = 'RESOURCE_INCOMPLETE'
            report['cumulative_budget'] = {'initial_pair_preparation_cpu': config['pair_processing_cpu'],
                'remaining_cpu': max(0., state['remaining_cpu_seconds']['learning']-report['accounted_cpu_seconds']),
                'remaining_wall': max(0., state['remaining_wall_seconds']-report['supervisor']['wall_seconds']),
                'reset_allowed': False}
            legacy.save(path, report)
            raise SystemExit(0 if not code and report['status'] == 'COMPLETE_PHASE' else 1)
        print(json.dumps(state, indent=2))

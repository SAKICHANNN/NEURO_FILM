import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from src.eval import tst_correspondence_operator as core
from src.eval import tst_reference_response as frozen

CONFIG = ROOT/'configs/tst_correspondence_operator_feasibility_v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def legacy_module():
    spec = importlib.util.spec_from_file_location('frozen_supervisor', ROOT/'scripts/run_tst_reference_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight():
    cfg = read(CONFIG)
    assert cfg['entry_sha256'] == frozen.digest(Path(__file__))
    for path, digest in cfg['pins'].items():
        assert frozen.digest(ROOT/path) == digest, path
    assert cfg['dimension'] == 7 and cfg['smoothness'] == 1e-6 and cfg['tie'] == 1e-8
    assert cfg['threads'] == 2 and cfg['cpu_seconds'] == cfg['wall_seconds_per_stage'] == 600
    pairs = read(ROOT/cfg['pair_report'])['pairs']
    first = [r for r in pairs if r['split'] == 'training'][:8]
    assert [(r['group'], r['target']) for r in cfg['pairs']] == [(r['group'], r['target']) for r in first]
    assert len(cfg['probes']) == 4 and len({p['key'] for p in cfg['probes']}) == 4
    for item in [v for r in cfg['pairs'] for v in (r['before'], r['after'])]+cfg['probes']:
        assert frozen.digest(ROOT/item['path']) == item['sha256']
    return cfg, {'status': 'READY', 'config_sha256': frozen.digest(CONFIG),
                 'remaining_cpu_seconds': {'learning': cfg['cpu_seconds']},
                 'scope': '8 consumed fit donor pairs, one fold direction,32 technical probe renders; no training/newdata/human-admission claim'}, legacy_module()


def image(meta):
    value = tifffile.imread(ROOT/meta['path'])
    assert value.dtype == np.uint16 and value.ndim == 3 and value.shape[-1] == 3
    return value.astype(np.float64)/65535


def statistics(source, target, predicted):
    error = deltaE_ciede2000(rgb2lab(predicted), rgb2lab(target))
    identity = float(deltaE_ciede2000(rgb2lab(source), rgb2lab(target)).mean())
    return {'error_mean': float(error.mean()), 'error_p95': float(np.quantile(error, .95)),
            'identity_error': identity, 'appearance_predicate': bool(error.mean() <= max(1., .25*identity)),
            'claim': 'Technical old-pair reconstruction; not a new scientific or photographic gate.'}


def native(cfg, meta, values, output, legacy):
    whole = image(meta)
    flat = whole.reshape(-1, 3)
    prediction = np.empty_like(flat)
    for start in range(0, len(flat), cfg['render_chunk_pixels']):
        prediction[start:start+cfg['render_chunk_pixels']] = core.render_absolute(flat[start:start+cfg['render_chunk_pixels']], values)
    assert prediction.min() >= -1e-12 and prediction.max() <= 1+1e-12
    with tifffile.TiffFile(ROOT/meta['path']) as handle:
        icc = handle.pages[0].tags[34675].value
    legacy.write16(output, np.clip(prediction, 0., 1.).reshape(whole.shape), icc)
    return whole, prediction.reshape(whole.shape)


def worker(directory):
    cfg, checks, legacy = preflight()
    locked = read(directory/'lock.json')
    assert all(v == locked[k] for k, v in checks.items() if k != 'remaining_cpu_seconds')
    assert 0 < locked['remaining_cpu_seconds']['learning'] <= cfg['cpu_seconds']
    threadpool_limits(limits=cfg['threads'])
    report = {'status': 'RUNNING', 'checks': locked, 'rows': [], 'fit_count': 0,
              'training_updates': 0, 'downloads': 0, 'GPU_jobs': 0}
    legacy.save(directory/'report.json', report)
    for position, row in enumerate(cfg['pairs']):
        started_cpu = time.process_time()
        source, target = image(row['before']), image(row['after'])
        assert source.shape == target.shape
        fit, check = frozen.scoring_folds(*source.shape[:2], row['before']['canonical_sha256'], cfg['seed'], cfg['pixels_per_fold'])
        x, y = source.reshape(-1, 3), target.reshape(-1, 3)
        values, solver = core.fit_operator(x[fit], y[fit], cfg)
        name = f"{position:02d}_{row['group']:02d}_R{row['target']}"
        np.save(directory/f'{name}_absolute_grid.npy', values)
        np.savez(directory/f'{name}_coordinates.npz', fit=fit, check=check)
        report['fit_count'] += 3
        result = {'position': position, 'group': row['group'], 'target': row['target'], 'solver': solver,
                  'fit_samples': len(fit), 'score_samples': len(check),
                  'support': core.support_summary(x[fit], x[check], cfg['dimension']),
                  'score': statistics(x[check], y[check], np.clip(core.render_absolute(x[check], values), 0, 1)),
                  'fit_cpu_seconds': time.process_time()-started_cpu, 'probe_responses': []}
        if solver['solved']:
            native(cfg, row['before'], values, directory/f'{name}_donor.tif', legacy)
            for p, meta in enumerate(cfg['probes']):
                original, generated = native(cfg, meta, values, directory/f'{name}_probe{p}.tif', legacy)
                pixels = np.linspace(0, original.shape[0]*original.shape[1]-1, min(cfg['pixels_per_fold'], original.shape[0]*original.shape[1]), dtype=np.int64)
                ox, py = original.reshape(-1, 3)[pixels], generated.reshape(-1, 3)[pixels]
                result['probe_responses'].append({'probe': meta['key'],
                    'change_deltaE_mean': float(deltaE_ciede2000(rgb2lab(ox), rgb2lab(py)).mean()),
                    'support': core.support_summary(x[fit], ox, cfg['dimension']),
                    'photographic_admission': 'NOT_ASSESSED; consumed inputs, technical throughput only'})
        report['rows'].append(result)
        report['worker_pid'], report['worker_cpu_seconds'] = os.getpid(), time.process_time()
        legacy.save(directory/'report.json', report)
        print(name, 'SOLVED' if solver['solved'] else 'NUMERICAL_INCOMPLETE', result['fit_cpu_seconds'], flush=True)
    report['status'] = 'COMPLETE_PHASE' if all(r['solver']['solved'] for r in report['rows']) else 'NUMERICAL_INCOMPLETE'
    report['claim'] = 'Bounded operator implementation/throughput evidence only; no corpus-family coverage, comfort, or learned generalization claim.'
    report['artifacts'] = [{'path': str(p.relative_to(directory)), 'sha256': frozen.digest(p)} for p in sorted(directory.iterdir()) if p.is_file() and p.name not in ('report.json', 'report.pending', 'worker.log')]
    report['worker_cpu_seconds'] = time.process_time()
    legacy.save(directory/'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['probe'], default='probe')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        config, state, legacy = preflight()
        if args.run:
            runtime = {**config, 'wall_seconds_per_stage': config['wall_seconds_per_stage']-(time.monotonic()-STARTED)}
            state['remaining_cpu_seconds']['learning'] -= time.process_time()
            assert runtime['wall_seconds_per_stage'] > 0 and state['remaining_cpu_seconds']['learning'] > 0
            legacy.__file__, legacy.CONFIG = __file__, CONFIG
            legacy.category = lambda _: 'learning'
            code = legacy.launch('probe', runtime, state)
            path = ROOT/config['output']/'probe/report.json'
            report = read(path)
            report['accounted_cpu_seconds'] += time.process_time()
            report['supervisor']['wall_seconds'] = time.monotonic()-STARTED
            if report['accounted_cpu_seconds'] > config['cpu_seconds'] or report['supervisor']['wall_seconds'] > config['wall_seconds_per_stage']:
                report['status'] = 'RESOURCE_INCOMPLETE'
            if report['supervisor']['reason'] in ('CPU_LIMIT', 'CPU_FINAL_ACCOUNTING_LIMIT', 'WALL_LIMIT', 'RSS_LIMIT'):
                report['status'] = 'RESOURCE_INCOMPLETE'
            report['ledger'] = {'remaining_cpu_seconds': max(0., config['cpu_seconds']-report['accounted_cpu_seconds']),
                'remaining_wall_seconds': max(0., config['wall_seconds_per_stage']-report['supervisor']['wall_seconds']),
                'scope': 'New crossed-study feasibility cost; does not reset or overwrite the old distillation ledger.'}
            legacy.save(path, report)
            raise SystemExit(0 if not code and report['status'] == 'COMPLETE_PHASE' else 1)
        print(json.dumps(state, indent=2))

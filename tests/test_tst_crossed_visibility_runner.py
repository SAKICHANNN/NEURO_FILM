from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
spec = importlib.util.spec_from_file_location('visibility_runner', ROOT / 'scripts/run_tst_crossed_visibility.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def identity_grid():
    axis = np.linspace(0, 1, 7)
    return np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(343, 3)


def synthetic_s2(tmp_path):
    folder = tmp_path / 's2'
    candidates = [{'operator_key': str(i), 'split': 'fit', 'before_canonical_sha256': 'canonical'} for i in range(2)]
    config = {'s2_output': str(folder), 's2_config': str(tmp_path / 'config.json'), 's2_entry': str(tmp_path / 'entry.py'), 'expected_candidates': 2}
    runner.save(tmp_path / 'config.json', {'synthetic': True})
    (tmp_path / 'entry.py').write_text('synthetic')
    queue = {'rows': candidates}
    identity = {'config_sha256': runner.digest(tmp_path / 'config.json'), 'entry_sha256': runner.digest(tmp_path / 'entry.py'), 'queue_sha256': runner.object_hash(queue)}
    runner.save(folder / 'queue_lock.json', {'identity': identity, **queue})
    statuses = [runner.SOLVED, 'NUMERICAL_FAILURE_NOT_ADMITTED']
    for candidate, status in zip(candidates, statuses):
        directory = folder / 'candidate' / candidate['operator_key']
        directory.mkdir(parents=True)
        np.save(directory / 'absolute_grid.npy', identity_grid())
        artifact = {'path': 'absolute_grid.npy', 'sha256': runner.digest(directory / 'absolute_grid.npy'), 'bytes': (directory / 'absolute_grid.npy').stat().st_size}
        runner.save(directory / 'record.json', {'identity': identity, 'candidate': candidate, 'status': status, 'artifacts': [artifact],
                    'solver': {'solved': status == runner.SOLVED}, 'training_admitted': False, 'canonical_before': 'canonical'})
    accounting = {'live': False, 'exit_code': 0, 'failure': None, 'aggregate_cpu_seconds': 1., 'active_wall_seconds': 2.}
    runner.save(folder / 'attempts/0001/accounting.json', accounting)
    runner.save(folder / 'report.json', {'status': 'COMPLETE_PROVISIONAL_NOT_ADMISSION', 'identity': identity, 'processed': 2,
               'counts': dict(Counter(statuses)), 'counts_by_split': {'fit': dict(Counter(statuses))}, 'last_attempt_accounting': accounting})
    return config


def test_actual_metadata_preflight_from_explicit_cwd():
    result = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'scripts/run_tst_crossed_visibility.py')],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['status'] in ('WAIT_S2_COMPLETE', 'READY') and report['photo_reads'] == 0


def test_running_s2_never_reads_partial_records(tmp_path, monkeypatch):
    (tmp_path / 'running.lock').write_text('live')
    monkeypatch.setattr(runner, 'read', lambda path: pytest.fail('partial receipt or record read'))
    assert runner.completed_s2({'s2_output': str(tmp_path)}) is None


def test_completed_s2_binds_entire_census_and_numerical_failure(tmp_path):
    config = synthetic_s2(tmp_path)
    receipt = runner.completed_s2(config)
    assert len(receipt['rows']) == 2
    assert receipt['rows'][0]['grid'] is not None and receipt['rows'][1]['grid'] is None
    (tmp_path / 's2/candidate/1/absolute_grid.npy').write_bytes(b'changed failure artifact')
    with pytest.raises(runner.s2.IdentityFailure, match='ARTIFACT'):
        runner.completed_s2(config)


def test_incomplete_or_changed_report_rejected(tmp_path):
    config = synthetic_s2(tmp_path)
    path = tmp_path / 's2/report.json'
    report = runner.read(path)
    report['processed'] = 1
    runner.save(path, report)
    with pytest.raises(runner.s2.IdentityFailure, match='INCOMPLETE'):
        runner.completed_s2(config)


def test_measure_candidate_full_native_and_cross_partition_rejection(tmp_path):
    config = synthetic_s2(tmp_path)
    item = runner.completed_s2(config)['rows'][0]
    probes = [runner.core.VisibilityProbe(str(i), 'fit', np.full((5, 7, 3), 30000, dtype=np.uint16)) for i in range(4)]
    result = runner.measure_candidate(item, probes, bound_tol=1e-12)
    assert result['strength'] == 1.0 and result['uncomputed_count'] == 2
    assert [r['pixel_count'] for r in result['probes'][:2]] == [35, 35]
    assert result['status'] == 'PROVISIONAL_OPERATOR_NOT_VISIBLE_PAIR_UNREVIEWED'
    with pytest.raises(ValueError, match='partition'):
        runner.measure_candidate({**item, 'split': 'monitor'}, probes, bound_tol=1e-12)


def test_worker_decodes_once_per_partition_and_preserves_non_solved(tmp_path, monkeypatch):
    config = synthetic_s2(tmp_path)
    receipt = runner.completed_s2(config)
    items = [receipt['rows'][0], {**receipt['rows'][0], 'operator_key': 'extra'}, receipt['rows'][1]]
    inputs = {'rows': items, 'probes': [{'split': 'fit'}]}
    out = tmp_path / 's3'
    attempt = out / 'attempts/0001'
    identity = {'synthetic': True}
    runner.save(attempt / 'identity.json', identity)
    monkeypatch.setattr(runner, 'preflight', lambda *args: ({'threads': 2, 'output': str(out), 'bound_tol': 1e-12}, inputs, {'status': 'READY', 'identity': identity}))
    calls = []
    def load(*args):
        calls.append('fit')
        return [runner.core.VisibilityProbe(str(i), 'fit', np.full((2, 3, 3), 30000, dtype=np.uint16)) for i in range(4)]
    monkeypatch.setattr(runner, 'load_probes', load)
    runner.worker(tmp_path / 'config.json', attempt)
    report = runner.read(attempt / 'worker_report.json')
    assert report['processed'] == 3 and calls == ['fit']
    excluded = runner.read(out / 'candidate/1/record.json')
    assert excluded['status'] == 'NOT_ELIGIBLE_S2_OUTCOME_RETAINED' and not excluded['training_admitted']
    runner.worker(tmp_path / 'config.json', attempt)
    assert calls == ['fit']


def test_worker_fatal_probe_failure_stops_and_persists(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempts/0001'
    runner.save(attempt / 'identity.json', {})
    inputs = {'rows': [{'operator_key': 'first', 'split': 'fit', 's2_status': runner.SOLVED}, {'operator_key': 'next', 'split': 'fit', 's2_status': runner.SOLVED}], 'probes': []}
    monkeypatch.setattr(runner, 'preflight', lambda *args: ({'threads': 2, 'output': str(tmp_path)}, inputs, {'status': 'READY', 'identity': {}}))
    def fail(*args):
        raise ValueError('synthetic decode integrity failure')
    monkeypatch.setattr(runner, 'load_probes', fail)
    with pytest.raises(runner.s2.IdentityFailure):
        runner.worker(tmp_path / 'config.json', attempt)
    record = runner.read(tmp_path / 'candidate/first/record.json')
    assert record['status'] == 'FATAL_IDENTITY_FAILURE' and not (tmp_path / 'candidate/next').exists()
    with pytest.raises(runner.s2.IdentityFailure, match='PRIOR_FATAL'):
        runner.reusable(tmp_path / 'candidate/first/record.json', {}, inputs['rows'][0])


def test_unaccounted_prior_attempt_blocks(tmp_path):
    (tmp_path / '0001').mkdir()
    with pytest.raises(runner.s2.IdentityFailure, match='UNACCOUNTED'):
        runner.attempts_accounting(tmp_path)
    runner.save(tmp_path / '0001/accounting.json', {'live': True})
    with pytest.raises(runner.s2.IdentityFailure, match='LIVE'):
        runner.attempts_accounting(tmp_path)


def test_actual_cpu_limit_records_failure_and_peak(tmp_path, monkeypatch):
    config = {'output': str(tmp_path), 'cpu_seconds': .8, 'wall_seconds': 20, 'study_cpu_seconds': 100, 'study_wall_seconds': 100, 'rss_bytes': 2147483648}
    checks = {'status': 'READY', 'identity': {}, 'accounting': {'stage_cpu': 0, 'stage_wall': 0, 'study_cpu': 0, 'study_wall': 0}}
    monkeypatch.setattr(runner, 'preflight', lambda path: (config, {}, checks))
    original = subprocess.Popen
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda args, **kwargs: original([sys.executable, '-c', 'while True: pass'], **kwargs))
    assert runner.launch(tmp_path / 'config.json') == 1
    record = runner.read(tmp_path / 'attempts/0001/accounting.json')
    assert record['failure'] == 'CPU_LIMIT' and record['aggregate_cpu_seconds'] >= .8
    assert record['peak_tree_rss_bytes'] > 0 and record['owned_processes'] and not record['live']


def test_cumulative_ledger_retains_prior_s2_and_failed_s3_cost(tmp_path):
    for name, cpu, wall in [('s0', 2., 3.), ('s1', 4., 5.), ('s2', 6., 7.), ('s3', 8., 9.)]:
        runner.save(tmp_path / name / 'attempts/0001/accounting.json', {'aggregate_cpu_seconds': cpu, 'active_wall_seconds': wall, 'live': False, 'failure': 'CPU_LIMIT' if name == 's3' else None})
    runner.save(tmp_path / 'sources.json', {'identity': {'download_accounting': {'cpu_seconds': 10., 'wall_seconds': 11.}}})
    config = {'prior_attempt_directories': [str(tmp_path / name / 'attempts') for name in ('s0', 's1', 's2')],
              'output': str(tmp_path / 's3'), 'source_report': str(tmp_path / 'sources.json'),
              'prior_reserve_cpu_seconds': 100., 'prior_reserve_wall_seconds': 200.}
    result = runner.ledger(config)
    assert result['stage_cpu'] == 8. and result['stage_wall'] == 9.
    assert result['study_cpu'] == 130. and result['study_wall'] == 235.
    assert len(result['prior_attempt_pins']) == 3


def test_s2_stale_success_report_cannot_hide_failed_latest_attempt(tmp_path):
    config = synthetic_s2(tmp_path)
    runner.save(tmp_path / 's2/attempts/0002/accounting.json', {'live': False, 'failure': 'CPU_LIMIT', 'exit_code': 1})
    with pytest.raises(runner.s2.IdentityFailure, match='LAST_ATTEMPT'):
        runner.completed_s2(config)


def test_frozen_solver_bound_tolerance_preserves_grid_and_rejects_beyond(tmp_path):
    config = synthetic_s2(tmp_path)
    item = runner.completed_s2(config)['rows'][0]
    path = Path(item['grid']['path'])
    probes = [runner.core.VisibilityProbe(str(i), 'fit', np.zeros((1, 1, 3), np.uint16)) for i in range(4)]
    values = identity_grid()
    values[0, 0], values[-1, -1] = -5e-13, 1 + 5e-13
    np.save(path, values)
    item['grid']['sha256'] = runner.digest(path)
    result = runner.measure_candidate(item, probes, bound_tol=1e-12)
    assert result['strength'] == 1.0
    np.testing.assert_array_equal(np.load(path), values)
    values[0, 0] = -2e-12
    np.save(path, values)
    item['grid']['sha256'] = runner.digest(path)
    with pytest.raises(runner.s2.IdentityFailure, match='BOUNDED'):
        runner.measure_candidate(item, probes, bound_tol=1e-12)

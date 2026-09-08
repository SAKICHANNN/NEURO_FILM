import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
spec = importlib.util.spec_from_file_location('crossed_operators', ROOT / 'scripts/run_tst_crossed_operators.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_final_metadata_preflight_from_actual_repository():
    result = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'scripts/run_tst_crossed_operators.py')],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['status'] == 'READY' and report['photo_reads'] == 0
    assert report['candidates'] > 3400
    assert report['accounting']['study_cpu'] > 4000


def test_all_aliases_reserved_and_stale_aliases_rejected():
    rows = [{'component': str(i), 'representative': str(i), 'all_component_aliases': [str(i)]} for i in range(12)]
    rows[0]['all_component_aliases'].append('alias')
    components = {'owner_to_component': {**{str(i): str(i) for i in range(13)}, 'alias': '0'}, 'quarantined_owners': []}
    candidates = [{'before': name, 'after': 'R', 'split': 'fit', 'before_component': components['owner_to_component'][name], 'operator_key': name} for name in ('alias', '12')]
    probes = {'status': 'FROZEN_12_SOURCE_ADMISSION_PROBES_ROOT_ACCEPTED', 'rows': rows}
    queue = runner.freeze_queue(candidates, probes, components)
    assert [r['before'] for r in queue['rows']] == ['12']
    assert [r['before'] for r in queue['excluded_probe_donors']] == ['alias']
    rows[0]['all_component_aliases'].remove('alias')
    with pytest.raises(ValueError, match='ALIAS'):
        runner.freeze_queue(candidates, probes, components)


def test_synthetic_identity_fit_persists_exact_coordinates_and_no_photos(tmp_path):
    config = runner.read(ROOT / 'configs/tst_crossed_operators_v1.json')
    runner.threadpool_limits(2)
    image = np.random.default_rng(42).integers(0, 65536, (128, 128, 3), dtype=np.uint16)
    canonical = runner.corpus.canonical_hash(image)
    record = runner.fit_arrays(image, image.copy(), canonical, config, tmp_path)
    assert record['status'] == 'PROVISIONAL_NUMERICAL_SOLVE_ONLY'
    assert record['training_admitted'] is False
    assert record['score']['mean_deltaE2000'] < 1e-6
    coordinates = np.load(tmp_path / 'coordinates.npz')
    expected = runner.scoring_folds(128, 128, canonical, config['seed'], 8192)
    np.testing.assert_array_equal(coordinates['fit_flat_indices'], expected[0])
    np.testing.assert_array_equal(coordinates['score_flat_indices'], expected[1])
    assert set(p.suffix for p in tmp_path.iterdir()) == {'.npy', '.npz'}
    record.update(identity={'test': 1}, candidate={'test': 2})
    runner.save(tmp_path / 'record.json', record)
    assert runner.reusable(tmp_path, {'test': 1}, {'test': 2}) == record
    with pytest.raises(ValueError, match='IDENTITY'):
        runner.reusable(tmp_path, {'test': 3}, {'test': 2})
    (tmp_path / 'absolute_grid.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='ARTIFACT'):
        runner.reusable(tmp_path, {'test': 1}, {'test': 2})


def test_dimension_mismatch_never_calls_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.core, 'fit_operator', lambda *args: pytest.fail('solver must not run'))
    record = runner.fit_arrays(np.zeros((128, 128, 3), np.uint16), np.zeros((127, 128, 3), np.uint16), 'hash', {}, tmp_path)
    assert record['status'] == 'DIMENSION_MISMATCH_NO_PIXEL_PAIRING'
    assert not list(tmp_path.iterdir())


def test_numerical_failure_retained_without_threshold_relaxation(tmp_path, monkeypatch):
    config = runner.read(ROOT / 'configs/tst_crossed_operators_v1.json')
    called = []
    def failed(x, y, cfg):
        called.append(cfg)
        return runner.core.identity_grid(7), {'solved': False, 'channels': []}
    monkeypatch.setattr(runner.core, 'fit_operator', failed)
    image = np.zeros((128, 128, 3), np.uint16)
    record = runner.fit_arrays(image, image, 'synthetic', config, tmp_path)
    assert record['status'] == 'NUMERICAL_FAILURE_NOT_ADMITTED'
    assert called == [config] and len(record['artifacts']) == 2


def test_unaccounted_attempt_blocks_resume(tmp_path):
    (tmp_path / 'attempts/0001').mkdir(parents=True)
    with pytest.raises(ValueError, match='UNACCOUNTED'):
        runner.ledger({'prior_accounting_globs': []}, tmp_path)
    runner.save(tmp_path / 'attempts/0001/accounting.json', {'live': True})
    with pytest.raises(ValueError, match='UNACCOUNTED_LIVE'):
        runner.ledger({'prior_accounting_globs': []}, tmp_path)


def test_interrupted_artifacts_preserved_and_only_one_exact_replay(tmp_path):
    runner.begin_candidate(tmp_path, {'same': True}, '0001')
    (tmp_path / 'absolute_grid.npy').write_bytes(b'partial')
    with pytest.raises(ValueError, match='IDENTITY'):
        runner.begin_candidate(tmp_path, {'changed': True}, '0002')
    runner.begin_candidate(tmp_path, {'same': True}, '0002')
    assert (tmp_path / 'interrupted_0001/absolute_grid.npy').read_bytes() == b'partial'
    with pytest.raises(ValueError, match='REPLAY_LIMIT'):
        runner.begin_candidate(tmp_path, {'same': True}, '0003')


def test_supervisor_records_actual_process_tree_cpu_failure(tmp_path, monkeypatch):
    config = {'output': str(tmp_path), 'cpu_seconds': .8, 'wall_seconds': 20,
              'study_cpu_seconds': 100, 'study_wall_seconds': 100, 'rss_bytes': 2147483648}
    checks = {'identity': {'synthetic': True}, 'accounting': {'stage_cpu': 0, 'stage_wall': 0, 'study_cpu': 0, 'study_wall': 0}}
    monkeypatch.setattr(runner, 'preflight', lambda path: (config, {'rows': []}, checks))
    original = subprocess.Popen
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda args, **kwargs: original([sys.executable, '-c', 'while True: pass'], **kwargs))
    assert runner.launch(tmp_path / 'synthetic_config.json') == 1
    record = runner.read(tmp_path / 'attempts/0001/accounting.json')
    assert record['failure'] == 'CPU_LIMIT'
    assert record['aggregate_cpu_seconds'] >= .8
    assert not record['live'] and not (tmp_path / 'running.lock').exists()
    assert record['peak_tree_rss_bytes'] > 0 and record['owned_processes']


def test_integrity_failures_are_fatal_and_not_reusable(tmp_path):
    with pytest.raises(runner.IdentityFailure):
        runner.verify_encoded(tmp_path / 'missing.png', {'bytes': 1, 'sha256': 'x'})
    with pytest.raises(runner.IdentityFailure):
        runner.verified_header(tmp_path / 'missing.json', 'x')
    runner.save(tmp_path / 'record.json', {'identity': {}, 'candidate': {}, 'status': 'FATAL_IDENTITY_FAILURE', 'artifacts': []})
    with pytest.raises(runner.IdentityFailure, match='PRIOR_FATAL'):
        runner.reusable(tmp_path, {}, {})


def test_worker_saves_identity_failure_and_aborts_before_next_candidate(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempts/0001'
    runner.save(attempt / 'identity.json', {'test': True})
    runner.save(tmp_path / 'acquisition.json', {'files': [{'path': 'missing.png', 'bytes': 1, 'sha256': 'x'}]})
    config = {'threads': 2, 'output': str(tmp_path), 'originals': str(tmp_path), 'acquisition_config': str(tmp_path / 'acquisition.json')}
    row = {'operator_key': 'first', 'before': 'missing.png', 'after': 'missing.png', 'split': 'fit'}
    monkeypatch.setattr(runner, 'preflight', lambda *args: (config, {'rows': [row, {**row, 'operator_key': 'second'}]}, {'identity': {'test': True}}))
    monkeypatch.setattr(runner.corpus, 'decode', lambda *args: pytest.fail('no decoding after integrity failure'))
    with pytest.raises(runner.IdentityFailure):
        runner.worker(tmp_path / 'config.json', attempt)
    assert runner.read(tmp_path / 'candidate/first/record.json')['status'] == 'FATAL_IDENTITY_FAILURE'
    assert not (tmp_path / 'candidate/second').exists()

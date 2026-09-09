import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
spec = importlib.util.spec_from_file_location('photo_v3_test', ROOT/'scripts/prepare_tst_crossed_photo_review_v3.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(key, component, split='fit'):
    return {'operator_key': key, 'split': split, 'candidate': {'before_component': component}}


def test_component_selection_order_invariant_and_excludes_all_variants():
    rows = [row('old', 'a'), row('alias', 'a'), row('b1', 'b'), row('b2', 'b'), row('c', 'c'), row('m', 'm', 'monitor')]
    a = m.select_rows(rows, ['old'], {'fit': 2, 'monitor': 1}, m.SEED)
    assert a == m.select_rows(rows[::-1], ['old'], {'fit': 2, 'monitor': 1}, m.SEED)
    assert {r['candidate']['before_component'] for r in a} == {'b', 'c', 'm'}
    assert len(a) == 3
    with pytest.raises(m.v2.old.s2.IdentityFailure):
        m.select_rows(rows, ['old'], {'fit': 3}, m.SEED)


def test_cross_split_component_rejected():
    with pytest.raises(m.v2.old.s2.IdentityFailure):
        m.select_rows([row('f', 'a'), row('m', 'a', 'monitor')], [], {'fit': 1, 'monitor': 1}, m.SEED)


def test_compact_json_hash_independent_known_vector():
    import hashlib
    value = [m.SEED, 'fit', 'abc']
    expected = hashlib.sha256(('['+'"'+m.SEED+'","fit","abc"]').encode()).hexdigest()
    assert m.rank(value) == expected


def test_adjudication_preserves_raw_and_rejects_duplicates():
    lock = {'imported': {'old': {'status': 'FAIL'}}}
    terminal = {'operator_key': 'new', 'status': 'PASS_CURATOR_ONLY'}
    override = {'operator_key': 'new', 'status': 'UNCERTAIN_NOT_ADMITTED', 'training_admitted': False}
    assert m.effective_history(lock, [terminal], [override]) == {'old': 'FAIL', 'new': 'UNCERTAIN_NOT_ADMITTED'}
    assert terminal['status'] == 'PASS_CURATOR_ONLY'
    with pytest.raises(m.v2.old.s2.IdentityFailure):
        m.effective_history(lock, [terminal], [override, override])


def test_endpoint_no_automatic_next_and_no_history_as_new(monkeypatch, tmp_path):
    monkeypatch.setattr(m.v2, 'ROOT', tmp_path)
    (tmp_path/'terminal').mkdir()
    terminal = {'identity': {'x': 1}, 'next_index': 96}
    (tmp_path/'terminal'/'last.json').write_text(json.dumps(terminal))
    (tmp_path/'cursor.json').write_text(json.dumps({'index': 96, 'previous_terminal_sha256': m.digest(tmp_path/'terminal'/'last.json')}))
    inputs = {'queue': [str(i) for i in range(95)]+['last']}
    state, item, probe = m.cursor({'output': '.'}, inputs, {'x': 1})
    assert state['status'] == 'ALL_CANDIDATES_RECORDED_FINAL_AUDIT_REQUIRED'
    assert state['terminal_count'] == 96 and state['historical_terminal_count'] == 25
    assert item is probe is None


def test_cumulative_costs_no_v2_double_count(monkeypatch, tmp_path):
    monkeypatch.setattr(m.v2, 'ROOT', tmp_path)
    monkeypatch.setattr(m.v2.old.s3, 'ledger', lambda config: {'study_cpu': 1000., 'study_wall': 900.})
    monkeypatch.setattr(m.v2.old.s3, 'attempts_accounting', lambda path: [{'aggregate_cpu_seconds': 67.528125, 'active_wall_seconds': 58.422}])
    cfg = {'output': 'new', 'legacy_output': 'old', 'external_photo_cpu_seconds': 68.578125+82.828125+10,
           'external_photo_wall_reserve_seconds': 150+81.064+20, 'legacy_candidate_bytes': 71762258+15097023}
    result = m.v2.costs(cfg)
    assert result['photo_cpu'] == pytest.approx(228.934375)
    assert result['study_cpu'] == pytest.approx(1161.40625)
    assert result['photo_wall'] == pytest.approx(309.486)
    assert result['retained_bytes'] == 86859281


def test_reuses_exact_renderer_review_and_worker_entry():
    panel, validation, worker, supervisor = m.v2.stage_panel, m.v2.validate_observation, m.v2.worker, m.v2.launch
    m.install()
    assert m.v2.stage_panel is panel and m.v2.validate_observation is validation
    assert m.v2.worker is worker and m.v2.launch is supervisor
    assert m.v2.__file__ == m.__file__
    assert m.v2.frozen_inputs is m.frozen_inputs


def test_inherited_launch_worker_and_final_audit_after_install(monkeypatch, tmp_path):
    from types import SimpleNamespace
    m.install()
    monkeypatch.setattr(m.v2, 'ROOT', tmp_path)
    cfg = {'output': 'out'}
    identity = {'synthetic': True}
    state = {'status': 'ALL_CANDIDATES_RECORDED_FINAL_AUDIT_REQUIRED'}
    checks = {'identity': identity, 'state': state, 'accounting': {'checkpoint': {'attempts': 0, 'cpu': 0., 'wall': 0., 'bytes': 0}}}
    monkeypatch.setattr(m.v2, 'preflight', lambda path, active=None: (cfg, {'population_keys': ['k']}, checks, None, None))
    calls = []
    monkeypatch.setattr(m, 'prefix_audit', lambda config: calls.append('prefix'))
    monkeypatch.setattr(m, 'BASE_AUDIT', lambda config, inputs, ident, active: calls.append('finite_audit') or {'training_admitted': False})

    class Process:
        pid = 123

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            assert argv[1] == m.__file__ and argv[2] == '--config' and argv[4] == '--worker'
            m.v2.worker(Path(argv[3]), Path(argv[5]))

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(m.v2.subprocess, 'Popen', Process)
    monkeypatch.setattr(m.v2.psutil, 'Process', lambda *args: SimpleNamespace())
    monkeypatch.setattr(m.v2.old.s2.corpus, 'sample_tree', lambda *args: (0., 0))
    monkeypatch.setattr(m.v2.old.s2.corpus, 'stop_tree', lambda *args: None)
    assert m.v2.launch(tmp_path/'config.json', audit=True) == 0
    report = json.loads((tmp_path/'out/attempts/0001/worker_report.json').read_text())
    assert calls[1:] == ['prefix', 'finite_audit']
    assert report['status'] == 'FINITE_96_HISTORY_VERIFIED_NOT_TRAINING'
    assert report['full_population_audited'] is False and report['automatic_next_tranche'] is False

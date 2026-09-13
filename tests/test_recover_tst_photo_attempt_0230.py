import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
SPEC = importlib.util.spec_from_file_location('photo_recovery_test', ROOT / 'scripts/recover_tst_photo_attempt_0230.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
Failure = m.v2.old.s2.IdentityFailure


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return m.digest(path)


@pytest.fixture
def retained_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'OUTPUT', tmp_path)
    directory = tmp_path / 'attempts/0230'
    value = {'aggregate_cpu_seconds': .340625, 'active_wall_seconds': .172,
             'peak_tree_rss_bytes': 0, 'live': False, 'exit_code': -1,
             'failure': 'WORKER_EXCEPTION_OR_INTERRUPTION',
             'previous_accounting_sha256': 'previous', 'added_candidate_bytes': 0}
    sha = put(directory / 'accounting.json', value)
    put(directory / 'failure.json', value)
    identity = put(directory / 'identity.json', {'identity': 'frozen'})
    monkeypatch.setattr(m, 'ATTEMPT_SHA', sha)
    monkeypatch.setattr(m, 'IDENTITY_SHA', identity)
    return directory, value


def test_exception_keeps_raw_failure_and_exact_cost(retained_failure):
    directory, value = retained_failure
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert m.accepted_failure(directory / 'accounting.json', value)
    assert m.verify_failed_attempt(directory)['failure'] == 'WORKER_EXCEPTION_OR_INTERRUPTION'
    assert m.verify_failed_attempt(directory)['aggregate_cpu_seconds'] == .340625
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize('extra', ['request.json', 'worker.log', 'worker_report.json', 'worker_final.json'])
def test_worker_or_partial_dispatch_evidence_refuses_recovery(retained_failure, extra):
    directory, value = retained_failure
    (directory / extra).write_text('{}')
    with pytest.raises(Failure, match='RECOVERY_WORKER_OR_UNKNOWN_FILE'):
        m.accepted_failure(directory / 'accounting.json', value)


@pytest.mark.parametrize('field,new_value', [('added_candidate_bytes', 1), ('exit_code', 0), ('live', True), ('peak_tree_rss_bytes', 4)])
def test_even_rehashed_non_pre_worker_failure_rejected(retained_failure, monkeypatch, field, new_value):
    directory, value = retained_failure
    value[field] = new_value
    sha = put(directory / 'accounting.json', value)
    put(directory / 'failure.json', value)
    monkeypatch.setattr(m, 'ATTEMPT_SHA', sha)
    with pytest.raises(Failure, match='RECOVERY_NOT_PRE_WORKER_FAILURE'):
        m.verify_failed_attempt(directory)


def test_different_attempt_and_changed_failure_rejected(retained_failure):
    directory, value = retained_failure
    with pytest.raises(Failure, match='UNADJUDICATED_FAILURE'):
        m.accepted_failure(directory.parent / '0231/accounting.json', value)
    changed = copy.deepcopy(value)
    changed['aggregate_cpu_seconds'] = 0
    with pytest.raises(Failure, match='RECOVERY_ACCOUNTING_VALUE_CHANGED'):
        m.accepted_failure(directory / 'accounting.json', changed)
    put(directory / 'failure.json', changed)
    with pytest.raises(Failure, match='RECOVERY_FAILURE_CHANGED'):
        m.verify_failed_attempt(directory)


def test_accounting_chain_and_total_checks_independent_of_exception(tmp_path, monkeypatch):
    value = {'previous_accounting_sha256': None, 'failure': 'retained',
             'aggregate_cpu_seconds': .34, 'active_wall_seconds': .17, 'added_candidate_bytes': 0}
    sha = put(tmp_path / 'attempts/0001/accounting.json', value)
    monkeypatch.setattr(m, 'accepted_failure', lambda path, record: True)
    checkpoint = {'attempts': 1, 'cpu': .34, 'wall': .17, 'bytes': 0, 'last_sha256': sha}
    m.verify_accounting(tmp_path, checkpoint)
    with pytest.raises(Failure, match='ACCOUNTING_TOTAL_CHANGED'):
        m.verify_accounting(tmp_path, {**checkpoint, 'cpu': 0})
    with pytest.raises(Failure, match='ACCOUNTING_HEAD_CHANGED'):
        m.verify_accounting(tmp_path, {**checkpoint, 'last_sha256': 'different'})
    value['previous_accounting_sha256'] = 'forged'
    checkpoint['last_sha256'] = put(tmp_path / 'attempts/0001/accounting.json', value)
    with pytest.raises(Failure, match='ACCOUNTING_CHAIN_FAILURE'):
        m.verify_accounting(tmp_path, checkpoint)


def test_routing_only_exact_frozen_worker_command(monkeypatch, tmp_path):
    calls = []
    original_module = m.subprocess
    monkeypatch.setattr(m.subprocess, 'Popen', lambda argv, **kwargs: calls.append((argv, kwargs)))
    argv = ['python', str(m.ENTRY), '--config', str(m.CONFIG), '--worker', str(tmp_path / '0231')]
    m.route_worker(argv, stdout='target')
    assert calls == [(['python', str(Path(m.__file__)), *argv[2:]], {'stdout': 'target'})]
    assert m.subprocess is original_module
    for pos, replacement in [(1, 'different.py'), (2, '--run'), (3, 'other.json'), (4, '--audit')]:
        bad = list(argv)
        bad[pos] = replacement
        with pytest.raises(Failure, match='RECOVERY_UNEXPECTED_WORKER_COMMAND'):
            m.route_worker(bad)
    assert len(calls) == 1


def test_changed_receipt_or_adapter_identity_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'RECEIPT', tmp_path / 'receipt.json')
    put(m.RECEIPT, {'adapter_sha256': 'old'})
    monkeypatch.setattr(m, 'proof', lambda: {'adapter_sha256': 'new'})
    with pytest.raises(Failure, match='RECOVERY_RECEIPT_CHANGED'):
        m.install()


def test_installed_proxy_supports_launch_stderr_and_keeps_frozen_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'RECEIPT', tmp_path / 'receipt.json')
    put(m.RECEIPT, {'adapter_sha256': 'fixed'})
    monkeypatch.setattr(m, 'proof', lambda: {'adapter_sha256': 'fixed'})
    for attr in ('costs', 'audit_history', 'subprocess', '__file__', 'frozen_inputs', 'cursor'):
        monkeypatch.setattr(m.v2, attr, getattr(m.v2, attr))
    calls = []
    monkeypatch.setattr(m.subprocess, 'Popen', lambda argv, **kwargs: calls.append((argv, kwargs)))
    m.install()
    with (tmp_path / 'worker.log').open('w') as log:
        m.v2.subprocess.Popen(['python', str(Path(m.v2.__file__)), '--config', str(m.CONFIG), '--worker', str(tmp_path / '0231')], stdout=log, stderr=m.v2.subprocess.STDOUT)
    assert calls[0][0][1] == str(Path(m.__file__))
    assert calls[0][1]['stderr'] == m.subprocess.STDOUT
    assert Path(m.v2.__file__) == m.ENTRY


def test_frozen_sources_unchanged_and_local_exact_failed_attempt_valid():
    for path, expected in m.PINS.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    value = m.verify_failed_attempt(m.OUTPUT / 'attempts/0230')
    assert value['aggregate_cpu_seconds'] == .340625
    assert value['added_candidate_bytes'] == 0


def test_final_audit_keeps_failure_disclosure_and_rejects_review_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'RECEIPT', tmp_path / 'receipt.json')
    monkeypatch.setattr(m.v3, 'prefix_audit', lambda config: None)
    put(m.RECEIPT, {'adjudicated': '0230'})
    put(tmp_path / 'probes.json', {'rows': [{'component': 'p', 'split': 'fit', 'category': 'portrait_face'}]})
    directory = m.OUTPUT / 'candidate/key/portrait_face'
    directory.mkdir(parents=True)
    (directory / 'sheet.png').write_bytes(b'synthetic fixture only')
    artifact = {'path': 'sheet.png', 'bytes': 22, 'sha256': m.digest(directory / 'sheet.png')}
    record = {'identity': {'frozen': True}, 'operator_key': 'key', 'stage': 'portrait_face', 'probe_id': 'p', 'artifact': artifact}
    record_sha = put(directory / 'record.json', record)
    review = {'operator_key': 'key', 'stage': 'portrait_face', 'probe_id': 'p',
              'stage_record_sha256': record_sha, 'reviewed_artifact_hashes': [artifact['sha256']],
              'actual_viewed': True, 'reviewer_kind': 'synthetic_test_fixture',
              'reason': 'fixture', 'decision': 'FAIL', 'training_admitted': False,
              'comfortable': False, 'visibly_changed_comfortably_rich': False,
              'objectionable_new_content_or_detail_loss': False}
    review_sha = put(directory / 'review.json', review)
    put(m.OUTPUT / 'terminal/key.json', {'identity': record['identity'], 'operator_key': 'key',
        'next_index': 1, 'previous_terminal_sha256': None, 'status': 'FAIL',
        'receipts': [{'stage': 'portrait_face', 'record_sha256': record_sha, 'review_sha256': review_sha}],
        'unreviewed_stages': list(m.v2.STAGES[1:])})
    retained = sum(p.stat().st_size for p in directory.iterdir())
    monkeypatch.setattr(m, 'costs', lambda config, active=None: {'checkpoint': {'bytes': retained}})
    inputs = {'queue': ['key'], 'population_keys': ['key'], 'imported': {}, 'rows': [{'operator_key': 'key', 'split': 'fit'}]}
    result = m.audit_history({'probe_manifest': 'probes.json'}, inputs, record['identity'])
    assert result['accepted_failed_attempt'] == '0230'
    assert result['failure_cost_included'] is True
    assert result['training_admitted'] is False
    assert 'ADJUDICATED_PRE_WORKER_FAILURE' in result['status']
    review['comfortable'] = True
    put(directory / 'review.json', review)
    with pytest.raises(Failure, match='HISTORICAL_REVIEW_CHANGED'):
        m.audit_history({'probe_manifest': 'probes.json'}, inputs, record['identity'])

import hashlib
import io
from types import SimpleNamespace

import pytest

from src.data.fable_bounded_fetch import fetch_once


class Response:
    status_code = 200
    def __init__(self, body, length):
        self.raw = io.BytesIO(body)
        self.headers = {'Content-Length': str(length)}
    def __enter__(self): return self
    def __exit__(self, *args): pass


def test_success_and_resume_do_not_repeat_request(tmp_path):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        return Response(b'raw', 3)
    session = SimpleNamespace(get=get)
    ledger = {'objects': {}, 'charged_body_bytes': 0}
    entry = {'identity': 'one', 'url': 'https://example.invalid/raw', 'expected_raw_sha256': hashlib.sha256(b'raw').hexdigest()}
    for _ in range(2):
        result = fetch_once(entry, session=session, directory=tmp_path, ledger_path=tmp_path/'ledger.json', ledger=ledger, cap=10)
    assert result['status'] == 'FETCHED' and ledger['charged_body_bytes'] == 3 and len(calls) == 1


def test_partial_failure_retains_reservation_and_never_retries(tmp_path):
    ledger = {'objects': {}, 'charged_body_bytes': 0}
    session = SimpleNamespace(get=lambda *a, **k: Response(b'raw', 8))
    result = fetch_once({'identity': 'one', 'url': 'test'}, session=session, directory=tmp_path, ledger_path=tmp_path/'ledger.json', ledger=ledger, cap=10)
    assert result['status'] == 'FAILED_NO_RETRY' and ledger['charged_body_bytes'] == 3


def test_uncertain_prior_attempt_blocks_new_requests(tmp_path):
    ledger = {'objects': {'old': {'status': 'STARTED'}}, 'charged_body_bytes': 10}
    with pytest.raises(ValueError, match='uncertain'):
        fetch_once({'identity': 'new', 'url': 'test'}, session=None, directory=tmp_path, ledger_path=tmp_path/'ledger.json', ledger=ledger, cap=10)


def test_http_failure_does_not_consume_entire_budget(tmp_path):
    ledger = {'objects': {}, 'charged_body_bytes': 0}
    response = Response(b'error', 5)
    response.status_code = 404
    result = fetch_once({'identity': 'one', 'url': 'test'}, session=SimpleNamespace(get=lambda *a, **k: response), directory=tmp_path, ledger_path=tmp_path/'ledger.json', ledger=ledger, cap=10)
    assert result['status'] == 'FAILED_NO_RETRY' and ledger['charged_body_bytes'] == 0


def test_read_exception_charges_inflight_request(tmp_path):
    ledger = {'objects': {}, 'charged_body_bytes': 0}
    response = Response(b'', 8)
    def fail(n): raise OSError('uncertain partial read')
    response.raw = SimpleNamespace(read=fail)
    result = fetch_once({'identity': 'one', 'url': 'test'}, session=SimpleNamespace(get=lambda *a, **k: response), directory=tmp_path, ledger_path=tmp_path/'ledger.json', ledger=ledger, cap=10)
    assert result['status'] == 'FAILED_NO_RETRY' and ledger['charged_body_bytes'] == 8

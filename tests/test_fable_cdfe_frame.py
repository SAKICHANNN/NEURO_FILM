import hashlib
from types import SimpleNamespace

import pytest

from scripts.fetch_fable_cdfe_frame import local_reuse, validate_resume, FRAME_SHA


def test_local_reuse_is_hash_bound_and_zero_transfer(tmp_path, monkeypatch):
    folder = tmp_path / 'data'
    folder.mkdir()
    path = folder / 'a0001.dng'
    path.write_bytes(b'raw')
    monkeypatch.setattr('scripts.fetch_fable_cdfe_frame.subprocess.run', lambda *a, **k: SimpleNamespace(stdout=str(path)+'\n'))
    plan = [{'identity': 'fivek/a0001', 'url': 'test', 'expected_raw_sha256': hashlib.sha256(b'raw').hexdigest()}]
    matches = local_reuse(tmp_path, plan)
    ledger = {'frame_sha256': FRAME_SHA, 'entries': plan, 'objects': matches, 'charged_body_bytes': 0}
    validate_resume(ledger, plan, 10)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_resume(ledger, plan, 10)


def test_inconsistent_accounting_rejected():
    ledger = {'frame_sha256': FRAME_SHA, 'entries': [], 'objects': {}, 'charged_body_bytes': 1}
    with pytest.raises(ValueError, match='accounting'):
        validate_resume(ledger, [], 10)

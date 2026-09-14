import hashlib
import json

import pytest

from src.training.fable_cache_seal import completed_cache_bindings


def test_completion_does_not_hide_missing_donor_payload(tmp_path):
    ids = [str(j) for j in range(256)]
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'rows': [{'identity': i} for i in ids],
                               'treatment_ids': [str(j) for j in range(32)]}))
    with pytest.raises(FileNotFoundError):
        completed_cache_bindings(tmp_path, plan, locked_ids=ids)
    (tmp_path/'complete.json').write_text(json.dumps({'contract_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
                                                    'donors': 256, 'examples': 8192}))
    with pytest.raises(ValueError, match='donor receipt'):
        completed_cache_bindings(tmp_path, plan, locked_ids=ids)


def test_every_payload_is_verified_and_receipts_remain_ordered(tmp_path):
    ids = [str(j) for j in range(256)]
    ts = [str(j) for j in range(32)]
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'rows': [{'identity': i} for i in ids], 'treatment_ids': ts}))
    contract = hashlib.sha256(plan.read_bytes()).hexdigest()
    (tmp_path/'complete.json').write_text(json.dumps({'contract_sha256': contract, 'donors': 256, 'examples': 8192}))
    for identity in ids:
        stem = hashlib.sha256(identity.encode()).hexdigest()
        payload = identity.encode()
        (tmp_path/(stem+'.npz')).write_bytes(payload)
        (tmp_path/(stem+'.json')).write_text(json.dumps({'identity': identity,
            'contract_sha256': contract, 'cache_sha256': hashlib.sha256(payload).hexdigest(), 'treatments': ts}))
    result = completed_cache_bindings(tmp_path, plan, locked_ids=ids)
    assert list(result['receipt_sha256']) == ids
    assert list(result['cache_sha256']) == ids
    stem = hashlib.sha256(ids[-1].encode()).hexdigest()
    (tmp_path/(stem+'.npz')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        completed_cache_bindings(tmp_path, plan, locked_ids=ids)

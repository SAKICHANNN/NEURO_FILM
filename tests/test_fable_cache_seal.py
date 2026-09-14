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

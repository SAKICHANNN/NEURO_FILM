import json

import pytest

from src.data.fable_cdfe_training_data import load_fitting_cache


def test_partial_cache_and_changed_identity_are_rejected_before_allocation(tmp_path):
    identities = [f'fit/{j}' for j in range(256)]
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'rows': [{'identity': i} for i in identities]}))
    with pytest.raises(FileNotFoundError):
        load_fitting_cache(tmp_path, plan, locked_fitting_ids=identities)
    with pytest.raises(ValueError, match='locked'):
        load_fitting_cache(tmp_path, plan, locked_fitting_ids=identities[:-1]+['held'])
    (tmp_path / 'complete.json').write_text(json.dumps({'donors': 255}))
    with pytest.raises(ValueError, match='certificate'):
        load_fitting_cache(tmp_path, plan, locked_fitting_ids=identities)

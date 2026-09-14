import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.fable_cdfe_cache import load_completed_donor
from src.data.fable_cdfe_preprocessing import fitting_target_normalizer


def load_fitting_cache(directory: Path, plan_path: Path, *, locked_fitting_ids: list[str]) -> dict:
    blob = plan_path.read_bytes()
    contract = hashlib.sha256(blob).hexdigest()
    plan = json.loads(blob)
    identities = [r['identity'] for r in plan['rows']]
    if len(identities) != 256 or len(set(identities)) != 256 or identities != locked_fitting_ids:
        raise ValueError('exact256 locked fitting identities required')
    completion = json.loads((directory / 'complete.json').read_text())
    if completion != {'contract_sha256': contract, 'donors': 256, 'examples': 8192}:
        raise ValueError('complete fitting cache certificate required')
    if len(plan['treatment_ids']) != 32 or len(set(plan['treatment_ids'])) != 32:
        raise ValueError('32 unique fixed treatment identities required')
    inputs = np.empty((256, 32, 3, 128, 128), dtype=np.float32)
    targets = np.empty((256, 4), dtype=np.float64)
    for j, identity in enumerate(identities):
        record = load_completed_donor(directory, identity, contract)
        if record is None or record['treatments'] != plan['treatment_ids']:
            raise ValueError('missing or reordered donor treatments')
        stem = hashlib.sha256(identity.encode()).hexdigest()
        with np.load(directory / (stem + '.npz'), allow_pickle=False) as donor:
            x, y = donor['inputs'], donor['target']
            if (x.shape != (32, 3, 128, 128) or x.dtype != np.float32 or y.shape != (4,)
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError('invalid cached fitting arrays')
            inputs[j], targets[j] = x, y
    normalizer = fitting_target_normalizer(targets, identities,
        expected_identities=locked_fitting_ids, minimum_scale=1e-12)
    standardized = ((targets-normalizer['mean'])/normalizer['scale']).astype(np.float32)
    return {'inputs': inputs.reshape(8192, 3, 128, 128),
            'targets': np.repeat(standardized, 32, axis=0),
            'normalizer': normalizer, 'contract_sha256': contract,
            'identities': identities, 'treatment_ids': plan['treatment_ids']}

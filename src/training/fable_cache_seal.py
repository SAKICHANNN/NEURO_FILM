import hashlib
import json
from pathlib import Path

from src.data.fable_cdfe_cache import load_completed_donor


def completed_cache_bindings(directory: Path, plan_path: Path, *, locked_ids: list[str]) -> dict:
    blob = plan_path.read_bytes()
    plan = json.loads(blob)
    contract = hashlib.sha256(blob).hexdigest()
    identities = [r['identity'] for r in plan['rows']]
    if len(identities) != 256 or len(set(identities)) != 256 or identities != locked_ids:
        raise ValueError('exact locked fitting identities required')
    completion = (directory / 'complete.json').read_bytes()
    if json.loads(completion) != {'contract_sha256': contract, 'donors': 256, 'examples': 8192}:
        raise ValueError('complete cache certificate required')
    if len(plan['treatment_ids']) != 32 or len(set(plan['treatment_ids'])) != 32:
        raise ValueError('fixed32 treatment order required')
    receipts = {}
    caches = {}
    for identity in identities:
        record = load_completed_donor(directory, identity, contract)
        if record is None or record['treatments'] != plan['treatment_ids']:
            raise ValueError('complete matching donor receipt required')
        path = directory / (hashlib.sha256(identity.encode()).hexdigest()+'.json')
        receipts[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
        caches[identity] = record['cache_sha256']
    return {'receipt_sha256': receipts, 'cache_sha256': caches,
            'completion_sha256': hashlib.sha256(completion).hexdigest()}

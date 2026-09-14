import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def exclusive_cache_writer(directory: Path):
    import msvcrt

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('another cache writer holds this directory') from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def load_completed_donor(directory: Path, identity: str, contract_sha256: str) -> dict | None:
    stem = hashlib.sha256(identity.encode()).hexdigest()
    receipt = directory / (stem + '.json')
    if not receipt.exists():
        return None
    record = json.loads(receipt.read_text())
    if record['identity'] != identity or record['contract_sha256'] != contract_sha256:
        raise ValueError('cache identity or computation contract differs')
    path = directory / (stem + '.npz')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != record['cache_sha256']:
        raise ValueError('completed donor cache changed')
    return record


def save_completed_donor(directory: Path, identity: str, contract_sha256: str,
                         examples: list[dict], expected_treatments: list[str]) -> dict:
    if load_completed_donor(directory, identity, contract_sha256) is not None:
        raise ValueError('completed donor must be reused, not overwritten')
    if [r['treatment_id'] for r in examples] != expected_treatments or not examples:
        raise ValueError('complete ordered treatment set required')
    inputs = np.stack([r['input'] for r in examples])
    targets = np.stack([r['target'] for r in examples])
    if (inputs.dtype != np.float32 or inputs.shape != (len(examples), 3, 128, 128)
            or targets.shape != (len(examples), 4) or not np.isfinite(inputs).all()
            or not np.isfinite(targets).all() or not np.all(targets == targets[0])):
        raise ValueError('invalid donor examples or inconsistent canonical target')
    stem = hashlib.sha256(identity.encode()).hexdigest()
    temporary = directory / (stem + '.npz.partial')
    with temporary.open('wb') as stream:
        np.savez(stream, inputs=inputs, target=targets[0],
                 after_mu=np.stack([r['after_mu'] for r in examples]),
                 after_scale=np.asarray([r['after_scale'] for r in examples]))
        stream.flush()
        os.fsync(stream.fileno())
    path = directory / (stem + '.npz')
    os.replace(temporary, path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = {'identity': identity, 'contract_sha256': contract_sha256,
              'cache_sha256': digest, 'treatments': expected_treatments,
              'shape': list(inputs.shape)}
    temporary = directory / (stem + '.json.partial')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / (stem + '.json'))
    return record

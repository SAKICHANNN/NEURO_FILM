import hashlib
import io
import json
import os
from pathlib import Path

import torch


def save_final_checkpoint(directory: Path, result: dict, *, contract_sha256: str,
                          normalizer: dict, expected_steps: int) -> dict:
    if (result['optimizer_updates'] != expected_steps
            or len(result['training_mse']) != expected_steps or expected_steps <= 0):
        raise ValueError('only completed final-step results can be sealed')
    tensors = result['state_dict']
    if not tensors or any(not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError('finite final model required')
    normalization = {key: torch.as_tensor(normalizer[key], dtype=torch.float64)
                     for key in ['mean', 'scale']}
    if (any(t.shape != (4,) or not torch.isfinite(t).all() for t in normalization.values())
            or torch.any(normalization['scale'] <= 1e-12)
            or not torch.isfinite(torch.as_tensor(result['training_mse'])).all()):
        raise ValueError('finite fitting history and valid four-coordinate normalization required')
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ['final.pt', 'final.json']):
        raise FileExistsError('existing final artifact cannot be overwritten')
    with (directory / 'final.claim').open('x') as stream:
        stream.write(contract_sha256)
        stream.flush()
        os.fsync(stream.fileno())
    payload = {'state_dict': tensors, 'optimizer_updates': expected_steps,
               'contract_sha256': contract_sha256,
               'normalizer': normalization,
               'training_mse': result['training_mse']}
    temporary = directory / 'final.pt.partial'
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    final = directory / 'final.pt'
    os.replace(temporary, final)
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    receipt = {'contract_sha256': contract_sha256, 'checkpoint_sha256': digest,
               'optimizer_updates': expected_steps}
    temporary = directory / 'final.json.partial'
    with temporary.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / 'final.json')
    return receipt


def load_final_checkpoint(directory: Path, *, contract_sha256: str,
                          expected_steps: int) -> dict:
    receipt = json.loads((directory / 'final.json').read_text())
    if (receipt['contract_sha256'] != contract_sha256
            or receipt['optimizer_updates'] != expected_steps):
        raise ValueError('final checkpoint contract or step count differs')
    blob = (directory / 'final.pt').read_bytes()
    if hashlib.sha256(blob).hexdigest() != receipt['checkpoint_sha256']:
        raise ValueError('final checkpoint content changed')
    payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if (payload['contract_sha256'] != contract_sha256
            or payload['optimizer_updates'] != expected_steps):
        raise ValueError('checkpoint payload differs from receipt')
    return payload

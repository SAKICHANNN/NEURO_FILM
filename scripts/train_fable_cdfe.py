import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from src.data.fable_windows_memory import limit_current_process_committed_memory


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blob = args.contract.read_bytes()
    seal = json.loads(blob)
    if seal.get('status') != 'SEALED_FOR_SINGLE_FITTING_RUN':
        raise ValueError('sealed training contract required; draft cannot execute')
    required = {'scripts/train_fable_cdfe.py', 'src/training/fable_cdfe_fit.py',
        'src/training/fable_cdfe_batches.py', 'src/training/fable_cdfe_checkpoint.py',
        'src/data/fable_cdfe_training_data.py', 'src/models/canonical_photometry.py',
        'src/data/fable_windows_memory.py', seal['training_config'], seal['cache_plan']}
    if not required.issubset(seal['source_sha256']):
        raise ValueError('training source bindings incomplete')
    cache_plan = json.loads((root / seal['cache_plan']).read_text())
    for relative, expected in cache_plan['source_sha256'].items():
        if seal['source_sha256'].get(relative) != expected:
            raise ValueError(f'missing or differing cache dependency binding: {relative}')
    for relative, expected in seal['source_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'training source changed: {relative}')
    config = json.loads((root / seal['training_config']).read_text())
    for key in ['role_lock', 'model_config']:
        if config[key] not in seal['source_sha256']:
            raise ValueError(f'missing {key} binding')
    training = config['training']
    if (training['device'] != 'cpu' or training['dtype'] != 'float32'
            or training['autocast'] or not training['deterministic_algorithms']
            or training['steps'] != 4000 or training['batch_size'] != 32
            or training['loader_workers'] != 0):
        raise ValueError('fixed CPU float32 4000-step protocol required')
    job = limit_current_process_committed_memory(training['allocation_limit_bytes'])
    import numpy as np
    import torch
    from src.data.fable_cdfe_training_data import load_fitting_cache
    from src.models.canonical_photometry import CanonicalPhotometryCNN
    from src.training.fable_cdfe_fit import fit_final_model
    from src.training.fable_cdfe_checkpoint import save_final_checkpoint

    if seal['runtime'] != {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}:
        raise ValueError('sealed training runtime differs')
    torch.set_num_threads(training['cpu_threads'])
    torch.set_num_interop_threads(training['interop_threads'])
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    lock = json.loads((root / config['role_lock']).read_text())
    data = load_fitting_cache(root / seal['cache_directory'], root / seal['cache_plan'],
        locked_fitting_ids=lock['assignment']['fit'], receipt_sha256=seal['receipt_sha256'])
    architecture = json.loads((root / config['model_config']).read_text())['architecture']
    output = root / seal['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    contract = hashlib.sha256(blob).hexdigest()
    with (output / 'run.claim').open('x') as stream:
        stream.write(contract)
        stream.flush()
        os.fsync(stream.fileno())
    torch.manual_seed(training['initialization_seed'])
    model = CanonicalPhotometryCNN(**architecture)
    with (output / 'training.jsonl').open('x') as stream:
        def progress(record):
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if record['optimizer_updates'] % 256 == 0:
                print(json.dumps(record), flush=True)
        result = fit_final_model(model, data['inputs'], data['targets'],
                                 training=training, progress=progress)
        os.fsync(stream.fileno())
    receipt = save_final_checkpoint(output, result, contract_sha256=contract,
        normalizer=data['normalizer'], expected_steps=training['steps'])
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

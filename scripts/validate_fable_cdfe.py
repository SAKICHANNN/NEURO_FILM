import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from src.data.fable_windows_memory import limit_current_process_committed_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blob = args.contract.read_bytes()
    seal = json.loads(blob)
    if seal.get('status') != 'SEALED_FOR_SINGLE_FITTING_RUN':
        raise ValueError('sealed training contract required before validation')
    required = {'scripts/validate_fable_cdfe.py', 'src/eval/fable_cdfe_validation.py',
        'src/eval/fable_cdfe_gates.py', 'src/training/fable_cdfe_checkpoint.py',
        'src/models/canonical_photometry.py', 'src/data/fable_windows_memory.py',
        seal['training_config'], seal['validation_plan'], seal['cache_plan']}
    if not required.issubset(seal['source_sha256']):
        raise ValueError('validation dependency bindings incomplete')
    cache_plan = json.loads((root / seal['cache_plan']).read_text())
    plan = json.loads((root / seal['validation_plan']).read_text())
    for dependency_plan in [cache_plan, plan]:
        for relative, expected in dependency_plan['source_sha256'].items():
            if seal['source_sha256'].get(relative) != expected:
                raise ValueError('computation bindings differ')
    for relative, expected in seal['source_sha256'].items():
        with (root / relative).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'sealed source changed: {relative}')
    config = json.loads((root / seal['training_config']).read_text())
    for key in ['role_lock', 'model_config', 'native_measurement_config']:
        if config[key] not in seal['source_sha256']:
            raise ValueError(f'unbound {key}')
    job = limit_current_process_committed_memory(config['training']['allocation_limit_bytes'])
    import numpy as np
    import torch
    from src.models.canonical_photometry import CanonicalPhotometryCNN
    from src.training.fable_cdfe_checkpoint import load_final_checkpoint
    from src.eval.fable_cdfe_validation import evaluate_validation

    if seal['runtime'] != {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}:
        raise ValueError('validation runtime differs')
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.set_num_interop_threads(config['training']['interop_threads'])
    torch.use_deterministic_algorithms(True)
    output = root / seal['output_directory']
    contract = hashlib.sha256(blob).hexdigest()
    checkpoint = load_final_checkpoint(output, contract_sha256=contract, expected_steps=4000)
    architecture = json.loads((root / config['model_config']).read_text())['architecture']
    model = CanonicalPhotometryCNN(**architecture)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    lock = json.loads((root / config['role_lock']).read_text())
    plan = json.loads((root / seal['validation_plan']).read_text())
    native = json.loads((root / config['native_measurement_config']).read_text())
    with (output / 'validation.claim').open('x') as stream:
        json.dump({'contract_sha256': contract,
                   'checkpoint_sha256': hashlib.sha256((output / 'final.pt').read_bytes()).hexdigest()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = evaluate_validation(model, plan['rows'], plan['treatments'],
        locked_ids=lock['assignment']['validation'], locked_treatment_ids=plan['treatment_ids'],
        normalizer={k: v.numpy() for k, v in checkpoint['normalizer'].items()}, native_config=native)
    temporary = output / 'validation.npz.partial'
    with temporary.open('xb') as stream:
        np.savez(stream, predicted=result['predicted'], targets=result['targets'])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / 'validation.npz')
    report = {k: v for k, v in result.items() if k not in ['predicted', 'targets']}
    report.update(contract_sha256=contract,
        arrays_sha256=hashlib.sha256((output / 'validation.npz').read_bytes()).hexdigest())
    with (output / 'validation.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report['gate']), flush=True)


if __name__ == '__main__':
    main()

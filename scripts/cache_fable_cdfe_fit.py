import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from src.data.fable_windows_memory import limit_current_process_committed_memory


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--one-donor', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = root / 'outputs/fable_cdfe68_computation_v1/fit_cache_plan_v2.json'
    plan = json.loads(plan_path.read_text())
    job = limit_current_process_committed_memory(plan['required_memory_cap_bytes'])
    import numpy as np
    import torch
    from src.data.fable_cdfe_cache import exclusive_cache_writer, load_completed_donor, save_completed_donor
    from src.data.fable_cdfe_preprocessing import donor_training_examples

    if plan['runtime'] != {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__}:
        raise ValueError('cache runtime differs from plan')
    for relative, expected in plan['source_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'cache dependency changed: {relative}')
    lock = json.loads((root / 'outputs/fable_cdfe68_exposure_v1/role_lock.json').read_text())
    if [r['identity'] for r in plan['rows']] != lock['assignment']['fit']:
        raise ValueError('cache input differs from locked fitting identities')
    treatments = json.loads((root / 'outputs/fable_cdfe68_computation_v1/treatments.json').read_text())['partitions']['fit']
    if [r['id'] for r in treatments] != plan['treatment_ids']:
        raise ValueError('treatment ordering differs')
    config = json.loads((root / 'configs/fable_cdfe68_native_v1.json').read_text())
    torch.set_num_threads(plan['cpu_threads'])
    output = root / 'outputs/fable_cdfe68_fit_cache_v1'
    contract = digest(plan_path)
    with exclusive_cache_writer(output):
        frame = {'contract_sha256': contract, 'runner_sha256': digest(Path(__file__)),
                 'memory_limit_bytes': plan['required_memory_cap_bytes']}
        frame_path = output / 'frame.json'
        if frame_path.exists():
            if json.loads(frame_path.read_text()) != frame:
                raise ValueError('existing cache execution frame differs')
        else:
            frame_path.write_text(json.dumps(frame, indent=2))
        processed = 0
        for row in plan['rows']:
            identity = row['identity']
            if load_completed_donor(output, identity, contract) is not None:
                continue
            if digest(row['native_path']) != row['native_sha256']:
                raise ValueError('native fitting image changed')
            started = time.perf_counter()
            linear = np.load(row['native_path'], mmap_mode='r', allow_pickle=False)
            examples = list(donor_training_examples(linear, treatments,
                seed=row['dequantization_seed'], measurement=config['measurement'],
                input_size=128, slope_limit=1.25, offset_limit=.35))
            save_completed_donor(output, identity, contract, examples, plan['treatment_ids'])
            del examples, linear
            print(json.dumps({'identity': identity, 'seconds': time.perf_counter()-started,
                              'status': 'DONOR_CACHE_COMPLETE'}), flush=True)
            processed += 1
            if args.one_donor and processed == 1:
                return
        receipts = [load_completed_donor(output, r['identity'], contract) for r in plan['rows']]
        if any(r is None for r in receipts):
            raise ValueError('cache incomplete')
        (output / 'complete.json').write_text(json.dumps({'contract_sha256': contract,
            'donors': len(receipts), 'examples': len(receipts)*len(treatments)}))


if __name__ == '__main__':
    main()

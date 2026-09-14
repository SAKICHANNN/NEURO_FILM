import hashlib
import json
import time
from pathlib import Path

from src.data.fable_windows_memory import limit_current_process_committed_memory


def main():
    job = limit_current_process_committed_memory(10 * 1024**3)
    import numpy as np
    import torch
    import win32api
    import win32process
    from src.data.fable_cdfe_cache import load_completed_donor
    from src.models.canonical_photometry import CanonicalPhotometryCNN

    root = Path(__file__).resolve().parents[1]
    plan_path = root / 'outputs/fable_cdfe68_computation_v1/fit_cache_plan_v2.json'
    blob = plan_path.read_bytes()
    plan = json.loads(blob)
    identity = plan['rows'][0]['identity']
    role_lock = json.loads((root / 'outputs/fable_cdfe68_exposure_v1/role_lock.json').read_text())
    assert identity == role_lock['assignment']['fit'][0]
    directory = root / 'outputs/fable_cdfe68_fit_cache_v1'
    receipt = load_completed_donor(directory, identity, hashlib.sha256(blob).hexdigest())
    assert receipt is not None and receipt['treatments'] == plan['treatment_ids']
    stem = hashlib.sha256(identity.encode()).hexdigest()
    with np.load(directory / (stem + '.npz'), allow_pickle=False) as data:
        inputs = data['inputs']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    architecture = json.loads((root / 'configs/fable_canonical_cnn_v1.json').read_text())['architecture']
    torch.manual_seed(2026091402)
    model = CanonicalPhotometryCNN(**architecture)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    started = time.perf_counter()
    prediction = model(torch.from_numpy(inputs))
    prediction.square().mean().backward()
    assert prediction.shape == (32, 4) and torch.isfinite(prediction).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
    report = {'identity': identity, 'cache_sha256': receipt['cache_sha256'],
        'batch_shape': list(inputs.shape), 'prediction_shape': list(prediction.shape),
        'seconds': time.perf_counter()-started, 'optimizer_updates': 0,
        'probe': 'mean squared prediction; engineering gradient probe, not fitted-target loss',
        'finite_gradients': True, 'weights_unchanged': True,
        'peak_working_set_bytes': win32process.GetProcessMemoryInfo(win32api.GetCurrentProcess())['PeakWorkingSetSize'],
        'process_committed_limit_bytes': 10 * 1024**3,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = root / 'outputs/fable_cdfe68_computation_v1/cached_backward_engineering.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.fable_runtime import computation_runtime
from src.training.fable_source_closure import local_import_closure


@pytest.mark.parametrize('validation_passes', [True, False])
def test_mock_cli_chain_and_single_run_claims(tmp_path, monkeypatch, validation_passes):
    import scripts.train_fable_cdfe as train
    import scripts.validate_fable_cdfe as evaluate
    import src.data.fable_cdfe_training_data as training_data
    import src.training.fable_cdfe_fit as fit
    import src.models.canonical_photometry as models
    import src.eval.fable_cdfe_validation as validation
    import src.eval.fable_cdfe_assessment as assessment
    import src.eval.fable_cdfe_visual_panel as panel
    from src.eval.fable_cdfe_gates import validation_gate

    root = Path(__file__).resolve().parents[1]
    assignment = {'fit': [f'synthetic-fit/{j}' for j in range(256)],
                  'validation': [f'synthetic-val/{j}' for j in range(32)]}
    files = {}
    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return str(path)
    lock = write('lock.json', {'assignment': assignment})
    architecture = write('model.json', {'architecture': {}})
    native = write('native.json', {})
    config = json.loads((root/'configs/fable_cdfe68_training_draft_v1.json').read_text())
    config.update(role_lock=lock, model_config=architecture, native_measurement_config=native)
    config_path = write('config.json', config)
    cache = write('cache.json', {'source_sha256': {}})
    tids = [str(j) for j in range(8)]
    vp = write('validation.json', {'source_sha256': {}, 'rows': [], 'treatments': [], 'treatment_ids': tids})
    ap = write('assessment.json', {'source_sha256': {}})
    sources = local_import_closure(root, ['scripts/train_fable_cdfe.py', 'scripts/validate_fable_cdfe.py'])
    sources.update(files)
    seal = {'status': 'SEALED_FOR_SINGLE_FITTING_RUN', 'source_sha256': sources,
        'computation_runtime': computation_runtime(),
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
        'training_config': config_path, 'cache_plan': cache, 'validation_plan': vp,
        'assessment_plan': ap, 'cache_directory': str(tmp_path/'cache'),
        'output_directory': str(tmp_path/'output'), 'receipt_sha256': {}}
    contract = write('seal.json', seal)
    monkeypatch.setattr(train, 'limit_current_process_committed_memory', lambda _: object())
    monkeypatch.setattr(evaluate, 'limit_current_process_committed_memory', lambda _: object())
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _: None)
    monkeypatch.setattr(models, 'CanonicalPhotometryCNN', lambda **_: torch.nn.Linear(3, 4))
    norm = {'mean': np.zeros(4), 'scale': np.ones(4)}
    monkeypatch.setattr(training_data, 'load_fitting_cache', lambda *a, **k:
        {'inputs': np.zeros((1, 3), dtype=np.float32), 'targets': np.zeros((1, 4), dtype=np.float32),
         'normalizer': norm})
    calls = []
    def fake_fit(model, *args, **kwargs):
        calls.append('mock_fit_zero_actual_updates')
        return {'optimizer_updates': 4000, 'training_mse': [0.]*4000, 'state_dict': model.state_dict()}
    monkeypatch.setattr(fit, 'fit_final_model', fake_fit)
    def fake_validation(*args, **kwargs):
        targets = np.ones((32, 8, 4))
        predicted = targets.copy() if validation_passes else targets*0
        return {'predicted': predicted, 'targets': targets, 'identities': assignment['validation'],
                'treatment_ids': tids, 'gate': validation_gate(predicted, targets, np.zeros(4))}
    monkeypatch.setattr(validation, 'evaluate_validation', fake_validation)
    def fake_assessment(*args, **kwargs):
        calls.append('mock_assessment')
        return {'arrays': {'synthetic': np.zeros(1)}, 'gates': {'numeric_passed': True},
                'identities': [], 'query_ids': [], 'treatment_ids': []}
    monkeypatch.setattr(assessment, 'evaluate_assessment', fake_assessment)
    monkeypatch.setattr(panel, 'render_comfort_panel', lambda *a, **k: calls.append('mock_comfort'))
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        monkeypatch.setattr(sys, 'argv', ['train', '--contract', contract])
        train.main()
        assert (tmp_path/'output/final.json').exists()
        with pytest.raises(FileExistsError):
            train.main()
        monkeypatch.setattr(sys, 'argv', ['validate', '--contract', contract])
        evaluate.main()
        with pytest.raises(FileExistsError):
            evaluate.main()
        monkeypatch.setattr(sys, 'argv', ['assess', '--contract', contract, '--assessment'])
        if validation_passes:
            evaluate.main()
            assert calls == ['mock_fit_zero_actual_updates', 'mock_assessment', 'mock_comfort']
            assert (tmp_path/'output/assessment.json').exists()
            with pytest.raises(FileExistsError):
                evaluate.main()
        else:
            with pytest.raises(ValueError, match='validation gate failed'):
                evaluate.main()
            assert calls == ['mock_fit_zero_actual_updates']
            assert not (tmp_path/'output/assessment.claim').exists()
    finally:
        torch.use_deterministic_algorithms(previous)

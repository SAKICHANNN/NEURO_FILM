import hashlib

import numpy as np
import pytest
import torch

from src.eval.fable_cdfe_validation import evaluate_validation


def test_wrong_validation_membership_rejected_before_native_open():
    ids = [f'validation/{j}' for j in range(32)]
    rows = [{'identity': i} for i in ids]
    treatments = [{'id': str(j)} for j in range(8)]
    with pytest.raises(ValueError, match='exact locked'):
        evaluate_validation(torch.nn.Linear(3, 4).eval(), rows[::-1], treatments,
            locked_ids=ids, locked_treatment_ids=[t['id'] for t in treatments],
            normalizer={}, native_config={})


def test_synthetic_validation_keeps_target_normalization_and_order(tmp_path, monkeypatch):
    import src.eval.fable_cdfe_validation as validation

    path = tmp_path / 'synthetic.npy'
    np.save(path, np.zeros((2, 2, 3), dtype=np.uint16))
    ids = [f'synthetic/{j}' for j in range(32)]
    rows = [dict(identity=i, native_path=str(path),
                 native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 dequantization_seed=j) for j, i in enumerate(ids)]
    treatments = [{'id': str(j)} for j in range(8)]
    observed = []

    def examples(linear, treatment_rows, **kwargs):
        observed.append(kwargs['seed'])
        for _ in treatment_rows:
            yield {'input': np.ones((3, 128, 128), dtype=np.float32),
                   'target': np.full(4, 5., dtype=np.float64)}

    class Ones(torch.nn.Module):
        def forward(self, x):
            return torch.ones((len(x), 4))

    monkeypatch.setattr(validation, 'donor_training_examples', examples)
    result = evaluate_validation(Ones().eval(), rows, treatments, locked_ids=ids,
        locked_treatment_ids=[t['id'] for t in treatments],
        normalizer={'mean': np.full(4, 3.), 'scale': np.full(4, 2.)},
        native_config={'measurement': {}})
    np.testing.assert_array_equal(result['targets'], np.ones((32, 8, 4)))
    np.testing.assert_array_equal(result['predicted'], result['targets'])
    assert observed == list(range(32))

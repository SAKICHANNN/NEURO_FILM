import numpy as np
import pytest
import torch

from src.training.fable_cdfe_fit import fit_final_model


def test_exact_updates_replay_and_no_broadcasting():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        config = dict(learning_rate=.0003, betas=[.9, .999], epsilon=1e-8,
                      weight_decay=.0001, batch_size=4, steps=5, batch_seed=73)
        x = np.arange(24, dtype=np.float32).reshape(8, 3)/24
        y = np.zeros((8, 4), dtype=np.float32)
        results = []
        for _ in range(2):
            torch.manual_seed(91)
            model = torch.nn.Linear(3, 4)
            initial = model.weight.detach().clone()
            reports = []
            result = fit_final_model(model, x, y, training=config, progress=reports.append)
            assert result['optimizer_updates'] == 5
            assert [r['optimizer_updates'] for r in reports] == [1, 2, 3, 4, 5]
            assert not torch.equal(initial, model.weight)
            assert not model.training
            results.append(result)
        assert results[0]['training_mse'] == results[1]['training_mse']
        for key in results[0]['state_dict']:
            assert torch.equal(results[0]['state_dict'][key], results[1]['state_dict'][key])
        with pytest.raises(ValueError, match='broadcasting'):
            fit_final_model(torch.nn.Linear(3, 1), x, y, training=config, progress=lambda _: None)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_nonfinite_gradient_stops_before_any_update():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = torch.nn.Linear(3, 4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.weight.register_hook(lambda gradient: gradient * float('nan'))
        config = dict(learning_rate=.0003, betas=[.9, .999], epsilon=1e-8,
                      weight_decay=.0001, batch_size=4, steps=5, batch_seed=73)
        reports = []
        with pytest.raises(FloatingPointError, match='gradient'):
            fit_final_model(model, np.ones((8, 3), dtype=np.float32),
                np.zeros((8, 4), dtype=np.float32), training=config, progress=reports.append)
        assert reports == []
        for key, value in model.state_dict().items():
            assert torch.equal(before[key], value)
    finally:
        torch.use_deterministic_algorithms(previous)

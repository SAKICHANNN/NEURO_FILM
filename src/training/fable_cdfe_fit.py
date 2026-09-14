from collections.abc import Callable

import numpy as np
import torch

from src.training.fable_cdfe_batches import shuffled_epoch_batches


def fit_final_model(model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray,
                    *, training: dict, progress: Callable[[dict], None]) -> dict:
    if (inputs.dtype != np.float32 or targets.dtype != np.float32
            or len(inputs) != len(targets) or targets.shape != (len(inputs), 4)
            or not np.isfinite(inputs).all() or not np.isfinite(targets).all()):
        raise ValueError('finite float32 fitting inputs and four-coordinate targets required')
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in parameters):
        raise ValueError('CPU float32 model required')
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError('deterministic algorithms required')
    optimizer = torch.optim.AdamW(parameters, lr=training['learning_rate'],
        betas=tuple(training['betas']), eps=training['epsilon'],
        weight_decay=training['weight_decay'], foreach=False, fused=False)
    model.train()
    losses = []
    for indices in shuffled_epoch_batches(examples=len(inputs), batch_size=training['batch_size'],
            steps=training['steps'], seed=training['batch_seed']):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.from_numpy(inputs[indices]))
        expected = torch.from_numpy(targets[indices])
        if prediction.shape != expected.shape:
            raise ValueError('four-coordinate predictions required without broadcasting')
        loss = torch.nn.functional.mse_loss(prediction, expected)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite fitting loss; stop without retry')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('missing or nonfinite gradient; stop without update')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('nonfinite updated parameter; stop without checkpoint')
        losses.append(float(loss.detach()))
        progress({'optimizer_updates': len(losses), 'training_mse': losses[-1]})
    model.eval()
    return {'optimizer_updates': len(losses), 'training_mse': losses,
            'state_dict': {key: value.detach().clone() for key, value in model.state_dict().items()}}

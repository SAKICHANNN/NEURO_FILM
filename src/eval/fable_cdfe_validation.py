import hashlib
from pathlib import Path

import numpy as np
import torch

from src.data.fable_cdfe_preprocessing import donor_training_examples
from src.eval.fable_cdfe_gates import validation_gate


@torch.inference_mode()
def evaluate_validation(model: torch.nn.Module, rows: list[dict], treatments: list[dict],
                        *, locked_ids: list[str], locked_treatment_ids: list[str],
                        normalizer: dict, native_config: dict) -> dict:
    identities = [row['identity'] for row in rows]
    if (len(identities) != 32 or len(set(identities)) != 32 or identities != locked_ids
            or len(treatments) != 8 or len(set(locked_treatment_ids)) != 8
            or [t['id'] for t in treatments] != locked_treatment_ids):
        raise ValueError('exact locked validation32 and treatment8 order required')
    if model.training:
        raise ValueError('sealed final model must be in evaluation mode')
    mean, scale = [np.asarray(normalizer[key], dtype=np.float64) for key in ['mean', 'scale']]
    if (mean.shape != (4,) or scale.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 1e-12)):
        raise ValueError('valid fitting-only normalization required')
    predicted = np.empty((32, 8, 4), dtype=np.float64)
    targets = np.empty_like(predicted)
    for j, row in enumerate(rows):
        path = Path(row['native_path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['native_sha256']:
                raise ValueError('validation native content changed')
        linear = np.load(path, mmap_mode='r', allow_pickle=False)
        examples = list(donor_training_examples(linear, treatments,
            seed=row['dequantization_seed'], measurement=native_config['measurement'],
            input_size=128, slope_limit=1.25, offset_limit=.35))
        inputs = torch.from_numpy(np.stack([e['input'] for e in examples]))
        prediction = model(inputs)
        if prediction.shape != (8, 4) or not torch.isfinite(prediction).all():
            raise ValueError('finite validation predictions required')
        predicted[j] = prediction.cpu().numpy()
        targets[j] = (np.stack([e['target'] for e in examples])-mean)/scale
        del linear, examples, inputs, prediction
    return {'predicted': predicted, 'targets': targets, 'identities': identities,
            'treatment_ids': locked_treatment_ids,
            'gate': validation_gate(predicted, targets, np.zeros(4, dtype=np.float64))}

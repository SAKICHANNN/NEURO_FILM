import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from src.data.fable_cdfe_preprocessing import donor_training_examples


@torch.inference_mode()
def assessment_predictions(model: torch.nn.Module, rows: list[dict], treatments: list[dict],
                           *, locked_ids: list[str], locked_treatment_ids: list[str],
                           normalizer: dict, native_config: dict,
                           progress: Callable[[dict], None]) -> dict:
    identities = [row['identity'] for row in rows]
    if (len(identities) != 32 or len(set(identities)) != 32 or identities != locked_ids
            or len(treatments) != 32 or len(set(locked_treatment_ids)) != 32
            or [t['id'] for t in treatments] != locked_treatment_ids):
        raise ValueError('exact locked assessment32 and treatment32 order required')
    if model.training:
        raise ValueError('final model must be in evaluation mode')
    mean, scale = [np.asarray(normalizer[k], dtype=np.float64) for k in ['mean', 'scale']]
    if (mean.shape != (4,) or scale.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 1e-12)):
        raise ValueError('valid fitting normalization required')
    predicted = np.empty((32, 32, 4), dtype=np.float64)
    canonical_targets = np.empty((32, 32, 4), dtype=np.float64)
    mu = np.empty((32, 32, 3), dtype=np.float64)
    after_scale = np.empty((32, 32), dtype=np.float64)
    for j, row in enumerate(rows):
        path = Path(row['native_path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['native_sha256']:
                raise ValueError('assessment native content changed')
        linear = np.load(path, mmap_mode='r', allow_pickle=False)
        examples = list(donor_training_examples(linear, treatments,
            seed=row['dequantization_seed'], measurement=native_config['measurement'],
            input_size=128, slope_limit=1.25, offset_limit=.35))
        inputs = torch.from_numpy(np.stack([e['input'] for e in examples]))
        output = model(inputs)
        if output.shape != (32, 4) or not torch.isfinite(output).all():
            raise ValueError('finite assessment predictions required')
        predicted[j] = output.cpu().numpy().astype(np.float64)*scale+mean
        canonical_targets[j] = np.stack([e['target'] for e in examples])
        mu[j] = np.stack([e['after_mu'] for e in examples])
        after_scale[j] = [e['after_scale'] for e in examples]
        progress({'identity': row['identity'], 'completed_donors': j+1})
        del linear, examples, inputs, output
    return {'predicted_canonical': predicted, 'canonical_targets': canonical_targets,
            'after_mu': mu, 'after_scale': after_scale,
            'identities': identities, 'treatment_ids': locked_treatment_ids}

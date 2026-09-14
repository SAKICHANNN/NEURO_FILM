import hashlib

import numpy as np
import torch

from src.eval.fable_cdfe_assessment_predictions import assessment_predictions


def test_predictor_never_receives_canonical_target(tmp_path, monkeypatch):
    import src.eval.fable_cdfe_assessment_predictions as module

    path = tmp_path / 'synthetic.npy'
    np.save(path, np.zeros((2, 2, 3), dtype=np.uint16))
    ids = [str(j) for j in range(32)]
    rows = [dict(identity=i, native_path=str(path), native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 dequantization_seed=j) for j, i in enumerate(ids)]
    treatments = [{'id': str(j)} for j in range(32)]

    def examples(linear, ts, **kwargs):
        for j, _ in enumerate(ts):
            yield dict(input=np.ones((3, 128, 128), dtype=np.float32)*j,
                       target=np.full(4, np.nan), after_mu=np.full(3, j), after_scale=j+1)

    class Probe(torch.nn.Module):
        def forward(self, x):
            return x.mean((1, 2, 3))[:, None].repeat(1, 4)

    monkeypatch.setattr(module, 'donor_training_examples', examples)
    result = assessment_predictions(Probe().eval(), rows, treatments, locked_ids=ids,
        locked_treatment_ids=ids, normalizer={'mean': np.ones(4), 'scale': np.full(4, 2.)},
        native_config={'measurement': {}}, progress=lambda _: None)
    np.testing.assert_array_equal(result['predicted_canonical'][0, :, 0], 2*np.arange(32)+1)
    np.testing.assert_array_equal(result['after_mu'][0, :, 0], np.arange(32))
    np.testing.assert_array_equal(result['after_scale'][0], np.arange(32)+1)

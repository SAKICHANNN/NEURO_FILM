import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.fable_canonical_inference import AfterOnlyCanonicalPredictor
from src.eval.fable_canonical_prior import canonical_measure, decode_canonical_prior
from src.models.canonical_photometry import CanonicalPhotometryCNN


@pytest.fixture
def setup():
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/fable_canonical_cnn_v1.json').read_text())
    with torch.random.fork_rng():
        torch.manual_seed(17)
        net = CanonicalPhotometryCNN(**config['architecture']).eval()
    mean = np.array([.1, -.2, .3, .4])
    scale = np.array([.2, .3, .4, .5])
    predictor = AfterOnlyCanonicalPredictor(net, target_mean=mean, target_scale=scale,
                                          **config['measurement'], **config['operator'])
    images = np.random.default_rng(9).integers(0, 256, (3, 128, 128, 3)) / 255
    return config, net, predictor, images


def test_batch_independence_and_serialization(setup):
    config, net, predictor, images = setup
    batch = predictor.predict(images)
    for i in range(len(images)):
        single = predictor.predict(images[i:i+1])[0]
        np.testing.assert_allclose(batch[i]['predicted_canonical'], single['predicted_canonical'], atol=1e-7)
        np.testing.assert_allclose(batch[i]['raw'], single['raw'], atol=1e-6)
    stream = io.BytesIO()
    torch.save(net.state_dict(), stream)
    stream.seek(0)
    restored = CanonicalPhotometryCNN(**config['architecture']).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    predictor.network = restored
    np.testing.assert_array_equal(predictor.predict(images)[0]['raw'], batch[0]['raw'])


def test_after_only_input_and_target_destandardization(setup):
    config, net, predictor, images = setup
    observed = []
    handle = net.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach().cpu().numpy()))
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.head[-1].bias.copy_(torch.tensor([1., -2., 3., -4.]))
    actual = predictor.predict(images)
    handle.remove()
    target = np.array([1., -2., 3., -4.]) * predictor.target_scale + predictor.target_mean
    for i, image in enumerate(images):
        measurement = canonical_measure(image, **config['measurement'])
        np.testing.assert_allclose(observed[0][i], measurement['tensor'].transpose(2, 0, 1), atol=3e-7)
        expected = decode_canonical_prior(measurement, target, **config['operator'])
        np.testing.assert_allclose(actual[i]['raw'], expected['raw'])
        np.testing.assert_allclose(actual[i]['predicted_canonical'], target)


def test_spatial_encoder_receives_gradients_without_weight_update(setup):
    _, net, _, _ = setup
    net.train()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    x = torch.linspace(-1, 1, 3 * 128 * 128).reshape(1, 3, 128, 128)
    net(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(layer.weight.grad.abs().sum() > 0 for layer in net.encoder if isinstance(layer, torch.nn.Conv2d))
    assert sum(p.numel() for p in net.parameters()) < 600_000
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())


def test_invalid_inference_boundary(setup):
    _, net, predictor, images = setup
    with pytest.raises(ValueError, match='BHWC'):
        predictor.predict(images[0])
    with pytest.raises(ValueError, match='configured input size'):
        predictor.predict(images[:, :64])
    bad = images.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite HWC'):
        predictor.predict(bad)
    net.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        predictor.predict(images)


def test_nonfinite_predictions_are_rejected(setup):
    _, net, predictor, images = setup
    with torch.no_grad():
        net.head[-1].bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='four finite standardized'):
        predictor.predict(images)

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.fable_canonical_inference import AfterOnlyCanonicalPredictor, native_canonical_input
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


def test_native_statistics_precede_resize(setup):
    config, _, _, _ = setup
    image = np.random.default_rng(21).integers(0, 256, (256, 384, 3)) / 255
    full = canonical_measure(image, **config['measurement'])
    native = native_canonical_input(image, input_size=128, **config['measurement'])
    expected = full['tensor'].astype(np.float32).reshape(128, 2, 128, 3, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(native['tensor'], expected.transpose(2, 0, 1), atol=3e-7)
    np.testing.assert_array_equal(native['target'], full['target'])
    np.testing.assert_array_equal(native['mu'], full['mu'])
    assert native['scale'] == full['scale']
    resized_rgb = image.reshape(128, 2, 128, 3, 3).mean(axis=(1, 3))
    wrong = canonical_measure(resized_rgb, **config['measurement'])
    assert abs(wrong['scale'] - native['scale']) > .1
    assert native['measurement_shape'] == image.shape


def test_native_variable_shapes_decode_with_full_after_statistics(setup):
    config, net, predictor, _ = setup
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    rng = np.random.default_rng(15)
    images = [rng.integers(0, 256, (h, w, 3)) / 255 for h, w in [(140, 193), (256, 384)]]
    actual = predictor.predict_native(images, input_size=128)
    for i, image in enumerate(images):
        full = canonical_measure(image, **config['measurement'])
        expected = decode_canonical_prior(full, predictor.target_mean, **config['operator'])
        np.testing.assert_allclose(actual[i]['raw'], expected['raw'])
        assert actual[i]['measurement_shape'] == image.shape
        single = predictor.predict_native([image], input_size=128)[0]
        np.testing.assert_array_equal(actual[i]['raw'], single['raw'])


def test_binding_scale_floor_is_not_invariant():
    from src.eval.fable_reference_photometry import transform
    image = np.full((8, 8, 3), .5)
    image[0, 0] += .0001
    before = canonical_measure(image, epsilon=0, scale_floor=.001, tensor_limit=8)
    after = canonical_measure(transform(image, np.array([1., 0., 0., 0.]),
                                        slope_limit=1.25, offset_limit=.35),
                              epsilon=0, scale_floor=.001, tensor_limit=8)
    assert before['scale_floored'] and after['scale_floored']
    assert not np.allclose(before['tensor'], after['tensor'])

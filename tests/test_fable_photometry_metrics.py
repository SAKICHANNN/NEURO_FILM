import numpy as np
import pytest

from src.eval.fable_photometry_metrics import (
    analytic_predict, analytic_prior, detail_loss, donor_error_decomposition,
    exact_sign_test, parameter_errors, image_histograms, histogram_errors, histogram_detail_loss,
)
from src.eval.fable_reference_photometry import quantize8, transform


def test_analytic_recovers_affine_logits_and_survives_quantization():
    codes = np.arange(32, 224, dtype=np.float64).reshape(8, 8, 3) / 255
    prior = analytic_prior([codes] * 6)
    target = np.array([.4, -.2, .3, .1])
    reference = transform(codes, target, slope_limit=1.25, offset_limit=.35)
    estimated = analytic_predict(reference, prior, 1.25, .35)
    np.testing.assert_allclose(estimated, target, atol=1e-13)
    quantized = analytic_predict(quantize8(reference), prior, 1.25, .35)
    np.testing.assert_allclose(quantized, target, atol=.035)


def test_analytic_endpoint_clipping_and_degenerate_prior():
    flat = np.full((2, 2, 3), .5)
    prior = analytic_prior([flat])
    assert prior["degenerate"] and prior["denominator"] == 0
    np.testing.assert_array_equal(analytic_predict(np.zeros_like(flat), prior, 1.25, .35), np.zeros(4))
    prior = analytic_prior([np.linspace(0, 1, 12).reshape(2, 2, 3)])
    estimated = analytic_predict(np.ones_like(flat), prior, 1.25, .35)
    assert np.isfinite(estimated).all() and np.all(np.abs(estimated) <= 1)
    assert estimated[0] == pytest.approx(-1, abs=1e-15)
    np.testing.assert_array_equal(estimated[1:], np.ones(3))


def test_prior_averages_donor_statistics_instead_of_pooling_pixels():
    prior = analytic_prior([np.full((1, 1, 3), .25), np.full((3, 4, 3), .75)])
    np.testing.assert_allclose(prior["mean_median"], 0., atol=1e-15)
    assert prior["degenerate"]


def test_exact_sign_test_12_of_16_and_ties():
    values = np.r_[np.ones(12), -np.ones(4)]
    result = exact_sign_test(values)
    assert result["pvalue"] == 2517 / 65536
    assert result["passed"] and result["nontied"] == 16
    tied = exact_sign_test(np.r_[values, 0., 0.])
    assert tied["pvalue"] == result["pvalue"] and tied["ties"] == 2
    assert not exact_sign_test(np.zeros(16))["passed"]
    assert not exact_sign_test(np.r_[np.ones(12), np.full(4, -100.)])["passed"]


def test_detail_counts_only_new_endpoints_and_lost_source_differences():
    source = np.repeat(np.array([[0., 1., 2.], [0., 2., 2.]])[..., None], 3, axis=2) / 255
    prediction = np.zeros_like(source)
    result = detail_loss(source, prediction)
    assert result["newly_saturated"] == {
        "count": 12, "total": 18, "source_eligible": 12,
        "fraction": 2 / 3, "fraction_of_eligible": 1.,
    }
    assert result["adjacency"]["horizontal"]["count"] == 9
    assert result["adjacency"]["horizontal"]["total"] == 12
    assert result["adjacency"]["vertical"]["count"] == 3
    assert result["adjacency"]["vertical"]["total"] == 9
    assert result["adjacency"]["combined"]["count"] == 12
    assert result["adjacency"]["combined"]["total"] == 21
    unchanged = detail_loss(source, source)
    assert unchanged["newly_saturated"]["count"] == 0
    assert unchanged["adjacency"]["combined"]["count"] == 0
    empty = detail_loss(np.ones((1, 1, 3)), np.ones((1, 1, 3)))
    assert empty["adjacency"]["combined"]["fraction_of_eligible"] is None


def test_parameter_losses_and_donor_bias_variance_identity():
    targets = np.zeros((2, 4))
    predictions = np.stack([np.ones_like(targets), np.full_like(targets, 3.)])
    parts = donor_error_decomposition(predictions, targets)
    np.testing.assert_array_equal(parts["mean_prediction_error"], [4., 4.])
    np.testing.assert_array_equal(parts["between_donor_variance"], [1., 1.])
    np.testing.assert_allclose(parts["donor_mean_error"],
                               parts["mean_prediction_error"] + parts["between_donor_variance"])
    losses = parameter_errors(predictions[0], targets)
    assert losses["mean_mse"] == 1
    np.testing.assert_array_equal(losses["signed_bias"], np.ones(4))
    with pytest.raises(ValueError):
        exact_sign_test([float("nan")])


@pytest.mark.parametrize("shape", [(7, 9, 3), (1, 1, 3), (1, 17, 3)])
def test_histogram_errors_and_detail_equal_native_brute_force(shape):
    rng = np.random.default_rng(438)
    codes = rng.integers(0, 256, shape, dtype=np.uint8)
    source = codes.astype(np.float64) / 255
    table_input = np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)
    predicted_table = transform(table_input, np.array([.8, -.9, .3, .4]), slope_limit=1.25, offset_limit=.35)
    target_table = transform(table_input, np.array([-.6, .4, -.7, .5]), slope_limit=1.25, offset_limit=.35)
    prediction = np.stack([predicted_table[codes[..., c], c] for c in range(3)], axis=-1)
    target = np.stack([target_table[codes[..., c], c] for c in range(3)], axis=-1)
    histograms = image_histograms(codes)
    errors = histogram_errors(predicted_table, target_table, histograms["counts"])
    assert errors["float_mse"] == pytest.approx(np.mean((prediction - target) ** 2), rel=1e-14)
    assert errors["code_mse"] == pytest.approx(np.mean((np.floor(prediction * 255 + .5)
                                                       - np.floor(target * 255 + .5)) ** 2), rel=1e-14)
    assert histogram_detail_loss(predicted_table, histograms) == detail_loss(source, prediction)
    assert histograms["counts"].sum() == codes.size
    assert histograms["adjacency"][0].sum() == shape[0] * (shape[1] - 1) * 3
    assert histograms["adjacency"][1].sum() == (shape[0] - 1) * shape[1] * 3

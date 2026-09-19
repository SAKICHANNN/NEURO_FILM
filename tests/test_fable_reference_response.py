import json
from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime

import src.color_match.research.fable_reference_response as response


CONFIG = json.loads((Path(__file__).resolve().parents[1]/"configs/fable_real_reference_conditional_v1.json").read_text())


def test_icc_creation_timestamp_is_not_color_transform():
    from scripts.run_fable_real_reference_conditional import profile_colorimetry_matches
    from src.color_match.srgb_icc_profile import srgb_icc_profile_v1
    profile = bytearray(srgb_icc_profile_v1())
    profile[24:36] = bytes(range(12))
    assert profile_colorimetry_matches(bytes(profile))
    profile[-1] ^= 1
    assert not profile_colorimetry_matches(bytes(profile))


def test_conversion_identity_and_tiled_render():
    rng = np.random.default_rng(84)
    encoded = rng.uniform(0, 1, (31, 29, 3))
    encoded[0, :8] = np.asarray([[r, g, b] for r in (0, 1) for g in (0, 1) for b in (0, 1)])
    lab = response.encoded_to_oklab(encoded)
    linear = response.oklab_to_linear_srgb(lab)
    restored = response.linear_srgb_to_encoded(np.clip(linear, 0, 1))
    assert np.max(np.abs(encoded-restored)) < 2e-5
    alpha = np.linspace(0, 1, 9)
    coefficients = np.r_[np.ones(36), np.zeros(42)]
    unchanged = response.render_lab(lab, alpha, coefficients, CONFIG)
    np.testing.assert_allclose(unchanged, lab, atol=1e-12)
    tiled, _ = response.render_image(encoded, alpha, coefficients, CONFIG, rows=7)
    full, _ = response.render_image(encoded, alpha, coefficients, CONFIG, rows=31)
    np.testing.assert_allclose(tiled, full, rtol=0, atol=2e-7)


def test_first_gamut_boundary_dense_independent_rays():
    lightness, hue = np.meshgrid([1e-5, .001, .05, .2, .5, .8, .99, .99999], np.linspace(-np.pi, np.pi, 73), indexing="ij")
    lightness, hue = lightness.ravel(), hue.ravel()
    directions = np.column_stack([np.cos(hue), np.sin(hue)])
    boundary = response.gamut_boundary(lightness, directions, 24)
    assert np.isfinite(boundary).all() and np.all(boundary >= 0)
    fractions = np.linspace(0, .999, 71)
    values = np.concatenate([np.broadcast_to(lightness[:, None, None], (len(lightness), len(fractions), 1)), directions[:, None, :]*boundary[:, None, None]*fractions[None, :, None]], axis=-1)
    rgb = response.oklab_to_linear_srgb(values)
    assert rgb.min() > -1e-9 and rgb.max() < 1+1e-9
    beyond = np.column_stack([lightness, directions*(boundary+2e-6)[:, None]])
    rgb = response.oklab_to_linear_srgb(beyond)
    assert np.all(np.any((rgb < 0) | (rgb > 1), axis=1))


def test_gray_hue_ramps_and_soft_knee_derivative():
    levels = np.linspace(0, 1, 4097)
    gray = np.column_stack([levels, np.zeros((len(levels), 2))])
    linear, _ = response.soft_gamut(gray, CONFIG)
    assert np.min(np.diff(linear, axis=0)) >= -1e-8
    hue = np.linspace(-np.pi, np.pi, 8193)
    color = np.column_stack([np.full(len(hue), .65), .12*np.cos(hue), .12*np.sin(hue)])
    coefficients = np.r_[1+.1*np.sin(np.arange(36)), .02*np.cos(np.arange(36)), np.zeros(6)]
    rendered = response.render_lab(color, np.linspace(0, 1, 9), coefficients, CONFIG)
    mapped, _ = response.soft_gamut(rendered, CONFIG)
    assert np.max(np.abs(np.diff(mapped, axis=0))) < .005
    np.testing.assert_allclose(mapped[0], mapped[-1], atol=1e-6)
    direction = np.array([[1., 0.]])
    maximum = response.gamut_boundary(np.array([.5]), direction, 24)[0]
    knee = .95*maximum
    epsilon = 1e-6
    ramp = np.column_stack([np.full(3, .5), [knee-epsilon, knee, knee+epsilon], np.zeros(3)])
    output, _ = response.soft_gamut(ramp, CONFIG)
    lab = response.encoded_to_oklab(response.linear_srgb_to_encoded(output.reshape(-1, 1, 3))).reshape(-1, 3)
    derivative = np.diff(lab[:, 1])/epsilon
    assert np.all(np.abs(derivative-1) < .05)


def test_tied_weighted_ranks_permutation():
    values = np.array([.1, .2, .1, .4])
    weights = np.array([1., 3., 2., 4.])
    knots, mid, ranks = response._weighted_distribution(values, weights)
    np.testing.assert_allclose(ranks, [.15, .45, .15, .8])
    order = np.array([3, 1, 0, 2])
    _, _, permuted = response._weighted_distribution(values[order], weights[order])
    np.testing.assert_allclose(permuted, ranks[order])


def test_solver_analytic_gradients_constraints_and_nested_objective(monkeypatch):
    original = response.minimize
    checked = []
    def verify(fun, x0, **kwargs):
        finite = approx_fprime(x0, fun, 1e-6)
        analytic = kwargs["jac"](x0)
        np.testing.assert_allclose(analytic, finite, atol=.003, rtol=.001)
        checked.append(True)
        return original(fun, x0, **kwargs)
    monkeypatch.setattr(response, "minimize", verify)
    rng = np.random.default_rng(172)
    encoded = rng.uniform(.03, .95, (8192, 1, 3))
    source = response.encoded_to_oklab(encoded).reshape(-1, 3)
    reference = response.encoded_to_oklab(np.clip(encoded*[1.02, .98, .97], 0, 1)).reshape(-1, 3)
    alpha, tone = response.fit_tone(source, reference, CONFIG)
    assert tone["valid"], tone
    derivative = 8*np.diff(alpha)
    assert derivative.min() >= .35-1e-7 and derivative.max() <= 2.2+1e-7
    targets = response.conditional_targets(source, reference, alpha, CONFIG)
    simple, basic = response.fit_color(targets, CONFIG, simple=True)
    full, extended = response.fit_color(targets, CONFIG, simple=False, initial=simple)
    assert basic["valid"] and extended["valid"], (basic, extended)
    assert extended["objective"] <= basic["objective"]+1e-6
    assert len(checked) == 3
    assert np.max(np.abs(full[36:72])-full[:36]*np.tan(np.deg2rad(8))) < 1e-7

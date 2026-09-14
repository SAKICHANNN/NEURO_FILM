import copy

import numpy as np
import pytest

from src.eval import fable_paired_contrast as contrast


def fixture():
    rng = np.random.default_rng(913)
    draws = rng.uniform(-1, 1, (16, 4))
    targets = np.tile(np.stack([draws, -draws], axis=1).reshape(32, 4), (6, 1))
    donors = np.repeat(np.arange(6), 32)
    pairs = np.tile(np.repeat(np.arange(16), 2), 6)
    identities = [rng.normal(size=(6, 3)), rng.normal(size=(6, 5))]
    after = [v[donors] + targets @ rng.normal(size=(4, v.shape[1])) for v in identities]
    return after, identities, targets, donors, pairs


def test_normalizer_has_unique_identity_rows_and_no_second_contrast_scaling():
    after, identities, targets, donors, _ = fixture()
    model = contrast.fit_contrast_decoder(after, identities, targets, donors, np.arange(6), .2)
    assert model['normalization_rows'] == 198 and model['regression_rows'] == 192
    for standard, edited, original in zip(model['standards'], after, identities):
        np.testing.assert_array_equal(standard['mean'], np.concatenate([edited, original]).mean(0))
    z = contrast._standardize(after, model['standards']) - contrast._standardize(identities, model['standards'])[donors]
    centered = z - z.mean(0)
    expected = np.linalg.solve(centered.T @ centered / 192 + .2 * np.eye(centered.shape[1]),
                               centered.T @ (targets - targets.mean(0)) / 192)
    np.testing.assert_allclose(model['coefficients'], expected, atol=1e-12)
    np.testing.assert_allclose(model['intercept'], targets.mean(0) - z.mean(0) @ expected, atol=1e-12)
    with pytest.raises(ValueError):
        contrast.fit_contrast_decoder(after, [v[donors] for v in identities], targets, donors, donors, .2)


def test_cv_normalization_excludes_both_validation_axes_and_identity_donors(monkeypatch):
    after, identities, targets, donors, pairs = fixture()
    calls = []
    original = contrast.fit_contrast_decoder

    def capture(a, i, y, ds, ids, penalty):
        fitted = original(a, i, y, ds, ids, penalty)
        calls.append((a, i, ds.copy(), ids.copy(), fitted))
        return fitted

    monkeypatch.setattr(contrast, 'fit_contrast_decoder', capture)
    result = contrast.select_paired_contrast(after, identities, targets, donors, pairs, [.1])
    assert len(calls) == 7
    for call, fold in zip(calls[:6], result['folds'], strict=True):
        a, identity, ds, ids, fitted = call
        assert fitted['normalization_rows'] == 68 and fitted['identity_rows'] == 4
        assert not set(ids) & set(fold['held_donors'])
        assert not set(pairs[fold['train']]) & set(fold['held_pairs'])
        for b in range(2):
            np.testing.assert_array_equal(a[b], after[b][fold['train']])
            np.testing.assert_array_equal(identity[b], identities[b][ids])
            np.testing.assert_array_equal(fitted['standards'][b]['mean'], np.concatenate([a[b], identity[b]]).mean(0))
    assert calls[-1][-1]['normalization_rows'] == 198


def test_after_only_interface_does_not_consume_evaluation_before_or_donor_identity():
    after, identities, targets, donors, _ = fixture()
    model = contrast.fit_contrast_decoder(after, identities, targets, donors, np.arange(6), .01)
    before = [v[donors].copy() for v in identities]
    primary = contrast.predict_after_raw_blocks(model, after)
    paired = contrast.predict_paired_control_raw_blocks(model, after, before)
    changed = [v + 100 for v in before]
    other = contrast.predict_paired_control_raw_blocks(model, after, changed)
    assert not np.array_equal(paired['raw'], other['raw'])
    model_copy = copy.deepcopy(model)
    model_copy['identity_donor_ids'] = np.arange(6)[::-1]
    np.testing.assert_array_equal(primary['raw'], contrast.predict_after_raw_blocks(model_copy, after)['raw'])
    with pytest.raises(TypeError):
        contrast.predict_after_raw_blocks(model, after, changed)


def test_exact_cv_ties_larger_lambda_and_all_constant_blocks():
    _, _, targets, donors, pairs = fixture()
    result = contrast.select_paired_contrast([np.ones((192, 2))] * 2, [np.ones((6, 2))] * 2,
                                             targets, donors, pairs, [100., .1, 1.])
    assert result['selected_ridge'] == 100.
    assert result['model']['coefficients'].shape == (0, 4)
    np.testing.assert_allclose(result['paired_oof_u'], 0., atol=1e-15)


def test_sign_test_keeps_ties_as_nonwins():
    result = contrast.strict_win_sign_test_16(np.r_[np.ones(12), np.zeros(4)])
    assert result['blocks'] == 16 and result['nonwins'] == 4 and result['ties'] == 4
    assert result['pvalue'] == 2517 / 65536 and result['passed']
    assert not contrast.strict_win_sign_test_16(np.r_[np.ones(11), np.zeros(5)])['passed']
    assert not contrast.strict_win_sign_test_16(np.r_[np.ones(12), -np.ones(4) * 100])['passed']
    with pytest.raises(ValueError):
        contrast.strict_win_sign_test_16(np.ones(12))


def test_engineering_gate_zero_oracle_strength_and_per_group_failures():
    donors, queries = [0, 0, 1, 1], [0, 1, 0, 1]
    result = contrast.joint_code_engineering_gate([64.] * 4, [16.] * 4, [16.] * 4, donors, queries)
    assert result['passed']
    assert not contrast.joint_code_engineering_gate([0.] * 4, [0.] * 4, [16.] * 4, donors, queries)['passed']
    assert not contrast.joint_code_engineering_gate([64.] * 4, [0, 0, 20, 20], [64.] * 4, donors, queries)['passed']
    assert not contrast.joint_code_engineering_gate([64.] * 4, [0.] * 4, [0, 32, 0, 32], donors, queries)['passed']

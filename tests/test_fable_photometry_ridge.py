import numpy as np
import pytest

from src.eval import fable_photometry_ridge as ridge


def crossed_data():
    rng = np.random.default_rng(27)
    draws = rng.uniform(-1, 1, (16, 4))
    donors = np.repeat(np.arange(6), 32)
    pairs = np.tile(np.repeat(np.arange(16), 2), 6)
    targets = np.tile(np.stack([draws, -draws], axis=1).reshape(32, 4), (6, 1))
    return donors, pairs, targets


def test_crossed_folds_exclude_both_axes_and_cover_validation_once():
    donors, pairs, _ = crossed_data()
    folds = ridge.crossed_folds(donors, pairs)
    assert len(folds) == 6
    for fold in folds:
        train, validation = fold["train"], fold["validation"]
        assert len(train) == 64 and len(validation) == 32
        assert not set(donors[train]) & set(donors[validation])
        assert not set(pairs[train]) & set(pairs[validation])
    assert np.array_equal(np.sort(np.concatenate([f["validation"] for f in folds])), np.arange(192))
    with pytest.raises(ValueError):
        ridge.crossed_folds(donors[:-1], pairs[:-1])


def test_dual_matches_mean_loss_primal_and_unpenalized_intercept():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(17, 5))
    y = rng.normal(size=(17, 4)) + 3
    model = ridge.fit_head([x], y, .2)
    z = (x - x.mean(0)) / x.std(0)
    coefficients = np.linalg.solve(z.T @ z / len(z) + .2 * np.eye(5), z.T @ (y - y.mean(0)) / len(z))
    np.testing.assert_allclose(model["coefficients"], coefficients, atol=1e-12)
    predicted = ridge.predict_head(model, [x])
    np.testing.assert_allclose(predicted["raw"], z @ coefficients + y.mean(0), atol=1e-12)
    np.testing.assert_array_equal(predicted["u"], np.clip(predicted["raw"], -1, 1))
    assert predicted["clipping_frequency"] == predicted["clipped"].mean()


def test_constant_columns_dropped_and_block_scale_uses_active_dimension():
    x = np.column_stack([np.arange(9), np.arange(9) ** 2, np.full(9, .1)])
    y = np.zeros((9, 4))
    model = ridge.fit_head([x, np.ones((9, 3))], y, .1, True)
    np.testing.assert_array_equal(model["standards"][0]["active"], [True, True, False])
    assert model["standards"][0]["divisor"] == np.sqrt(2)
    assert model["standards"][1]["divisor"] == 1
    assert model["coefficients"].shape == (2, 4)
    changed = x.copy()
    changed[:, 2] = 1e8
    np.testing.assert_array_equal(ridge.predict_head(model, [changed, np.zeros((9, 3))])["raw"], y)


def test_cv_fits_normalization_only_on_each_fold(monkeypatch):
    donors, pairs, targets = crossed_data()
    stats = np.column_stack([np.arange(192), donors, pairs, donors < 2])
    deep = targets.copy()
    calls = []
    original = ridge.fit_head

    def capture(blocks, y, penalty, balanced=False):
        model = original(blocks, y, penalty, balanced)
        calls.append((blocks[0].copy(), model["standards"][0]))
        return model

    monkeypatch.setattr(ridge, "fit_head", capture)
    result = ridge.select_heads(stats, deep, targets, donors, pairs, [.1])
    for arm in range(2):
        for i, fold in enumerate(result["folds"]):
            values, standard = calls[arm * 7 + i]
            np.testing.assert_array_equal(values, stats[fold["train"]])
            np.testing.assert_array_equal(standard["mean"], values.mean(0))
            np.testing.assert_array_equal(standard["active"], np.ptp(values, axis=0) > 0)
    assert not calls[0][1]["active"][-1]
    assert calls[2][1]["active"][-1]


def test_exact_ties_choose_strongest_penalty_then_stats():
    donors, pairs, targets = crossed_data()
    zeros = np.zeros((192, 2))
    result = ridge.select_heads(zeros, zeros, targets, donors, pairs, [1., .01, 100.])
    assert result["primary"] == "stats"
    for arm in ("stats", "combined"):
        assert result["models"][arm]["ridge"] == 100.
        assert result["cv"][arm]["selected_index"] == 2


def test_pair_label_corruption_is_rejected():
    donors, pairs, targets = crossed_data()
    targets[0, 0] += .001
    with pytest.raises(ValueError, match="antithetic"):
        ridge.select_heads(np.zeros((192, 2)), np.zeros((192, 2)), targets, donors, pairs, [.1])

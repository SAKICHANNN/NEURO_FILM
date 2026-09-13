import numpy as np
from scipy import linalg


def fit_head(blocks: list[np.ndarray], targets: np.ndarray, ridge: float,
             block_balance: bool = False) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 4 or not np.isfinite(targets).all():
        raise ValueError("targets must be finite (N, 4)")
    if not np.isfinite(ridge) or ridge <= 0 or len(targets) == 0 or not blocks:
        raise ValueError("positive ridge, rows and feature blocks required")
    standards, matrices = [], []
    for block in blocks:
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 2 or len(values) != len(targets) or not np.isfinite(values).all():
            raise ValueError("feature blocks must be finite and row aligned")
        mean, scale = values.mean(0), values.std(0)
        active = (np.ptp(values, axis=0) > 0) & (scale > 0)
        divisor = np.sqrt(int(active.sum())) if block_balance and active.any() else 1.
        standards.append({"mean": mean, "scale": scale, "active": active,
                          "divisor": float(divisor), "dimension": values.shape[1]})
        matrices.append((values[:, active] - mean[active]) / scale[active] / divisor)
    matrix = np.concatenate(matrices, axis=1)
    feature_mean, target_mean = matrix.mean(0), targets.mean(0)
    centered = matrix - feature_mean
    # Mean squared loss requires n * lambda in the dual system.
    gram = centered @ centered.T + len(targets) * ridge * np.eye(len(targets))
    dual = linalg.cho_solve(linalg.cho_factor(gram, lower=True), targets - target_mean)
    coefficients = centered.T @ dual
    return {"standards": standards, "coefficients": coefficients,
            "intercept": target_mean - feature_mean @ coefficients,
            "ridge": float(ridge), "block_balance": bool(block_balance),
            "training_rows": len(targets)}


def predict_head(model: dict, blocks: list[np.ndarray]) -> dict:
    if len(blocks) != len(model["standards"]):
        raise ValueError("feature block count differs from fitted head")
    matrices = []
    rows = None
    for block, standard in zip(blocks, model["standards"], strict=True):
        values = np.asarray(block, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != standard["dimension"]
                or not np.isfinite(values).all() or (rows is not None and rows != len(values))):
            raise ValueError("prediction features must be finite, dimension matched and row aligned")
        rows = len(values)
        active = standard["active"]
        matrices.append((values[:, active] - standard["mean"][active])
                        / standard["scale"][active] / standard["divisor"])
    raw = np.concatenate(matrices, axis=1) @ model["coefficients"] + model["intercept"]
    clipped = (raw < -1.) | (raw > 1.)
    return {"raw": raw, "u": np.clip(raw, -1., 1.), "clipped": clipped,
            "clipping_frequency": float(clipped.mean()) if clipped.size else 0.,
            "row_clipping_frequency": float(clipped.any(1).mean()) if len(clipped) else 0.}


def crossed_folds(donor_ids: np.ndarray, pair_ids: np.ndarray) -> list[dict]:
    donors, pairs = np.asarray(donor_ids), np.asarray(pair_ids)
    if (donors.shape != (192,) or pairs.shape != (192,)
            or not np.array_equal(np.unique(donors), np.arange(6))
            or not np.array_equal(np.unique(pairs), np.arange(16))):
        raise ValueError("requires donors 0..5, pairs 0..15 and 192 crossed rows")
    if any(np.count_nonzero((donors == d) & (pairs == p)) != 2
           for d in range(6) for p in range(16)):
        raise ValueError("every donor/pair must have exactly two antithetic rows")
    folds = []
    for donor_group in range(3):
        held_donors = np.arange(2 * donor_group, 2 * donor_group + 2)
        for pair_group in range(2):
            held_pairs = np.arange(8 * pair_group, 8 * pair_group + 8)
            donor_held = np.isin(donors, held_donors)
            pair_held = np.isin(pairs, held_pairs)
            folds.append({"held_donors": held_donors, "held_pairs": held_pairs,
                          "train": np.flatnonzero(~donor_held & ~pair_held),
                          "validation": np.flatnonzero(donor_held & pair_held)})
    return folds


def select_heads(stats: np.ndarray, deep: np.ndarray, targets: np.ndarray,
                 donor_ids: np.ndarray, pair_ids: np.ndarray,
                 ridge_grid: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    stats, deep = np.asarray(stats, dtype=np.float64), np.asarray(deep, dtype=np.float64)
    grid = np.asarray(ridge_grid, dtype=np.float64)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or np.any(grid <= 0) or len(np.unique(grid)) != len(grid)):
        raise ValueError("ridge grid must contain distinct positive finite values")
    folds = crossed_folds(donor_ids, pair_ids)
    if targets.shape != (192, 4) or not np.isfinite(targets).all() or np.any(np.abs(targets) > 1):
        raise ValueError("normalized targets must be finite (192, 4) within [-1, 1]")
    if (stats.ndim != 2 or deep.ndim != 2 or len(stats) != 192 or len(deep) != 192
            or not np.isfinite(stats).all() or not np.isfinite(deep).all()):
        raise ValueError("both descriptor blocks must have 192 finite rows")
    donors, pairs = np.asarray(donor_ids), np.asarray(pair_ids)
    for pair in range(16):
        reference = targets[(donors == 0) & (pairs == pair)]
        if not np.array_equal(reference[0], -reference[1]):
            raise ValueError("pair targets must be exact antithetic vectors")
        for donor in range(1, 6):
            current = targets[(donors == donor) & (pairs == pair)]
            if not (np.array_equal(current, reference) or np.array_equal(current[::-1], reference)):
                raise ValueError("transformation targets must agree across donors")
    models, cv = {}, {}
    for name, blocks, balanced in (("stats", [stats], False), ("combined", [stats, deep], True)):
        errors = np.empty((len(grid), len(folds)))
        for column, fold in enumerate(folds):
            train, validation = fold["train"], fold["validation"]
            for row, ridge in enumerate(grid):
                model = fit_head([b[train] for b in blocks], targets[train], ridge, balanced)
                predicted = predict_head(model, [b[validation] for b in blocks])["u"]
                errors[row, column] = np.mean((predicted - targets[validation]) ** 2)
        means = errors.mean(1)
        selected = min(range(len(grid)), key=lambda i: (means[i], -grid[i]))
        models[name] = fit_head(blocks, targets, grid[selected], balanced)
        cv[name] = {"grid": grid.copy(), "fold_errors": errors, "mean_errors": means,
                    "selected_index": selected, "selected_error": float(means[selected]),
                    "selected_ridge": float(grid[selected])}
    primary = min(cv, key=lambda name: (cv[name]["selected_error"],
                                       -cv[name]["selected_ridge"], name != "stats"))
    return {"models": models, "primary": primary, "cv": cv, "folds": folds}

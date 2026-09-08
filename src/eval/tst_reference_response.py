import ast
import hashlib
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import brentq


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_vgg(root: Path):
    import torch
    from torch import nn

    base = root / "data/ai_models/nlut_feature_v1"
    source, weights = base / "net.py", base / "models/vgg_normalised.pth"
    assert digest(source) == "b3c6df287b20210b8d471431c18dfda52fe26079cf0a86632b6705a6a91b1e20"
    assert digest(weights) == "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "vgg" for t in n.targets)]
    assert len(nodes) == 1
    namespace = {"nn": nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - SHA-bound reviewed VGG declaration only
    model = namespace["vgg"]
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    expected = torch.tensor([[0., 0., 255.], [0., 255., 0.], [255., 0., 0.]])
    assert torch.equal(state["0.weight"].reshape(3, 3), expected)
    torch.testing.assert_close(state["0.bias"], torch.tensor([-103.939, -116.779, -123.680]))
    return model[:31].eval().requires_grad_(False)


def descriptors(image, model):
    import torch
    from torch.nn import functional

    assert image.ndim == 3 and image.shape[-1] == 3
    assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)[None]
    tensor = functional.interpolate(tensor, (128, 128), mode="bilinear", align_corners=False, antialias=True)
    thumbnail = tensor[0].permute(1, 2, 0).numpy().astype(np.float64)
    rgb = thumbnail.reshape(-1, 3)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    luminance = linear @ [.2126, .7152, .0722]
    summaries = []
    for values in [rgb[:, 0], rgb[:, 1], rgb[:, 2], luminance]:
        summaries.extend([values.mean(), values.std(ddof=0), *np.quantile(values, [.05, .25, .5, .75, .95])])
    summaries.extend([np.any(rgb <= 1/255, axis=1).mean(), np.any(rgb >= 254/255, axis=1).mean()])
    features = []
    with torch.inference_mode():
        for index, layer in enumerate(model):
            tensor = layer(tensor)
            if index in (3, 10, 17, 30):
                features.extend([tensor.mean((2, 3)).flatten(), tensor.std((2, 3), correction=0).flatten()])
        features.append(functional.adaptive_avg_pool2d(tensor, (2, 2)).flatten())
    stats, learned = np.asarray(summaries), torch.cat(features).numpy().astype(np.float64)
    assert stats.shape == (30,) and learned.shape == (3968,)
    assert np.isfinite(stats).all() and np.isfinite(learned).all()
    return stats, learned


def fit_projection(blocks, dimensions=6):
    standards, matrices = [], []
    for values in blocks:
        values = np.asarray(values, dtype=np.float64)
        mean, std = values.mean(0), values.std(0)
        active = std >= 1e-6
        scale = np.where(active, std, 1.)
        standards.append({"mean": mean, "scale": scale, "active": active, "dimension": values.shape[1]})
        matrices.append((values - mean) / scale * active / np.sqrt(values.shape[1]))
    matrix = np.concatenate(matrices, axis=1)
    center = matrix.mean(0)
    _, singular, vt = np.linalg.svd(matrix - center, full_matrices=False)
    cutoff = max(matrix.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.)
    rank = min(dimensions, int(np.sum(singular > cutoff)))
    components = vt[:rank].copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    scores = (matrix - center) @ components.T
    mean, scale = scores.mean(0), scores.std(0)
    scale = np.where(scale >= 1e-6, scale, 1.)
    return {"standards": standards, "center": center, "components": components,
            "score_mean": mean, "score_scale": scale, "dimensions": dimensions}


def project(model, blocks):
    matrices = [(np.asarray(v) - s["mean"]) / s["scale"] * s["active"] / np.sqrt(s["dimension"])
                for v, s in zip(blocks, model["standards"], strict=True)]
    scores = (np.concatenate(matrices, axis=1) - model["center"]) @ model["components"].T
    scores = np.clip((scores - model["score_mean"]) / model["score_scale"], -3, 3)
    return np.pad(scores, ((0, 0), (0, model["dimensions"] - scores.shape[1])))


def lattice_design(rgb, dimension=7):
    rgb = np.asarray(rgb, dtype=np.float64)
    assert rgb.ndim == 2 and rgb.shape[1] == 3 and np.isfinite(rgb).all()
    assert rgb.min() >= 0 and rgb.max() <= 1
    position = rgb * (dimension - 1)
    lower = np.minimum(position.astype(np.int64), dimension - 2)
    frac = position - lower
    columns, weights = [], []
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                offset = np.array([r, g, b])
                columns.append((lower[:, 0]+r)*dimension**2 + (lower[:, 1]+g)*dimension + lower[:, 2]+b)
                weights.append(np.prod(np.where(offset, frac, 1-frac), axis=1))
    return sparse.csr_matrix((np.stack(weights, 1).ravel(),
                              (np.repeat(np.arange(len(rgb)), 8), np.stack(columns, 1).ravel())),
                             shape=(len(rgb), dimension**3))


def second_derivatives(dimension=7):
    d = dimension
    one = sparse.diags([np.ones(d-2), -2*np.ones(d-2), np.ones(d-2)], [0, 1, 2], shape=(d-2, d))
    eye = sparse.eye(d)
    return sparse.vstack([sparse.kron(sparse.kron(one, eye), eye),
                          sparse.kron(sparse.kron(eye, one), eye),
                          sparse.kron(sparse.kron(eye, eye), one)], format="csr") * (d-1)**2


def scoring_folds(height, width, source_id, seed=20260908, limit=8192):
    y, x = np.indices((height, width))
    parity = (x//64 + y//64) % 2
    salt = int(hashlib.sha256(source_id.encode()).hexdigest()[:16], 16)
    folds = []
    for fold in (0, 1):
        pool = np.flatnonzero(parity == fold)
        assert len(pool) > 0
        rng = np.random.default_rng(seed + salt + fold)
        folds.append(np.sort(rng.choice(pool, min(limit, len(pool)), replace=False)))
    assert not np.intersect1d(*folds).size
    return folds


def teacher_factor(source_fit, dimension=7, smoothness=1e-6, residual=1e-4):
    design = lattice_design(source_fit, dimension)
    derivatives = second_derivatives(dimension)
    normal = (design.T @ design / len(source_fit)
              + smoothness * (derivatives.T @ derivatives) / derivatives.shape[0]).toarray()
    normal += np.eye(dimension**3) * residual / dimension**3
    factor = linalg.cho_factor(normal, lower=True)
    return {"design": design, "normal": normal, "factor": factor, "source": source_fit}


def teacher_solve(factor, target_fit):
    rhs = factor["design"].T @ (target_fit-factor["source"]) / len(target_fit)
    values = linalg.cho_solve(factor["factor"], rhs)
    residual = np.linalg.norm(factor["normal"] @ values-rhs) / max(np.linalg.norm(rhs), 1e-15)
    assert np.isfinite(values).all() and residual < 1e-8
    return values, float(residual)


def render(source, values, dimension=7):
    return np.clip(source + lattice_design(source, dimension) @ values, 0, 1)


def paired_rows(features, labels, group_ids):
    rows, targets, weights = [], [], []
    for group in dict.fromkeys(group_ids):
        indices = np.flatnonzero(np.asarray(group_ids) == group)
        assert len(indices) in (1, 2)
        if len(indices) == 1:
            rows.append(features[indices[0]])
            targets.append(labels[indices[0]])
            weights.append(1.)
        else:
            a, b = indices
            rows.extend([(features[a]+features[b])/2, (features[b]-features[a])/2])
            targets.extend([(labels[a]+labels[b])/2, (labels[b]-labels[a])/2])
            weights.extend([1., 2.])
    return np.asarray(rows), np.asarray(targets), np.asarray(weights)


def centered_head_system(features, weights):
    root_weight = np.sqrt(weights)
    intercept = features[:, 0] * root_weight
    norm = intercept @ intercept
    assert norm > 0
    matrix = features[:, 1:] * root_weight[:, None]
    slope_mean = intercept @ matrix / norm
    centered = matrix - intercept[:, None] * slope_mean
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * max(singular[0], 1.)
    return {"intercept": intercept, "norm": norm, "slope_mean": slope_mean,
            "centered": centered, "singular": singular[singular > tolerance], "root_weight": root_weight}


def matched_heads(arm_features, labels, group_ids):
    systems, rows = {}, {}
    for arm, features in arm_features.items():
        transformed, targets, weights = paired_rows(features, labels, group_ids)
        rows[arm] = targets
        systems[arm] = centered_head_system(transformed, weights)
    minimum_rank = min(len(s["singular"]) for s in systems.values())
    target_df = 4. if minimum_rank > 4 else .8 * minimum_rank
    models = {}
    for arm, system in systems.items():
        centered, singular = system["centered"], system["singular"]
        weighted_targets = rows[arm] * system["root_weight"][:, None]
        target_mean = system["intercept"] @ weighted_targets / system["norm"]
        target_centered = weighted_targets - system["intercept"][:, None] * target_mean
        if target_df == 0:
            penalty, slopes = None, np.zeros((centered.shape[1], labels.shape[1]))
            achieved = 0.
        else:
            squared = singular**2
            def trace(penalty, squared=squared):
                return np.sum(squared/(squared+penalty))
            penalty = brentq(lambda value: trace(value)-target_df, 0., squared.max()*1e6, xtol=1e-12)
            slopes = linalg.solve(centered.T @ centered + np.eye(centered.shape[1])*penalty,
                                   centered.T @ target_centered, assume_a="pos")
            achieved = float(trace(penalty))
        intercept = target_mean - system["slope_mean"] @ slopes
        models[arm] = {"coefficients": np.vstack([intercept, slopes]), "ridge": penalty,
                       "target_df": target_df, "achieved_df": achieved, "rank": len(singular)}
    return models

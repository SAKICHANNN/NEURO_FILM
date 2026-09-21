import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def encode(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12).pow(1 / 2.4) - 0.055)


def luminance(x: torch.Tensor) -> torch.Tensor:
    return (decode(x) * x.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)


def load_image(path: Path, long_edge: int | None = None) -> torch.Tensor:
    with Image.open(path) as im:
        x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float()[None] / 255
    h, w = x.shape[-2:]
    if long_edge and max(h, w) > long_edge:
        shape = (round(h * long_edge / max(h, w)), round(w * long_edge / max(h, w)))
        x = encode(F.interpolate(decode(x), size=shape, mode="bilinear", align_corners=False, antialias=True))
    return x


def centers_for(x: torch.Tensor, spacing: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ny, nx = math.ceil(h / spacing), math.ceil(w / spacing)
    yy = (torch.arange(ny, device=x.device) + 0.5) * h / ny - 0.5
    xx = (torch.arange(nx, device=x.device) + 0.5) * w / nx - 0.5
    y, z = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((z.flatten(), y.flatten()), dim=1)


def patches(x: torch.Tensor, centers: torch.Tensor, width: int, samples: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    offsets = (torch.arange(samples, device=x.device, dtype=x.dtype) + 0.5) * width / samples - width / 2
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    coordinates = centers[:, None, None, :] + torch.stack((xx, yy), dim=-1)[None]
    grid = (coordinates + 0.5) * x.new_tensor([2 / w, 2 / h]) - 1
    return F.grid_sample(x.expand(len(centers), -1, -1, -1), grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def load_descriptor(root: Path, config: dict, device: torch.device) -> nn.Sequential:
    source, weights = root / config["vgg_source"], root / config["vgg_weights"]
    for path, key in ((source, "vgg_source_sha256"), (weights, "vgg_weights_sha256")):
        if sha256(path) != config[key]:
            raise ValueError(f"Pinned VGG mismatch: {path}")
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    assignment = next(n for n in module.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "vgg" for t in n.targets))
    namespace = {"nn": nn}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), namespace)
    model = namespace["vgg"]
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    expected = torch.tensor([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=torch.float32)
    if not torch.equal(state["0.weight"].reshape(3, 3), expected):
        raise ValueError("VGG embedded RGB-to-BGR255 transform mismatch")
    if not torch.allclose(state["0.bias"], torch.tensor([-103.939, -116.779, -123.680]), atol=1e-4, rtol=0):
        raise ValueError("VGG embedded mean subtraction mismatch")
    layers = list(model.children())
    for index in config["descriptor_layers"]:
        if not isinstance(layers[index], nn.ReLU) or not isinstance(layers[index + 1], nn.MaxPool2d):
            raise ValueError("Descriptor must use the last ReLU before its pool")
    return model[:max(config["descriptor_layers"]) + 1].eval().requires_grad_(False).to(device)


@torch.no_grad()
def descriptors(x: torch.Tensor, centers: torch.Tensor, model: nn.Sequential, config: dict) -> torch.Tensor:
    gray = encode(luminance(x))
    result = []
    for offset in range(0, len(centers), config["descriptor_batch"]):
        p = patches(gray, centers[offset:offset + config["descriptor_batch"]], config["descriptor_patch"], config["descriptor_patch"])
        quantiles = torch.quantile(p.flatten(1), p.new_tensor([0.1, 0.5, 0.9]), dim=1)
        spread = (quantiles[2] - quantiles[0]).clamp_min(config["grayscale_spread_floor"])
        z = (config["grayscale_center"] + config["grayscale_scale"] * (p - quantiles[1, :, None, None, None]) / spread[:, None, None, None]).clamp(0, 1).expand(-1, 3, -1, -1)
        features = []
        for index, layer in enumerate(model):
            z = layer(z)
            if index in config["descriptor_layers"]:
                features.append(F.normalize(F.adaptive_avg_pool2d(z, 2).flatten(1), dim=1))
        result.append(F.normalize(torch.cat(features, dim=1), dim=1))
    return torch.cat(result)


def projected_quantiles(x: torch.Tensor, centers: torch.Tensor, config: dict) -> torch.Tensor:
    p = patches(x, centers, config["statistic_patch"], config["statistic_samples"]).flatten(2)
    directions = F.normalize(x.new_tensor(config["projection_directions"]), dim=1)
    projected = torch.einsum("dc,ncs->nds", directions, p)
    return torch.quantile(projected, x.new_tensor(config["quantiles"]), dim=2).permute(1, 2, 0)


def reference_threshold(features: torch.Tensor, files: torch.Tensor, config: dict) -> float:
    distances = (1 - features @ features.T).clamp_min(0)
    distances.masked_fill_(files[:, None] == files[None, :], float("inf"))
    nearest = distances.min(1).values
    if not torch.isfinite(nearest).all():
        return 0.0
    return min(config["distance_cap"], float(torch.quantile(nearest, config["distance_quantile"])))


def match_references(source: torch.Tensor, reference: torch.Tensor, files: torch.Tensor, targets: torch.Tensor, tau: float, config: dict) -> dict:
    distances = (1 - source @ reference.T).clamp_min(0)
    reciprocal = distances.topk(min(config["reciprocal_k"], len(source)), dim=0, largest=False).indices
    records, selected, confidence, target = [], [], [], []
    for source_index in range(len(source)):
        candidates = []
        for file_index in files.unique(sorted=True).tolist():
            indices = torch.nonzero(files == file_index).flatten()
            reference_index = int(indices[distances[source_index, indices].argmin()])
            distance = float(distances[source_index, reference_index])
            if tau > 0 and distance < tau and bool((reciprocal[:, reference_index] == source_index).any()):
                candidates.append((distance, reference_index, file_index))
        candidates.sort()
        candidates = candidates[:config["maximum_reference_files"]]
        admitted = len(candidates) >= config["minimum_reference_files"]
        records.append({"source_index": source_index, "admitted": admitted, "candidates": [{"distance": d, "reference_patch_index": ri, "reference_file_index": fi} for d, ri, fi in candidates]})
        if admitted:
            selected.append(source_index)
            dmedian = float(np.median([r[0] for r in candidates]))
            confidence.append(len(candidates) / config["maximum_reference_files"] * math.exp(-(dmedian / tau) ** 2))
            target.append(torch.quantile(targets[[r[1] for r in candidates]], 0.5, dim=0))
    return {"records": records, "selected": torch.tensor(selected, dtype=torch.long, device=source.device), "confidence": source.new_tensor(confidence), "targets": torch.stack(target) if target else targets[:0], "tau": tau}


def target_permutation(centers: torch.Tensor) -> torch.Tensor:
    n = len(centers)
    indices = torch.arange(n, device=centers.device)
    if n < 2:
        return indices
    scores = [float((centers - centers.roll(k, 0)).square().sum()) for k in range(1, n)]
    return indices.roll(1 + int(np.argmax(scores)))


def spline_basis(coordinates: torch.Tensor, cells: int, extent: int) -> torch.Tensor:
    u = (coordinates + 0.5) * cells / extent
    lower = u.floor().long()
    t = u - lower
    weights = torch.stack(((1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6), dim=1)
    indices = (lower[:, None] + coordinates.new_tensor([-1, 0, 1, 2], dtype=torch.long)).clamp(0, cells)
    return coordinates.new_zeros((len(coordinates), cells + 1)).scatter_add(1, indices, weights)


def pixel_coordinates(shape: tuple[int, int], working_shape: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = shape
    wh, ww = working_shape
    return (torch.arange(h, device=device).float() + 0.5) * wh / h - 0.5, (torch.arange(w, device=device).float() + 0.5) * ww / w - 0.5


def support_field(shape: tuple[int, int], working_shape: tuple[int, int], centers: torch.Tensor, confidence: torch.Tensor, config: dict) -> torch.Tensor:
    yy, xx = pixel_coordinates(shape, working_shape, centers.device)
    result = centers.new_zeros(shape)
    for center, weight in zip(centers, confidence):
        ky = (1 - ((yy - center[1]) / config["support_radius"]).square()).clamp_min(0).pow(3)
        kx = (1 - ((xx - center[0]) / config["support_radius"]).square()).clamp_min(0).pow(3)
        result += weight * ky[:, None] * kx[None, :]
    return (1 - torch.exp(-config["support_exponent"] * result))[None, None]


def bounded_nodes(raw: torch.Tensor, config: dict) -> torch.Tensor:
    bias = config["bias_bound"] * raw[:, :3].tanh()
    matrix = raw[:, 3:].tanh().reshape(raw.shape[0], 3, 3, *raw.shape[-2:])
    matrix = config["matrix_row_l1_bound"] * matrix / matrix.abs().sum(2, keepdim=True).clamp_min(1)
    return torch.cat((bias, matrix.flatten(1, 2)), dim=1)


def interpolate_nodes(nodes: torch.Tensor, shape: tuple[int, int], working_shape: tuple[int, int]) -> torch.Tensor:
    if nodes.shape[-2:] == (1, 1):
        return nodes.expand(-1, -1, *shape)
    yy, xx = pixel_coordinates(shape, working_shape, nodes.device)
    wy = spline_basis(yy.to(nodes.dtype), nodes.shape[-2] - 1, working_shape[0])
    wx = spline_basis(xx.to(nodes.dtype), nodes.shape[-1] - 1, working_shape[1])
    return torch.einsum("hi,bcij,wj->bchw", wy, nodes, wx)


def apply_field(x: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    matrix = field[:, 3:].reshape(x.shape[0], 3, 3, *x.shape[-2:])
    delta = field[:, :3] + (matrix * (2 * x[:, None] - 1)).sum(2)
    numerator = x * delta.exp()
    return numerator / (1 - x + numerator)


def edge_guard(x: torch.Tensor, y: torch.Tensor, config: dict) -> torch.Tensor:
    before, after = encode(luminance(x)), encode(luminance(y))
    terms = []
    low, high = config["edge_gain_limits"]
    for dim in (-1, -2):
        dx, dy = before.diff(dim=dim), after.diff(dim=dim)
        mask = dx.abs() >= config["edge_threshold"]
        if bool(mask.any()):
            gain = dy[mask] / dx[mask]
            terms.append((F.relu(low - gain).square() + F.relu(gain - high).square()).mean())
    return torch.stack(terms).mean() if terms else y.sum() * 0


def objective(x: torch.Tensor, y: torch.Tensor, nodes: torch.Tensor, centers: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    residual = (projected_quantiles(y, centers, config) - target) / config["quantile_scale"]
    style = (residual.square().mean((1, 2)) * confidence).sum() / confidence.sum().clamp_min(1e-8)
    normalized = nodes / nodes.new_tensor([config["bias_bound"]] * 3 + [config["matrix_row_l1_bound"]] * 9)[None, :, None, None]
    coefficient = normalized.square().mean()
    adjacent = [normalized.diff(dim=dim).square().mean() for dim in (-1, -2) if normalized.shape[dim] > 1]
    spatial = torch.stack(adjacent).mean() if adjacent else coefficient * 0
    edge = edge_guard(x, y, config)
    source_y, output_y = luminance(x), luminance(y)
    shadow_mask = (encode(source_y) < config["shadow_encoded_threshold"]) & (source_y > 1e-5)
    if bool(shadow_mask.any()):
        log_gain = (output_y[shadow_mask].clamp_min(1e-10) / source_y[shadow_mask]).log()
        low, high = map(math.log, config["shadow_linear_gain_limits"])
        shadow = (F.relu(low - log_gain).square() + F.relu(log_gain - high).square()).mean()
    else:
        shadow = y.sum() * 0
    terms = {"style": style, "coefficient": coefficient, "spatial": spatial, "edge": edge, "shadow": shadow}
    loss = style + sum(config["regularization"][key] * terms[key] for key in config["regularization"])
    return loss, {key: float(value.detach()) for key, value in terms.items()}


@torch.no_grad()
def image_metrics(x: torch.Tensor, y: torch.Tensor, support: torch.Tensor, config: dict) -> dict:
    before, after = encode(luminance(x)), encode(luminance(y))
    gain, flat_steps = [], []
    for dim in (-1, -2):
        dx, dy = before.diff(dim=dim), after.diff(dim=dim)
        selected = dx.abs() >= config["edge_threshold"]
        gain.append(dy[selected] / dx[selected])
        flat = dx.abs() <= 1 / 255
        flat_steps.append((dy[flat] - dx[flat]).abs())
    edge = torch.cat(gain)
    flat = torch.cat(flat_steps)
    source_y, output_y = luminance(x), luminance(y)
    shadow = (before < config["shadow_encoded_threshold"]) & (source_y > 1e-5)
    ratios = output_y[shadow] / source_y[shadow]
    low, high = config["edge_gain_limits"]
    slo, shi = config["shadow_linear_gain_limits"]
    unsupported = (support == 0).expand_as(x)
    return {"finite": bool(torch.isfinite(y).all()), "minimum": float(y.min()), "maximum": float(y.max()), "mean_absolute_rgb_change": float((y - x).abs().mean()), "p95_absolute_rgb_change": float(torch.quantile((y - x).abs().flatten(), 0.95)), "support_fraction_over_0_01": float((support > 0.01).float().mean()), "unsupported_max_rgb_change": float((y - x).abs()[unsupported].max()) if bool(unsupported.any()) else None, "edge_reversal_fraction": float((edge < 0).float().mean()) if edge.numel() else None, "edge_gain_outside_limits_fraction": float(((edge < low) | (edge > high)).float().mean()) if edge.numel() else None, "flat_extra_step_over_3_255_fraction": float((flat > 3 / 255).float().mean()) if flat.numel() else None, "shadow_gain_outside_limits_fraction": float(((ratios < slo) | (ratios > shi)).float().mean()) if ratios.numel() else None, "new_endpoint_fraction": float((((y <= 0) | (y >= 1)) & (x > 0) & (x < 1)).float().mean())}

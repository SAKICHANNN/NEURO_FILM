import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.color_match.research.local_reference_field_v1 import apply_field, bounded_nodes, edge_guard, encode, interpolate_nodes, luminance, pixel_coordinates, spline_basis


def token_mask(mask: torch.Tensor, stride: int = 14) -> torch.Tensor:
    return F.avg_pool2d(mask, stride, stride) >= 1


def dino_input(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    gray = (image * image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    gray = gray * mask + 0.5 * (1 - mask)
    return (gray.expand(-1, 3, -1, -1) - image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]


def valid_features(features: torch.Tensor, masks: torch.Tensor, boxes: list, size: int, stride: int = 14) -> tuple[torch.Tensor, torch.Tensor]:
    valid = token_mask(masks, stride).to(features.dtype)
    pooled = (features * valid).sum((-2, -1), keepdim=True) / valid.sum((-2, -1), keepdim=True).clamp_min(1)
    grids = []
    for i, (top, left, h, w) in enumerate(boxes):
        ys, xs = torch.where(valid[i, 0] > 0)
        if not len(ys):
            raise ValueError("No fully valid DINO tokens")
        unit = (torch.arange(size, device=features.device, dtype=features.dtype) + 0.5) / size
        yy = ((top + h * unit) / stride - 0.5).clamp(ys.min(), ys.max())
        xx = ((left + w * unit) / stride - 0.5).clamp(xs.min(), xs.max())
        yy = 2 * (yy + 0.5) / features.shape[-2] - 1
        xx = 2 * (xx + 0.5) / features.shape[-1] - 1
        grids.append(torch.stack((xx[None].expand(size, -1), yy[:, None].expand(-1, size)), -1))
    local = F.grid_sample(features, torch.stack(grids), align_corners=False, padding_mode="border")
    return local, pooled.expand(-1, -1, size, size)


class SemanticField(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        width = config["hidden_channels"]
        self.body = nn.Sequential(nn.Conv2d(2 * config["feature_dim"] + 3, width, 1), nn.LeakyReLU(0.2), nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.2))
        self.head = nn.Conv2d(width, 12, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor, masks: torch.Tensor, boxes: list, features: torch.Tensor) -> torch.Tensor:
        size = self.config["grid_size"]
        local, global_features = valid_features(features, masks, boxes, size, self.config["patch_stride"])
        rgb = torch.cat([F.interpolate(images[i:i + 1, :, t:t + h, l:l + w], (size, size), mode="bilinear", align_corners=False, antialias=True) for i, (t, l, h, w) in enumerate(boxes)])
        return bounded_nodes(self.head(self.body(torch.cat((local, global_features, rgb), 1))), self.config)


class PhotometricCritic(nn.Module):
    def __init__(self, config: dict, conditional: bool):
        super().__init__()
        width = config["critic_channels"]
        self.conditional = conditional
        self.body = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, width, 3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(width, width * 2, 3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)), nn.LeakyReLU(0.2),
        )
        self.score = nn.utils.spectral_norm(nn.Linear(width * 2, 1))
        self.embedding = nn.Embedding(config["clusters"], width * 2)

    def forward(self, patches: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
        features = self.body(patches).mean((-2, -1))
        conditioning = self.embedding(clusters) * float(self.conditional)
        return self.score(features).squeeze(1) + (features * conditioning).sum(1) / math.sqrt(features.shape[1])


def render_frames(images: torch.Tensor, nodes: torch.Tensor, boxes: list) -> torch.Tensor:
    outputs = []
    for i, (top, left, h, w) in enumerate(boxes):
        crop = images[i:i + 1, :, top:top + h, left:left + w]
        output = apply_field(crop, interpolate_nodes(nodes[i:i + 1], (h, w), (h, w)))
        outputs.append(F.pad(output, (left, images.shape[-1] - left - w, top, images.shape[-2] - top - h)))
    return torch.cat(outputs)


def render_native(image: torch.Tensor, nodes: torch.Tensor, tile: int) -> torch.Tensor:
    h, w = image.shape[-2:]
    yy, xx = pixel_coordinates((h, w), (h, w), nodes.device)
    wy = spline_basis(yy, nodes.shape[-2] - 1, h)
    wx = spline_basis(xx, nodes.shape[-1] - 1, w)
    result = torch.empty_like(image)
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            field = torch.einsum("hi,bcij,wj->bchw", wy[top:top + tile], nodes, wx[left:left + tile])
            result[..., top:top + tile, left:left + tile] = apply_field(image[..., top:top + tile, left:left + tile], field)
    return result


def photometric_patches(images: torch.Tensor, centers: torch.Tensor, config: dict) -> torch.Tensor:
    half = config["critic_patch"] // 2
    patches = torch.cat([images[i:i + 1, :, int(y) - half:int(y) + half, int(x) - half:int(x) + half] for i, (y, x) in enumerate(centers)])
    if patches.shape[-2:] != (2 * half, 2 * half):
        raise ValueError("Critic patch crossed image/padding boundary")
    kernel = patches.new_tensor([1, 4, 6, 4, 1]) / 16
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, -1, -1)
    blurred = F.conv2d(F.pad(patches, (2, 2, 2, 2), mode="reflect"), kernel, groups=3)
    return F.avg_pool2d(blurred, config["critic_downsample"])


def field_penalties(source: torch.Tensor, output: torch.Tensor, nodes: torch.Tensor, boxes: list, config: dict) -> dict:
    scaled = nodes / nodes.new_tensor([config["bias_bound"]] * 3 + [config["matrix_row_l1_bound"]] * 9)[None, :, None, None]
    terms = {"coefficient": scaled.square().mean(), "spatial": (scaled.diff(dim=-1).square().mean() + scaled.diff(dim=-2).square().mean()) / 2}
    edges, shadows = [], []
    for i, (t, l, h, w) in enumerate(boxes):
        x, y = source[i:i + 1, :, t:t + h, l:l + w], output[i:i + 1, :, t:t + h, l:l + w]
        edges.append(edge_guard(x, y, config))
        lx, ly = luminance(x), luminance(y)
        selected = (encode(lx) < config["shadow_encoded_threshold"]) & (lx > 1e-5)
        if selected.any():
            gain = (ly[selected].clamp_min(1e-10) / lx[selected]).log()
            low, high = map(math.log, config["shadow_linear_gain_limits"])
            shadows.append((F.relu(low - gain).square() + F.relu(gain - high).square()).mean())
        else:
            shadows.append(y.sum() * 0)
    terms["edge"], terms["shadow"] = torch.stack(edges).mean(), torch.stack(shadows).mean()
    return terms


def train_step(generator: SemanticField, critic: PhotometricCritic, generator_optimizer, critic_optimizer, images: torch.Tensor, masks: torch.Tensor, boxes: list, features: torch.Tensor, source_centers: torch.Tensor, real_images: torch.Tensor, real_centers: torch.Tensor, clusters: torch.Tensor, step: int, config: dict) -> dict:
    generator_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    nodes = generator(images, masks, boxes, features)
    output = render_frames(images, nodes, boxes)
    fake = photometric_patches(output, source_centers, config)
    real = photometric_patches(real_images, real_centers, config)
    use_r1 = step % config["r1_interval"] == 0
    real.requires_grad_(use_r1)
    real_score, fake_score = critic(real, clusters), critic(fake.detach(), clusters)
    discriminator = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
    r1 = real_score.sum() * 0
    if use_r1:
        gradient = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0]
        r1 = gradient.square().flatten(1).sum(1).mean()
        discriminator = discriminator + 0.5 * config["r1_weight"] * config["r1_interval"] * r1
    if not torch.isfinite(discriminator):
        raise FloatingPointError("Nonfinite discriminator loss")
    discriminator.backward()
    critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    critic_optimizer.step()
    critic.eval().requires_grad_(False)
    terms = field_penalties(images, output, nodes, boxes, config)
    terms["adversarial"] = -critic(fake, clusters).mean()
    loss = sum(config["loss_weights"][name] * value for name, value in terms.items())
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite generator loss")
    loss.backward()
    generator_norm = nn.utils.clip_grad_norm_(generator.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    generator_optimizer.step()
    critic.train().requires_grad_(True)
    return {"generator": float(loss.detach()), "critic": float(discriminator.detach()), "r1": float(r1.detach()), "generator_gradient_norm": float(generator_norm), "critic_gradient_norm": float(critic_norm), "real_score": float(real_score.detach().mean()), "fake_score": float(fake_score.detach().mean()), **{name: float(value.detach()) for name, value in terms.items()}}


def patch_candidates(boxes: list, side: int, stride: int, patch: int) -> np.ndarray:
    rows = []
    half = patch // 2
    for i, (top, left, h, w) in enumerate(boxes):
        for iy in range(side // stride):
            for ix in range(side // stride):
                y, x = stride * iy + stride // 2, stride * ix + stride // 2
                if y - half >= top and y + half <= top + h and x - half >= left and x + half <= left + w:
                    rows.append((i, iy, ix, y, x))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 5)


def balanced_schedule(source: np.ndarray, reference: np.ndarray, collections: np.ndarray, config: dict, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    source_pools, reference_pools, supported = {}, {}, {}
    for cluster in range(config["clusters"]):
        candidates = source[source[:, -1] == cluster]
        source_pools[cluster] = {int(i): candidates[candidates[:, 0] == i] for i in np.unique(candidates[:, 0])}
    for collection in range(3):
        supported[collection] = []
        for cluster in range(config["clusters"]):
            candidates = reference[(reference[:, -1] == cluster) & (collections[reference[:, 0]] == collection)]
            reference_pools[collection, cluster] = {int(i): candidates[candidates[:, 0] == i] for i in np.unique(candidates[:, 0])}
            if source_pools[cluster] and len(candidates):
                supported[collection].append(cluster)
        if not supported[collection]:
            raise ValueError(f"NO_SUPPORT collection {collection}; no threshold or cluster rescue")
    count = config["steps"] * config["batch_size"]
    if count % 3:
        raise ValueError("Exact equal collection exposure requires schedule size divisible by three")
    collection_order = np.tile(np.arange(3), count // 3)
    rng.shuffle(collection_order)
    sr, rr, cr = [], [], []
    for collection in collection_order:
        cluster = int(rng.choice(supported[int(collection)]))
        a, b = source_pools[cluster], reference_pools[int(collection), cluster]
        ai, bi = int(rng.choice(list(a))), int(rng.choice(list(b)))
        sr.append(a[ai][rng.integers(len(a[ai]))]); rr.append(b[bi][rng.integers(len(b[bi]))]); cr.append(cluster)
    shape = (config["steps"], config["batch_size"])
    result = {"source": np.asarray(sr).reshape(*shape, -1), "reference": np.asarray(rr).reshape(*shape, -1), "clusters": np.asarray(cr).reshape(shape), "collections": collection_order.reshape(shape)}
    report = {"supported_clusters_by_collection": supported, "collection_counts": np.bincount(collection_order, minlength=3).tolist(), "source_image_counts": {str(i): int((result["source"][..., 0] == i).sum()) for i in np.unique(source[:, 0])}, "reference_image_counts": {str(i): int((result["reference"][..., 0] == i).sum()) for i in range(len(collections))}, "source_cluster_counts": np.bincount(result["source"][..., -1].ravel(), minlength=config["clusters"]).tolist(), "reference_cluster_counts": np.bincount(result["reference"][..., -1].ravel(), minlength=config["clusters"]).tolist()}
    return result, report

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.color_match.research.local_reference_field_v1 import decode, encode, edge_guard, interpolate_nodes, luminance, pixel_coordinates, spline_basis
from src.models.semantic_film_adversarial_v1 import SemanticField, photometric_patches, valid_features


def color_matrices(like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_lms = torch.tensor([[.4122214708, .5363325363, .0514459929], [.2119034982, .6806995451, .1073969566], [.0883024619, .2817188376, .6299787005]], dtype=torch.float64, device=like.device)
    rgb_lms = rgb_lms / rgb_lms.sum(1, keepdim=True)
    lab_root = torch.tensor([[1., .3963377773761749, .2158037573099136], [1., -.1055613458156586, -.0638541728258133], [1., -.0894841775298119, -1.2914855480194092]], dtype=torch.float64, device=like.device)
    return rgb_lms, lab_root


def encoded_to_oklab(image: torch.Tensor, root_floor: float = 1e-12) -> torch.Tensor:
    rgb_lms, lab_root = color_matrices(image)
    image = image.double()
    lms = torch.einsum("ij,bjhw->bihw", rgb_lms, decode(image))
    roots = torch.where(lms > 0, lms.clamp_min(root_floor).pow(1 / 3), torch.zeros_like(lms))
    return torch.einsum("ij,bjhw->bihw", torch.linalg.inv(lab_root), roots)


def rgb_polynomial(lightness: torch.Tensor, chroma: torch.Tensor) -> tuple[torch.Tensor, ...]:
    lightness, chroma = lightness.double(), chroma.double()
    rgb_lms, lab_root = color_matrices(lightness)
    inverse = torch.linalg.inv(rgb_lms)
    direction = torch.einsum("ij,bjhw->bihw", lab_root[:, 1:], chroma)
    q0 = lightness.pow(3).expand(-1, 3, -1, -1)
    q1 = torch.einsum("ij,bjhw->bihw", inverse, 3 * lightness.square() * direction)
    q2 = torch.einsum("ij,bjhw->bihw", inverse, 3 * lightness * direction.square())
    q3 = torch.einsum("ij,bjhw->bihw", inverse, direction.pow(3))
    return q0, q1, q2, q3


def polynomial_rgb(q: tuple, scale: torch.Tensor) -> torch.Tensor:
    return q[0] + scale * (q[1] + scale * (q[2] + scale * q[3]))


class RadialGamutScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q0, q1, q2, q3, iterations, inward_fraction, derivative_floor, trigger_tolerance=0.):
        q = tuple(value.double() for value in (q0, q1, q2, q3))
        full = polynomial_rgb(q, torch.ones_like(q[0][:, :1]))
        outside = ((full < -trigger_tolerance) | (full > 1 + trigger_tolerance)).any(1, keepdim=True)
        a, b, c = 3 * q[3], 2 * q[2], q[1]
        discriminant = b.square() - 4 * a * c
        discriminant_root = discriminant.clamp_min(0).sqrt()
        sign = torch.where(b >= 0, 1., -1.)
        stable_q = -.5 * (b + sign * discriminant_root)
        quadratic = a.abs() > 1e-20
        safe_a = torch.where(quadratic, a, torch.ones_like(a))
        safe_q = torch.where(stable_q.abs() > 1e-20, stable_q, torch.ones_like(stable_q))
        root_a = stable_q / safe_a
        root_b = c / safe_q
        valid_a = quadratic & (discriminant >= 0)
        valid_b = valid_a & (stable_q.abs() > 1e-20)
        linear = (~quadratic) & (b.abs() > 1e-20)
        root_a = torch.where(linear, -c / torch.where(linear, b, torch.ones_like(b)), root_a)
        valid_a |= linear
        root_a = torch.where(valid_a & (root_a > 0) & (root_a < 1), root_a, torch.ones_like(root_a))
        root_b = torch.where(valid_b & (root_b > 0) & (root_b < 1), root_b, torch.ones_like(root_b))
        zero, one = torch.zeros_like(q[0][:, :1]), torch.ones_like(q[0][:, :1])
        knots = torch.cat((zero, root_a, root_b, one), 1).sort(1).values
        values = polynomial_rgb(tuple(value[:, None] for value in q), knots[:, :, None])
        infeasible = ((values < 0) | (values > 1)).any(2)
        positions = torch.arange(8, device=q0.device)[None, :, None, None]
        first = torch.where(infeasible, positions, 8).amin(1, keepdim=True).clamp_max(7)
        high = knots.gather(1, first)
        low = knots.gather(1, (first - 1).clamp_min(0))
        for _ in range(iterations):
            middle = (low + high) / 2
            rgb = polynomial_rgb(q, middle)
            feasible = ((rgb >= 0) & (rgb <= 1)).all(1, keepdim=True)
            low, high = torch.where(feasible, middle, low), torch.where(feasible, high, middle)
        boundary_rgb = polynomial_rgb(q, high)
        active = torch.maximum(-boundary_rgb, boundary_rgb - 1).argmax(1, keepdim=True)
        root = (low + high) / 2
        derivative = (q[1] + 2 * root * q[2] + 3 * root.square() * q[3]).gather(1, active)
        ctx.save_for_backward(root, active, derivative, outside)
        ctx.inward_fraction, ctx.derivative_floor = inward_fraction, derivative_floor
        ctx.input_dtype = q0.dtype
        return torch.where(outside, low * (1 - inward_fraction), one)

    @staticmethod
    def backward(ctx, gradient):
        root, active, derivative, outside = ctx.saved_tensors
        if bool((outside & (derivative.abs() < ctx.derivative_floor)).any()):
            raise FloatingPointError("Nonregular active gamut root: implicit derivative undefined; no detached fallback")
        safe_derivative = torch.where(outside, derivative, torch.ones_like(derivative))
        factor = -gradient.double() * (1 - ctx.inward_fraction) / safe_derivative
        factor = torch.where(outside, factor, torch.zeros_like(factor))
        result = []
        for power in range(4):
            value = torch.zeros((root.shape[0], 3, *root.shape[-2:]), dtype=root.dtype, device=root.device)
            result.append(value.scatter(1, active, factor * root.pow(power)).to(ctx.input_dtype))
        return *result, None, None, None, None


def tone_curve(lightness: torch.Tensor, bias: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    gain = (bias + slope * (2 * lightness - 1)).exp()
    numerator = lightness * gain
    return numerator / (1 - lightness + numerator)


def apply_chroma_field(image: torch.Tensor, field: torch.Tensor, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    # The whole colour path runs in float64 so the solver's inside/outside decision and the roundoff check see identical values.
    dtype, field = image.dtype, field.double()
    lab = encoded_to_oklab(image, config["cube_root_floor"])
    lightness = lab[:, :1].clamp(0, 1)
    output_lightness = tone_curve(lightness, field[:, :1], field[:, 1:2])
    sigma, angle = field[:, 2:3], field[:, 3:4]
    a, b = lab[:, 1:2], lab[:, 2:3]
    rotated = torch.cat((angle.cos() * a - angle.sin() * b, angle.sin() * a + angle.cos() * b), 1)
    chroma = sigma.exp() * rotated + 4 * lightness * (1 - lightness) * field[:, 4:6]
    q = rgb_polynomial(output_lightness, chroma)
    scale = RadialGamutScale.apply(*q, config["gamut_iterations"], config["gamut_inward_fraction"], config["gamut_derivative_floor"], config["gamut_roundoff_tolerance"])
    linear = polynomial_rgb(q, scale)
    tolerance = config["gamut_roundoff_tolerance"]
    if bool(((linear < -tolerance) | (linear > 1 + tolerance) | ~torch.isfinite(linear)).any()):
        raise FloatingPointError("Gamut solver exceeded numerical roundoff contract")
    output = encode(linear.clamp(0, 1)).to(dtype)
    endpoints = ((image == 0).all(1, keepdim=True) | (image == 1).all(1, keepdim=True))
    return torch.where(endpoints, image, output).clamp(0, 1), scale.to(dtype)


class ChromaField(SemanticField):
    def __init__(self, config: dict, arm: str):
        super().__init__(config)
        self.arm = arm
        self.head = nn.Conv2d(config["hidden_channels"], 6, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("bounds", torch.tensor(config["field_bounds"][arm])[None, :, None, None])

    def forward(self, images: torch.Tensor, masks: torch.Tensor, boxes: list, features: torch.Tensor) -> torch.Tensor:
        size = self.config["grid_size"]
        local, pooled = valid_features(features, masks, boxes, size, self.config["patch_stride"])
        rgb = torch.cat([F.interpolate(images[i:i + 1, :, t:t + h, l:l + w], (size, size), mode="bilinear", align_corners=False, antialias=True) for i, (t, l, h, w) in enumerate(boxes)])
        return self.bounds * self.head(self.body(torch.cat((local, pooled, rgb), 1))).tanh()


def render_frames(images: torch.Tensor, nodes: torch.Tensor, boxes: list, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, scales = [], []
    for i, (t, l, h, w) in enumerate(boxes):
        result, scale = apply_chroma_field(images[i:i + 1, :, t:t + h, l:l + w], interpolate_nodes(nodes[i:i + 1], (h, w), (h, w)), config)
        pad = (l, images.shape[-1] - l - w, t, images.shape[-2] - t - h)
        outputs.append(F.pad(result, pad)); scales.append(F.pad(scale, pad, value=1))
    return torch.cat(outputs), torch.cat(scales)


def render_native(image: torch.Tensor, nodes: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    h, w = image.shape[-2:]
    yy, xx = pixel_coordinates((h, w), (h, w), image.device)
    wy, wx = spline_basis(yy, nodes.shape[-2] - 1, h), spline_basis(xx, nodes.shape[-1] - 1, w)
    output = torch.empty_like(image)
    affected, total, smallest = 0, 0, 1.
    tile = config["native_tile_size"]
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            field = torch.einsum("hi,bcij,wj->bchw", wy[top:top + tile], nodes, wx[left:left + tile])
            result, scale = apply_chroma_field(image[..., top:top + tile, left:left + tile], field, config)
            output[..., top:top + tile, left:left + tile] = result
            affected += int((scale < .99999).sum()); total += scale.numel(); smallest = min(smallest, float(scale.min()))
    return output, {"gamut_scaled_fraction": affected / total, "minimum_chroma_retention": smallest}


def penalties(source: torch.Tensor, output: torch.Tensor, nodes: torch.Tensor, boxes: list, config: dict) -> dict:
    normalized = nodes / nodes.new_tensor(config["regularizer_units"])[None, :, None, None]
    terms = {"coefficient": normalized.square().mean(), "spatial": (normalized.diff(dim=-1).square().mean() + normalized.diff(dim=-2).square().mean()) / 2}
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


def train_step(generator, critic, go, co, images, masks, boxes, features, source_centers, real_images, real_centers, clusters, step, config):
    go.zero_grad(set_to_none=True); co.zero_grad(set_to_none=True)
    nodes = generator(images, masks, boxes, features)
    output, scale = render_frames(images, nodes, boxes, config)
    fake, real = photometric_patches(output, source_centers, config), photometric_patches(real_images, real_centers, config)
    use_r1 = step % config["r1_interval"] == 0
    real.requires_grad_(use_r1)
    real_score, fake_score = critic(real, clusters), critic(fake.detach(), clusters)
    discriminator = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
    r1 = real_score.sum() * 0
    if use_r1:
        r1 = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0].square().flatten(1).sum(1).mean()
        discriminator = discriminator + .5 * config["r1_weight"] * config["r1_interval"] * r1
    if not torch.isfinite(discriminator):
        raise FloatingPointError("Nonfinite critic loss")
    discriminator.backward()
    cnorm = nn.utils.clip_grad_norm_(critic.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    co.step()
    critic.eval().requires_grad_(False)
    terms = penalties(images, output, nodes, boxes, config)
    terms["adversarial"] = -critic(fake, clusters).mean()
    loss = sum(config["loss_weights"][key] * value for key, value in terms.items())
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite generator loss")
    loss.backward()
    gnorm = nn.utils.clip_grad_norm_(generator.parameters(), config["gradient_clip"], error_if_nonfinite=True)
    go.step()
    critic.train().requires_grad_(True)
    normalized = (nodes / generator.bounds).abs()
    return {"generator": float(loss.detach()), "critic": float(discriminator.detach()), "r1": float(r1.detach()), "generator_gradient_norm": float(gnorm), "critic_gradient_norm": float(cnorm), "real_score": float(real_score.detach().mean()), "fake_score": float(fake_score.detach().mean()), "gamut_scaled_fraction": float(((scale < .99999) * masks).sum() / masks.sum()), "sigma_95_fraction": float((normalized[:, 2] > .95).float().mean()), "theta_95_fraction": float((normalized[:, 3] > .95).float().mean()), "tint_95_fraction": float((normalized[:, 4:6] > .95).float().mean()), **{key: float(value.detach()) for key, value in terms.items()}}

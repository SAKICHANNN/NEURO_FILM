import torch
from torch import nn
from torch.nn import functional as F


def preview_frame(x: torch.Tensor, side: int) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    h, w = x.shape[-2:]
    scale = side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (side - nh) // 2, (side - nw) // 2
    small = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False, antialias=True)
    pad = (left, side - nw - left, top, side - nh - top)
    image = F.pad(small, pad)
    mask = F.pad(torch.ones_like(small[:, :1]), pad)
    return image, mask, (top, left, nh, nw)


def constrained_coefficients(raw: torch.Tensor, common_bound: float, balance_bound: float) -> torch.Tensor:
    a = common_bound * raw[:, :1].tanh()
    b = raw[:, 1:].tanh()
    b = b - b.mean(1, keepdim=True)
    b = balance_bound * b / b.abs().amax(1, keepdim=True).clamp_min(1)
    return torch.cat((a, b), 1)


def mean_field(coefficients: torch.Tensor) -> torch.Tensor:
    return coefficients.mean((-2, -1), keepdim=True).expand_as(coefficients)


def sample_valid_frame(features: torch.Tensor, mask: torch.Tensor, grid_size: int, stride: int = 8) -> torch.Tensor:
    row_valid, col_valid = mask[:, 0].amax(-1), mask[:, 0].amax(-2)
    rows = torch.arange(mask.shape[-2], device=mask.device, dtype=mask.dtype)[None]
    cols = torch.arange(mask.shape[-1], device=mask.device, dtype=mask.dtype)[None]
    top = torch.where(row_valid > 0, rows, torch.inf).amin(-1)
    left = torch.where(col_valid > 0, cols, torch.inf).amin(-1)
    unit = (torch.arange(grid_size, device=mask.device, dtype=mask.dtype) + 0.5) / grid_size
    y_pixel = top[:, None] + row_valid.sum(-1)[:, None] * unit - 0.5
    x_pixel = left[:, None] + col_valid.sum(-1)[:, None] * unit - 0.5
    # Three stride2/kernel3/pad1 convolutions place feature centers at input indices0,8,16,... .
    yy = 2 * (y_pixel / stride + 0.5) / features.shape[-2] - 1
    xx = 2 * (x_pixel / stride + 0.5) / features.shape[-1] - 1
    grid = torch.stack((xx[:, None, :].expand(-1, grid_size, -1), yy[:, :, None].expand(-1, -1, grid_size)), -1)
    return F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_coefficients(x: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    gain = (dense[:, :1] + dense[:, 1:]).exp()
    numerator = gain * x
    return numerator / ((1 - x) + numerator)


def render_native(x: torch.Tensor, coefficients: torch.Tensor, tile_size: int | None = None) -> torch.Tensor:
    dense = F.interpolate(coefficients, x.shape[-2:], mode="bilinear", align_corners=False)
    if tile_size is None:
        return apply_coefficients(x, dense)
    result = torch.empty_like(x)
    for top in range(0, x.shape[-2], tile_size):
        for left in range(0, x.shape[-1], tile_size):
            sl = (..., slice(top, top + tile_size), slice(left, left + tile_size))
            result[sl] = apply_coefficients(x[sl], dense[sl])
    return result


def render_frames(x: torch.Tensor, coefficients: torch.Tensor, boxes: list) -> torch.Tensor:
    rendered = []
    for index, (top, left, h, w) in enumerate(boxes):
        cut = x[index:index + 1, :, top:top + h, left:left + w]
        y = render_native(cut, coefficients[index:index + 1])
        rendered.append(F.pad(y, (left, x.shape[-1] - left - w, top, x.shape[-2] - top - h)))
    return torch.cat(rendered)


def masked_l1(x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_source = ((x - target).abs() * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3))).clamp_min(1)
    return per_source.mean()


def coefficient_penalties(coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    smooth = sum(coefficients.diff(dim=axis).square().mean() for axis in (-2, -1)) / 2
    return smooth, coefficients.square().mean()


class SourcePreparation(nn.Module):
    def __init__(self, local: bool, grid_size: int = 16, common_bound: float = 1.0, balance_bound: float = 0.25):
        super().__init__()
        self.local, self.grid_size = local, grid_size
        self.common_bound, self.balance_bound = common_bound, balance_bound
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, preview: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padded_raw = self.head(self.encoder(torch.cat((preview, mask), 1)))
        raw = sample_valid_frame(padded_raw, mask, self.grid_size)
        coefficients = constrained_coefficients(raw, self.common_bound, self.balance_bound)
        return coefficients if self.local else mean_field(coefficients)

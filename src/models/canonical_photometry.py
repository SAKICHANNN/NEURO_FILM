import torch
from torch import nn


class CanonicalPhotometryCNN(nn.Module):
    def __init__(self, *, channels: list[int], hidden: int, input_size: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.input_size = input_size
        layers = []
        previous = 3
        spatial = input_size
        for channel in channels:
            layers.extend([nn.Conv2d(previous, channel, kernel_size, stride, padding), nn.GELU()])
            spatial = (spatial + 2 * padding - kernel_size) // stride + 1
            previous = channel
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(previous * spatial * spatial, hidden),
                                  nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, canonical: torch.Tensor) -> torch.Tensor:
        if canonical.ndim != 4 or tuple(canonical.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError('BCHW canonical tensors at configured input size required')
        return self.head(self.encoder(canonical))

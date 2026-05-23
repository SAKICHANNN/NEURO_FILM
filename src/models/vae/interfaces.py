"""VAE latent space interface consumed by CFM and inference pipeline.
契约: CFM 和两阶段推理通过此接口调用 VAE。潜空间形状固定。
"""

import torch
import torch.nn as nn


class VAEInterface(nn.Module):
    """
    AutoencoderKL 风格的潜空间编解码器。

    潜空间规格:
    - Channels: 4
    - Spatial compression: 8× per axis (H→H/8, W→W/8)
    - 像素→潜空间: 64× 压缩比
    - Latent scaling factor: 0.18215 (SD 标准)
    - 目标 VRAM (encode): <1.5 GB at 512×512

    用于 src/models/cfm/velocity_net.py 和 src/inference/pipeline.py
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — linear RGB, [0, 1]
        Returns:
            z: (B, 4, H//8, W//8) — latent, ~N(0,1) after scaling_factor
        """
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 4, H//8, W//8) — latent features
        Returns:
            x: (B, 3, H, W) — reconstructed linear RGB
        """
        raise NotImplementedError

    def tiled_encode(self, x: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
        """
        大数据编码：将输入切分为瓦片以节省显存。
        Args:
            x: high-res input
            tile_size: 每个瓦片的边长（像素）
        """
        raise NotImplementedError

    def tiled_decode(self, z: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
        """
        大数据解码：将潜空间切分为瓦片。
        """
        raise NotImplementedError

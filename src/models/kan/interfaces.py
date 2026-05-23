"""KAN physics layer interfaces consumed by film simulation modules.
契约: 所有 6 个 Film 物理模块通过此接口调用 KAN 函数。
KAN 仅用于低维 (<64) 物理映射，不在骨干中使用。
"""

import torch
import torch.nn as nn


class HnDCurveInterface(nn.Module):
    """
    单个 H&D 特性曲线的 KAN 表示。

    物理背景: 光学密度 D = f(log10 曝光)
    - 线性 RGB → log10 曝光
    - KAN 学习 D = f(log H) 的非线性映射
    - 输出为胶片密度值

    用于 src/models/film/tone.py
    """

    def forward(self, log_exposure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_exposure: (B,) or (B, 1) — log10 曝光值
        Returns:
            density: (B,) — 光学密度 (0 ~ Dmax)
        """
        raise NotImplementedError


class KANScalarFunctionInterface(nn.Module):
    """
    通用标量物理映射的 KAN 接口。
    用于: 曝光→颗粒强度, 局部对比度→DIR 抑制量, 波长→吸收等
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) — 输入物理量
        Returns:
            y: (B, out_dim) — 输出物理量
        """
        raise NotImplementedError

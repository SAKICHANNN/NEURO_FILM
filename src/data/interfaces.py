"""Data pipeline interface consumed by all training modules.
契约: 训练/评估/推理脚本通过此接口获取数据。Batch 结构固定。
"""

from typing import Dict, Any
import torch
from torch.utils.data import Dataset


class FilmDatasetInterface(Dataset):
    """
    配对胶片数据集接口。

    返回结构:
    batch = {
        'digital': Tensor (B, 3, H, W),    # 数字图像，linear RGB [0,1]
        'film':    Tensor (B, 3, H, W),    # 胶片扫描，linear RGB [0,1]
        'meta':    Dict[str, List],         # 元数据 (见 RawLoader)
        'id':      str,                     # 图像标识符
    }

    用于:
    - src/training/trainer.py
    - src/eval/metrics.py
    - scripts/train_*.py
    """

    def __init__(self, data_dir: str, split: str = "train", resolution: int = 512):
        """
        Args:
            data_dir: 预处理后数据的根目录
            split: "train" | "val" | "test"
            resolution: 裁剪/缩放分辨率
        """
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys: 'digital', 'film', 'meta', 'id'
        """
        raise NotImplementedError


class RAWPipelineInterface:
    """
    RAW 文件 → 标准化线性 RGB 的处理管线。

    处理步骤:
    1. 黑电平减法
    2. 白平衡
    3. 去马赛克 (AMaZE)
    4. 色彩空间转换 (Camera → XYZ → sRGB Linear)
    5. 归一化 [0, 1]

    用于 src/data/raw_io.py
    """

    def process(self, raw_path: str) -> torch.Tensor:
        """
        Args:
            raw_path: RAW 文件路径 (.CR2/.NEF/.ARW/.DNG)
        Returns:
            (3, H, W) float32 tensor, linear RGB, [0, 1]
        """
        raise NotImplementedError

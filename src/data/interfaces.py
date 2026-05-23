from typing import Optional
import torch
from PIL import Image


def load_input_image(path: str, target_size: int = 1024) -> Image.Image:
    """Load and preprocess input image for diffusion pipeline."""
    raise NotImplementedError


def load_lora(pipe, lora_path: str):
    """Load LoRA weights into an SDXL pipeline."""
    raise NotImplementedError


def load_ip_adapter(pipe, adapter_path: str, weight_name: str):
    """Load IP-Adapter into an SDXL pipeline."""
    raise NotImplementedError

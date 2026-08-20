"""ICC-aware REPID JPEG ingress for one shared explicit colour operator."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms


def load_jpeg_as_srgb(
    path: Path,
    *,
    expected_sha256: str,
    expected_icc_sha256: str,
) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"input hash drift: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError(f"JPEG RGB contract drift: {path}")
        icc = image.info.get("icc_profile")
        if not icc or hashlib.sha256(icc).hexdigest() != expected_icc_sha256:
            raise ValueError(f"ICC identity drift: {path}")
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        destination_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            image,
            source_profile,
            destination_profile,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            inPlace=False,
            flags=ImageCms.Flags.NONE,
        )
        values = np.asarray(converted, dtype=np.uint8)
    return np.asarray(values, dtype=np.float32) / np.float32(255.0)

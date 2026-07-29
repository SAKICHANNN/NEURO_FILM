from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.preprocess import raw_decode
from src.preprocess.types import InputInspection, SourceProfile


def test_p8bh_raw_decode_explicitly_applies_camera_orientation(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def postprocess(self, **kwargs: object) -> np.ndarray:
            calls.update(kwargs)
            return np.full((3, 4, 3), 32768, dtype=np.uint16)

    fake_module = SimpleNamespace(
        imread=lambda path: FakeRaw(),
        ColorSpace=SimpleNamespace(sRGB="srgb"),
    )
    monkeypatch.setattr(raw_decode, "_rawpy", lambda: fake_module)
    monkeypatch.setattr(
        raw_decode,
        "inspect_raw",
        lambda path: InputInspection(
            path=Path(path),
            exists=True,
            source_kind="raw",
            bit_depth=14,
            source_profile=SourceProfile(
                "raw_metadata", "synthetic"
            ),
            transfer_state="scene_linear",
        ),
    )
    working = raw_decode.load_raw_working_image(
        Path("synthetic.nef")
    )
    assert calls["user_flip"] is None
    assert working.orientation_applied
    assert working.transfer_state == "scene_linear"

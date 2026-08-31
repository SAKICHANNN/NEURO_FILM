from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from src.inference import style_safe_engine
from src.inference.style_safe_engine import (
    StyleSafeEngineError,
    replay_style_safe_recipe_to_file,
)
from src.preprocess.output_encode import (
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
)


def _pixels() -> np.ndarray:
    values = np.arange(7 * 11 * 3, dtype=np.float32).reshape(7, 11, 3)
    return values / float(values.max())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe_for(
    tmp_path: Path,
    *,
    suffix: str,
    bit_depth: int,
    format_name: str,
) -> tuple[dict[str, object], bytes]:
    expected = tmp_path / f"expected{suffix}"
    pixels = _pixels()
    if bit_depth == 8:
        actual_format = save_srgb8(pixels, expected)
    elif format_name == "PNG":
        actual_format = save_srgb16_png(pixels, expected, compression_level=6)
    else:
        actual_format = save_srgb16_tiff(pixels, expected)
    assert actual_format == format_name
    payload = expected.read_bytes()
    recipe: dict[str, object] = {
        "render": {},
        "output": {
            "bit_depth": bit_depth,
            "format": format_name,
            "icc_profile_fingerprint_sha256": (
                style_safe_engine.srgb_icc_profile_fingerprint_sha256()
            ),
            "png_compression": 6,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    return recipe, payload


@pytest.mark.parametrize(
    ("suffix", "bit_depth", "format_name"),
    [
        (".png", 8, "PNG"),
        (".jpg", 8, "JPEG"),
        (".tiff", 8, "TIFF"),
        (".png", 16, "PNG"),
        (".tiff", 16, "TIFF"),
    ],
)
def test_successful_replay_bytes_remain_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    bit_depth: int,
    format_name: str,
) -> None:
    recipe, expected = _recipe_for(
        tmp_path,
        suffix=suffix,
        bit_depth=bit_depth,
        format_name=format_name,
    )
    output = tmp_path / f"replay{suffix}"
    recipe_before = copy.deepcopy(recipe)
    monkeypatch.setattr(
        style_safe_engine, "replay_style_safe_recipe", lambda *_a, **_k: _pixels()
    )

    digest = replay_style_safe_recipe_to_file(
        recipe,
        profile_path=tmp_path / "unused-profile.json",
        output_path=output,
        root=tmp_path,
    )

    assert output.read_bytes() == expected
    assert digest == _sha256(output)
    assert recipe == recipe_before
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_late_foreign_destination_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.preprocess import output_encode

    recipe, _expected = _recipe_for(
        tmp_path, suffix=".png", bit_depth=8, format_name="PNG"
    )
    output = tmp_path / "late.png"
    foreign = b"late-foreign-output"
    real_publish = output_encode.publish_create_only
    monkeypatch.setattr(
        style_safe_engine, "replay_style_safe_recipe", lambda *_a, **_k: _pixels()
    )

    def inject(stage: Path, final: Path):
        final.write_bytes(foreign)
        return real_publish(stage, final)

    monkeypatch.setattr(output_encode, "publish_create_only", inject)
    with pytest.raises(FileExistsError):
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=tmp_path / "unused-profile.json",
            output_path=output,
            root=tmp_path,
        )

    assert output.read_bytes() == foreign
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_postpublication_foreign_replacement_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, _expected = _recipe_for(
        tmp_path, suffix=".png", bit_depth=8, format_name="PNG"
    )
    output = tmp_path / "replaced.png"
    replacement = tmp_path / "replacement.bin"
    foreign = b"postpublication-foreign"
    replacement.write_bytes(foreign)
    monkeypatch.setattr(
        style_safe_engine, "replay_style_safe_recipe", lambda *_a, **_k: _pixels()
    )

    def replace_then_hash(path: Path) -> str:
        os.replace(replacement, path)
        return "0" * 64

    monkeypatch.setattr(style_safe_engine, "sha256_file", replace_then_hash)
    with pytest.raises(StyleSafeEngineError, match="byte identity differs"):
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=tmp_path / "unused-profile.json",
            output_path=output,
            root=tmp_path,
        )

    assert output.read_bytes() == foreign


def test_still_owned_mismatch_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, _expected = _recipe_for(
        tmp_path, suffix=".png", bit_depth=8, format_name="PNG"
    )
    output = tmp_path / "owned-mismatch.png"
    monkeypatch.setattr(
        style_safe_engine, "replay_style_safe_recipe", lambda *_a, **_k: _pixels()
    )
    monkeypatch.setattr(style_safe_engine, "sha256_file", lambda _path: "0" * 64)

    with pytest.raises(StyleSafeEngineError, match="byte identity differs"):
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=tmp_path / "unused-profile.json",
            output_path=output,
            root=tmp_path,
        )

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_existing_destination_rejects_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, _expected = _recipe_for(
        tmp_path, suffix=".png", bit_depth=8, format_name="PNG"
    )
    output = tmp_path / "existing.png"
    foreign = b"existing-foreign"
    output.write_bytes(foreign)

    def must_not_render(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("render must not run")

    monkeypatch.setattr(style_safe_engine, "replay_style_safe_recipe", must_not_render)
    with pytest.raises(StyleSafeEngineError, match="already exists"):
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=tmp_path / "unused-profile.json",
            output_path=output,
            root=tmp_path,
        )
    assert output.read_bytes() == foreign


def test_dangling_symlink_destination_rejects_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, _expected = _recipe_for(
        tmp_path, suffix=".png", bit_depth=8, format_name="PNG"
    )
    output = tmp_path / "dangling.png"
    try:
        output.symlink_to(tmp_path / "missing-target.png")
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    def must_not_render(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("render must not run")

    monkeypatch.setattr(style_safe_engine, "replay_style_safe_recipe", must_not_render)
    with pytest.raises(StyleSafeEngineError, match="already exists"):
        replay_style_safe_recipe_to_file(
            recipe,
            profile_path=tmp_path / "unused-profile.json",
            output_path=output,
            root=tmp_path,
        )
    assert output.is_symlink()

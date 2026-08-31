from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocess.output_encode import save_srgb8

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2l_product_create_only_render_transaction_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _command(source: Path, output: Path, *, style: str = "velvia_50") -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--output",
        str(output),
    ]


def _run(
    source: Path, output: Path, *, style: str = "velvia_50"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(source, output, style=style),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_product_same_path_rejects_before_decode(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    completed = _run(missing, missing)
    assert completed.returncode != 0
    assert "product output must not identify the input path" in completed.stderr
    assert "No such file" not in completed.stderr
    assert not missing.exists()


def test_product_existing_regular_and_hardlink_reject_before_decode(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must_not_decode.png"
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"foreign")
    hardlink = tmp_path / "hardlink.png"
    os.link(existing, hardlink)
    for destination in (existing, hardlink):
        before = destination.read_bytes()
        completed = _run(missing, destination)
        assert completed.returncode != 0
        assert "product output destination must not already exist" in completed.stderr
        assert "must_not_decode" not in completed.stderr
        assert destination.read_bytes() == before


@pytest.mark.parametrize("broken", [False, True])
def test_product_symlink_entries_reject_before_decode(
    tmp_path: Path, broken: bool
) -> None:
    missing = tmp_path / "must_not_decode.png"
    target = tmp_path / "target.png"
    if not broken:
        target.write_bytes(b"foreign")
    destination = tmp_path / ("broken-link.png" if broken else "link.png")
    try:
        destination.symlink_to(target.name)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    completed = _run(missing, destination)
    assert completed.returncode != 0
    assert "product output destination must not already exist" in completed.stderr
    assert "must_not_decode" not in completed.stderr
    assert destination.is_symlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point contract")
def test_product_reparse_entry_rejects_before_decode(tmp_path: Path) -> None:
    missing = tmp_path / "must_not_decode.png"
    target = tmp_path / "junction-target"
    target.mkdir()
    destination = tmp_path / "output.png"
    destination_arg = str(destination).replace("'", "''")
    target_arg = str(target).replace("'", "''")
    command = (
        "$null = New-Item -ItemType Junction "
        f"-Path '{destination_arg}' -Target '{target_arg}'"
    )
    created = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr}")
    try:
        completed = _run(missing, destination)
        assert completed.returncode != 0
        assert "product output destination must not already exist" in completed.stderr
        assert "must_not_decode" not in completed.stderr
        assert destination.is_dir()
    finally:
        destination.rmdir()


def test_create_only_encoder_preserves_late_foreign_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.preprocess import output_encode

    destination = tmp_path / "late.png"
    foreign = b"late-foreign"
    real_publish = output_encode.publish_create_only

    def inject(stage: Path, final: Path):
        final.write_bytes(foreign)
        return real_publish(stage, final)

    monkeypatch.setattr(output_encode, "publish_create_only", inject)
    with pytest.raises(FileExistsError):
        save_srgb8(np.zeros((3, 4, 3), dtype=np.float32), destination, create_only=True)
    assert destination.read_bytes() == foreign
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_create_only_encoder_failure_leaves_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "failed.png"

    def fail_save(*_args, **_kwargs) -> None:
        raise RuntimeError("injected encode failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected encode failure"):
        save_srgb8(np.zeros((3, 4, 3), dtype=np.float32), destination, create_only=True)
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.parametrize("style", ["velvia_50", "portra_400", "ektar_100"])
def test_product_outputs_remain_byte_exact_and_source_immutable(
    tmp_path: Path, style: str
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / f"{style}.png"
    _source(source)
    source_before = source.read_bytes()
    completed = _run(source, output, style=style)
    assert completed.returncode == 0, completed.stderr
    assert _sha(output) == config["prechange_output_sha256"][style]
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_concurrent_product_publication_has_one_exact_winner(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / "shared.png"
    _source(source)
    source_before = source.read_bytes()
    processes = [
        subprocess.Popen(
            _command(source, output),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=60) for process in processes]
    codes = [process.returncode for process in processes]
    assert sorted(codes) == [0, 1]
    assert _sha(output) == config["prechange_output_sha256"]["velvia_50"]
    assert source.read_bytes() == source_before
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
    failed_stderr = next(
        stderr
        for process, (_, stderr) in zip(processes, results, strict=True)
        if process.returncode != 0
    )
    assert failed_stderr
    assert "source.png" not in failed_stderr


def test_legacy_profile_still_replaces_existing_target(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / "legacy.png"
    _source(source)
    output.write_bytes(b"replace-me")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--style",
            "velvia_50",
            "--use-render-profile",
            "--render-profile",
            str(LEGACY_PROFILE),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _sha(output) == config["prechange_output_sha256"]["legacy_velvia_50"]

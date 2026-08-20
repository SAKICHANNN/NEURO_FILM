from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.acquire_u5_r2repid3_fit_calibration import _download


def test_download_reuses_exact_existing_file(tmp_path: Path) -> None:
    payload = b"exact-existing"
    destination = tmp_path / "member.jpeg"
    destination.write_bytes(payload)
    _download(
        "https://invalid.example/never-read",
        destination,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        ".part",
    )
    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()

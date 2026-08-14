"""Exact receipt replay for the strict opt-in ProPhoto Rec.2020 renderer."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.inference.romm_rec2020_velvia import (
    PROPHOTO_PROFILE_ID,
    PROPHOTO_RECEIPT_SCHEMA,
    ROMMRec2020RenderError,
    load_profile,
    render_supported_prophoto_velvia_rec2020,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_supported_prophoto_velvia_rec2020(
    receipt: Mapping[str, Any],
    input_path: Path,
    output_path: Path,
    *,
    profile_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Re-render one strict receipt and require full receipt identity."""

    if (
        receipt.get("schema") != PROPHOTO_RECEIPT_SCHEMA
        or receipt.get("profile_id") != PROPHOTO_PROFILE_ID
        or receipt.get("production_default_changed") is not False
        or not isinstance(receipt.get("input"), Mapping)
        or not isinstance(receipt.get("output"), Mapping)
    ):
        raise ROMMRec2020RenderError("unsupported ProPhoto replay receipt")
    input_sha256 = receipt["input"].get("sha256")
    output_sha256 = receipt["output"].get("sha256")
    profile_sha256 = receipt.get("profile_sha256")
    if not all(
        isinstance(value, str) and _SHA256.fullmatch(value)
        for value in (input_sha256, output_sha256, profile_sha256)
    ):
        raise ROMMRec2020RenderError("ProPhoto replay receipt identity drift")
    input_path = Path(input_path)
    if not input_path.is_file() or _sha256(input_path) != input_sha256:
        raise ROMMRec2020RenderError("ProPhoto replay input identity drift")
    _, current_profile_sha256 = load_profile(profile_path, root=root)
    if current_profile_sha256 != profile_sha256:
        raise ROMMRec2020RenderError("ProPhoto replay profile identity drift")
    replayed = render_supported_prophoto_velvia_rec2020(
        input_path,
        output_path,
        profile_path=profile_path,
        root=root,
    )
    if replayed != dict(receipt):
        Path(output_path).unlink(missing_ok=True)
        raise ROMMRec2020RenderError("ProPhoto replay receipt mismatch")
    return replayed


__all__ = ["replay_supported_prophoto_velvia_rec2020"]

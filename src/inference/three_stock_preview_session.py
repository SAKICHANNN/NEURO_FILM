"""Immutable warm-session snapshots for verified three-stock previews."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .render_contract import sha256_file
from .three_stock_preview_cache import (
    ThreeStockPreviewCacheError,
    inspect_receipt_bound_three_stock_preview_cache,
    inspect_three_stock_preview_cache,
)


@dataclass(frozen=True, slots=True)
class VerifiedPreviewPayload:
    """One immutable, hash-bound preview payload."""

    style_id: str
    filename: str
    output_sha256: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class VerifiedThreeStockPreviewSnapshot:
    """One lookup result backed only by immutable admitted memory."""

    input_sha256: str
    profile_sha256: str
    preview_width: int
    preview_height: int
    preview_pixels: int
    look_amount: float
    rows: tuple[VerifiedPreviewPayload, ...]


@dataclass(frozen=True, slots=True)
class VerifiedThreeStockPreviewSession:
    """An admitted preview snapshot whose lookups require no file access."""

    snapshot: VerifiedThreeStockPreviewSnapshot


@dataclass(frozen=True, slots=True)
class ReceiptBoundVerifiedThreeStockPreviewSnapshot:
    """One immutable preview snapshot plus its admitted cache-index receipt."""

    cache_index_sha256: str
    preview: VerifiedThreeStockPreviewSnapshot


@dataclass(frozen=True, slots=True)
class ReceiptBoundVerifiedThreeStockPreviewSession:
    """A cache-index-bound session whose lookups require no file access."""

    snapshot: ReceiptBoundVerifiedThreeStockPreviewSnapshot


def _admit_verified_three_stock_preview_snapshot(
    preview_directory: Path,
    *,
    input_path: Path,
    profile_path: Path,
    index: dict[str, object],
) -> VerifiedThreeStockPreviewSnapshot:
    admitted_rows: list[VerifiedPreviewPayload] = []
    rows = index["rows"]
    if not isinstance(rows, list):
        raise ThreeStockPreviewCacheError("preview cache row inventory drift")
    for row in rows:
        if not isinstance(row, dict):
            raise ThreeStockPreviewCacheError("preview cache row drift")
        payload = (preview_directory / str(row["filename"])).read_bytes()
        if sha256(payload).hexdigest() != row["output_sha256"]:
            raise ThreeStockPreviewCacheError(
                "preview output changed during session admission"
            )
        admitted_rows.append(
            VerifiedPreviewPayload(
                style_id=str(row["style_id"]),
                filename=str(row["filename"]),
                output_sha256=str(row["output_sha256"]),
                payload=payload,
            )
        )

    # Close mutations that occur while the preview payloads are admitted.
    if sha256_file(input_path) != index["input_sha256"]:
        raise ThreeStockPreviewCacheError(
            "preview input changed during session admission"
        )
    if sha256_file(profile_path) != index["profile_sha256"]:
        raise ThreeStockPreviewCacheError(
            "preview profile changed during session admission"
        )

    return VerifiedThreeStockPreviewSnapshot(
        input_sha256=str(index["input_sha256"]),
        profile_sha256=str(index["profile_sha256"]),
        preview_width=int(index["preview_width"]),
        preview_height=int(index["preview_height"]),
        preview_pixels=int(index["preview_pixels"]),
        look_amount=float(index["look_amount"]),
        rows=tuple(admitted_rows),
    )


def admit_verified_three_stock_preview_session(
    preview_directory: Path,
    *,
    input_path: Path,
    profile_path: Path,
) -> VerifiedThreeStockPreviewSession:
    """Fully validate a U7.3H cache and retain owned immutable preview bytes."""

    preview_directory = Path(preview_directory)
    input_path = Path(input_path)
    profile_path = Path(profile_path)
    index = inspect_three_stock_preview_cache(
        preview_directory,
        input_path=input_path,
        profile_path=profile_path,
    )
    return VerifiedThreeStockPreviewSession(
        snapshot=_admit_verified_three_stock_preview_snapshot(
            preview_directory,
            input_path=input_path,
            profile_path=profile_path,
            index=index,
        )
    )


def admit_receipt_bound_three_stock_preview_session(
    preview_directory: Path,
    *,
    input_path: Path,
    profile_path: Path,
    cache_index_sha256: str,
) -> ReceiptBoundVerifiedThreeStockPreviewSession:
    """Admit immutable previews only from exact receipt-bound index bytes."""

    preview_directory = Path(preview_directory)
    input_path = Path(input_path)
    profile_path = Path(profile_path)
    index = inspect_receipt_bound_three_stock_preview_cache(
        preview_directory,
        input_path=input_path,
        profile_path=profile_path,
        cache_index_sha256=cache_index_sha256,
    )
    return ReceiptBoundVerifiedThreeStockPreviewSession(
        snapshot=ReceiptBoundVerifiedThreeStockPreviewSnapshot(
            cache_index_sha256=cache_index_sha256,
            preview=_admit_verified_three_stock_preview_snapshot(
                preview_directory,
                input_path=input_path,
                profile_path=profile_path,
                index=index,
            ),
        )
    )


def lookup_verified_three_stock_preview_session(
    session: VerifiedThreeStockPreviewSession,
) -> VerifiedThreeStockPreviewSnapshot:
    """Return the admitted immutable snapshot without filesystem access."""

    if not isinstance(session, VerifiedThreeStockPreviewSession):
        raise TypeError("session must be a VerifiedThreeStockPreviewSession")
    return session.snapshot


def lookup_receipt_bound_three_stock_preview_session(
    session: ReceiptBoundVerifiedThreeStockPreviewSession,
) -> ReceiptBoundVerifiedThreeStockPreviewSnapshot:
    """Return the receipt-bound snapshot without filesystem access."""

    if not isinstance(session, ReceiptBoundVerifiedThreeStockPreviewSession):
        raise TypeError(
            "session must be a ReceiptBoundVerifiedThreeStockPreviewSession"
        )
    return session.snapshot


__all__ = [
    "ReceiptBoundVerifiedThreeStockPreviewSession",
    "ReceiptBoundVerifiedThreeStockPreviewSnapshot",
    "VerifiedPreviewPayload",
    "VerifiedThreeStockPreviewSession",
    "VerifiedThreeStockPreviewSnapshot",
    "admit_receipt_bound_three_stock_preview_session",
    "admit_verified_three_stock_preview_session",
    "lookup_receipt_bound_three_stock_preview_session",
    "lookup_verified_three_stock_preview_session",
]

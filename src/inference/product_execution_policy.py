"""Fixed internal execution policy for bounded product rendering."""

from __future__ import annotations

from dataclasses import dataclass

PRODUCT_PROFILE_ID = "safe-rich-product-v1"
PRODUCT_DEFAULT_TILE_SIZE = 256
PRODUCT_DEFAULT_TILE_WORKERS = 1


@dataclass(frozen=True)
class ProductExecutionPolicy:
    tile_size: int | None
    tile_workers: int
    automatic: bool


def resolve_product_execution_policy(
    *,
    profile_id: str,
    tile_size: int | None,
    tile_workers: int,
) -> ProductExecutionPolicy:
    """Resolve the fixed product default without rewriting explicit requests."""

    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be a non-empty string")
    if tile_size is not None and (
        isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1
    ):
        raise ValueError("tile_size must be a positive integer or None")
    if (
        isinstance(tile_workers, bool)
        or not isinstance(tile_workers, int)
        or tile_workers < 1
    ):
        raise ValueError("tile_workers must be a positive integer")
    if tile_size is None and tile_workers != 1:
        raise ValueError("tile_workers requires tile_size")

    if profile_id == PRODUCT_PROFILE_ID and tile_size is None:
        return ProductExecutionPolicy(
            tile_size=PRODUCT_DEFAULT_TILE_SIZE,
            tile_workers=PRODUCT_DEFAULT_TILE_WORKERS,
            automatic=True,
        )
    return ProductExecutionPolicy(
        tile_size=tile_size,
        tile_workers=tile_workers,
        automatic=False,
    )


__all__ = [
    "PRODUCT_DEFAULT_TILE_SIZE",
    "PRODUCT_DEFAULT_TILE_WORKERS",
    "PRODUCT_PROFILE_ID",
    "ProductExecutionPolicy",
    "resolve_product_execution_policy",
]

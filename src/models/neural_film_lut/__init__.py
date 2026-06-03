"""Neural film LUT V2 components.

Keep this package init lightweight. Some utilities use the numpy distilled LUT
runtime and should not pay for torch initialization.
"""

STYLE_NAMES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]

__all__ = ["STYLE_NAMES"]

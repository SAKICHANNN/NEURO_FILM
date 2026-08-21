from .interpretation import (
    COLOR_DOMAINS,
    EVIDENCE_SCOPES,
    INTERPRETATION_CONTRACT_VERSION,
    INTERPRETATION_IDS,
    InterpretationContractError,
    InterpretationMetadata,
    InterpretationPlugin,
    InterpretationRequest,
    InterpretationResult,
    execute_interpretation,
)
from .render_contract import (
    LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID,
    PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID,
    PROFILE_SCHEMA_ID,
    RECIPE_SCHEMA_ID,
    RenderContractError,
    atomic_write_json,
    build_render_recipe,
    load_render_profile,
    migrate_legacy_safe_rich,
    sha256_file,
    summarize_legacy_style_evidence_inventory,
    summarize_render_profile_evidence,
    validate_render_profile,
    validate_render_recipe,
    verify_render_recipe_files,
)
from .tiled_render import (
    TiledExecutionMetadata,
    TiledRenderError,
    TileWindow,
    execute_tiled_local_operator,
    plan_tile_windows,
)

__all__ = [
    "COLOR_DOMAINS",
    "EVIDENCE_SCOPES",
    "INTERPRETATION_CONTRACT_VERSION",
    "INTERPRETATION_IDS",
    "LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID",
    "PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID",
    "PROFILE_SCHEMA_ID",
    "RECIPE_SCHEMA_ID",
    "InterpretationContractError",
    "InterpretationMetadata",
    "InterpretationPlugin",
    "InterpretationRequest",
    "InterpretationResult",
    "RenderContractError",
    "StyleSafeEngineError",
    "TileWindow",
    "TiledExecutionMetadata",
    "TiledRenderError",
    "atomic_write_json",
    "build_render_recipe",
    "execute_interpretation",
    "execute_tiled_local_operator",
    "load_render_profile",
    "migrate_legacy_safe_rich",
    "plan_tile_windows",
    "render_resolved_safe_lab_rgb",
    "render_style_safe_working_image",
    "sha256_file",
    "summarize_legacy_style_evidence_inventory",
    "summarize_render_profile_evidence",
    "validate_render_profile",
    "validate_render_recipe",
    "verify_render_recipe_files",
]


def __getattr__(name: str):
    if name in {
        "StyleSafeEngineError",
        "render_resolved_safe_lab_rgb",
        "render_style_safe_working_image",
    }:
        from . import style_safe_engine

        return getattr(style_safe_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

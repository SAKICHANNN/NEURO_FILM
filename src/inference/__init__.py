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
from .portable_recipe_bundle import (
    bind_portable_recipe_recovery_bundle,
    inspect_portable_recipe_recovery_bundle,
)
from .recipe_history import (
    RECIPE_HISTORY_SCHEMA_ID,
    RecipeHistoryError,
    build_render_recipe_history,
)
from .recipe_history_html import (
    RECIPE_HISTORY_HTML_SCHEMA_ID,
    RecipeHistoryHtmlError,
    render_recipe_history_html,
)
from .recipe_preview import (
    RecipePreview,
    RecipePreviewError,
    build_recipe_output_preview,
    build_recipe_output_previews,
    render_recipe_preview_html,
)
from .recipe_recovery_bundle import (
    RECOVERY_BUNDLE_FORMAT,
    RECOVERY_BUNDLE_SCHEMA_ID,
    RecipeRecoveryBundleError,
    build_recipe_recovery_bundle,
    inspect_materialized_recipe_recovery_tree,
    inspect_recipe_recovery_bundle,
    materialize_recipe_recovery_bundle,
    update_materialized_recipe_recovery_tree,
)
from .render_contract import (
    LEGACY_STYLE_EVIDENCE_INVENTORY_SCHEMA_ID,
    PROFILE_EVIDENCE_SUMMARY_SCHEMA_ID,
    PROFILE_SCHEMA_ID,
    RECIPE_SCHEMA_ID,
    RECIPE_SCHEMA_ID_V2,
    RECIPE_SCHEMA_ID_V3,
    RECIPE_SCHEMA_ID_V4,
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
    verify_render_recipe_inputs,
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
    "RECIPE_HISTORY_HTML_SCHEMA_ID",
    "RECIPE_HISTORY_SCHEMA_ID",
    "RECIPE_SCHEMA_ID",
    "RECIPE_SCHEMA_ID_V2",
    "RECIPE_SCHEMA_ID_V3",
    "RECIPE_SCHEMA_ID_V4",
    "RECOVERY_BUNDLE_FORMAT",
    "RECOVERY_BUNDLE_SCHEMA_ID",
    "InterpretationContractError",
    "InterpretationMetadata",
    "InterpretationPlugin",
    "InterpretationRequest",
    "InterpretationResult",
    "NativeThreeStockPreviewBackend",
    "RecipeHistoryError",
    "RecipeHistoryHtmlError",
    "RecipePreview",
    "RecipePreviewError",
    "RecipeRecoveryBundleError",
    "RenderContractError",
    "StyleSafeEngineError",
    "TileWindow",
    "TiledExecutionMetadata",
    "TiledRenderError",
    "atomic_write_json",
    "bind_portable_recipe_recovery_bundle",
    "build_native_three_stock_preview_backend",
    "build_recipe_output_preview",
    "build_recipe_output_previews",
    "build_recipe_recovery_bundle",
    "build_render_recipe",
    "build_render_recipe_history",
    "execute_interpretation",
    "execute_tiled_local_operator",
    "inspect_materialized_recipe_recovery_tree",
    "inspect_portable_recipe_recovery_bundle",
    "inspect_recipe_recovery_bundle",
    "iter_three_stock_look_rgb_native",
    "list_three_stock_looks",
    "load_render_profile",
    "materialize_recipe_recovery_bundle",
    "migrate_legacy_safe_rich",
    "plan_tile_windows",
    "render_recipe_history_html",
    "render_recipe_preview_html",
    "render_resolved_safe_lab_rgb",
    "render_style_safe_working_image",
    "render_three_stock_look_rgb",
    "replay_portable_recipe_recovery_bundle_to_file",
    "replay_style_safe_color_recipe",
    "replay_style_safe_recipe",
    "replay_style_safe_recipe_to_file",
    "resolve_three_stock_look_parameters",
    "sha256_file",
    "summarize_legacy_style_evidence_inventory",
    "summarize_render_profile_evidence",
    "update_materialized_recipe_recovery_tree",
    "validate_render_profile",
    "validate_render_recipe",
    "verify_render_recipe_files",
    "verify_render_recipe_inputs",
]


def __getattr__(name: str):
    if name in {
        "StyleSafeEngineError",
        "render_resolved_safe_lab_rgb",
        "render_style_safe_working_image",
        "replay_style_safe_color_recipe",
        "replay_style_safe_recipe",
        "replay_style_safe_recipe_to_file",
        "replay_portable_recipe_recovery_bundle_to_file",
        "list_three_stock_looks",
        "render_three_stock_look_rgb",
        "resolve_three_stock_look_parameters",
        "NativeThreeStockPreviewBackend",
        "build_native_three_stock_preview_backend",
        "iter_three_stock_look_rgb_native",
    }:
        if name == "replay_portable_recipe_recovery_bundle_to_file":
            from . import portable_recipe_replay

            return getattr(portable_recipe_replay, name)
        if name in {
            "NativeThreeStockPreviewBackend",
            "build_native_three_stock_preview_backend",
            "iter_three_stock_look_rgb_native",
        }:
            from . import three_stock_native_preview

            return getattr(three_stock_native_preview, name)
        if name in {
            "list_three_stock_looks",
            "render_three_stock_look_rgb",
            "resolve_three_stock_look_parameters",
        }:
            from . import three_stock_look

            return getattr(three_stock_look, name)
        from . import style_safe_engine

        return getattr(style_safe_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

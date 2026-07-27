"""Product-facing reference colour matching contracts and engine."""

from .contracts import (
    REFERENCE_LOOK_RECIPE_SCHEMA_ID,
    SUPPORTED_WORKING_SPACES,
    ReferenceLookPolicy,
    ReferenceLookRecipe,
    ReferenceMatchContractError,
    recipe_from_json,
    recipe_to_json,
    validate_recipe,
)
from .composition import (
    REFERENCE_COMPOSITION_SCHEMA_ID,
    FilmEffectBinding,
    ReferenceCompositionPlan,
    build_reference_composition,
    composition_plan_from_dict,
    composition_plan_from_json,
    composition_plan_to_json,
    validate_reference_composition,
)
from .fit import fit_reference_look
from .files import (
    FileReferenceMatchOutput,
    FileReferenceMatchResult,
    match_reference_files,
)
from .evaluation import (
    KnownOperatorBatchMetrics,
    KnownOperatorSampleMetrics,
    evaluate_known_operator_batch,
)
from .render import (
    ReferenceMatchDiagnostics,
    ReferenceMatchResult,
    render_reference_batch,
    render_reference_look,
)
from .replay import (
    load_reference_look_recipe,
    replay_reference_batch,
    replay_reference_batch_guarded,
    save_reference_look_recipe,
)
from .reporting import (
    REFERENCE_MATCH_REPORT_SCHEMA_ID,
    build_file_match_report,
    save_file_match_report,
)
from .safety import (
    REFERENCE_RENDER_GUARD_POLICY_ID,
    GuardedReferenceMatchResult,
    ReferenceRenderGuardPolicy,
    ReferenceSafetyDecision,
    render_reference_batch_guarded,
    render_reference_look_guarded,
    validate_guard_policy,
)

__all__ = [
    "REFERENCE_LOOK_RECIPE_SCHEMA_ID",
    "REFERENCE_MATCH_REPORT_SCHEMA_ID",
    "REFERENCE_RENDER_GUARD_POLICY_ID",
    "REFERENCE_COMPOSITION_SCHEMA_ID",
    "SUPPORTED_WORKING_SPACES",
    "FilmEffectBinding",
    "GuardedReferenceMatchResult",
    "KnownOperatorBatchMetrics",
    "KnownOperatorSampleMetrics",
    "FileReferenceMatchOutput",
    "FileReferenceMatchResult",
    "ReferenceLookPolicy",
    "ReferenceLookRecipe",
    "ReferenceCompositionPlan",
    "ReferenceMatchDiagnostics",
    "ReferenceMatchResult",
    "ReferenceMatchContractError",
    "ReferenceRenderGuardPolicy",
    "ReferenceSafetyDecision",
    "fit_reference_look",
    "evaluate_known_operator_batch",
    "build_reference_composition",
    "composition_plan_from_dict",
    "composition_plan_from_json",
    "composition_plan_to_json",
    "build_file_match_report",
    "load_reference_look_recipe",
    "match_reference_files",
    "recipe_from_json",
    "recipe_to_json",
    "replay_reference_batch",
    "replay_reference_batch_guarded",
    "render_reference_batch",
    "render_reference_batch_guarded",
    "render_reference_look",
    "render_reference_look_guarded",
    "save_reference_look_recipe",
    "save_file_match_report",
    "validate_recipe",
    "validate_reference_composition",
    "validate_guard_policy",
]

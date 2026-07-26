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
    save_reference_look_recipe,
)

__all__ = [
    "REFERENCE_LOOK_RECIPE_SCHEMA_ID",
    "REFERENCE_COMPOSITION_SCHEMA_ID",
    "SUPPORTED_WORKING_SPACES",
    "FilmEffectBinding",
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
    "fit_reference_look",
    "evaluate_known_operator_batch",
    "build_reference_composition",
    "load_reference_look_recipe",
    "match_reference_files",
    "recipe_from_json",
    "recipe_to_json",
    "replay_reference_batch",
    "render_reference_batch",
    "render_reference_look",
    "save_reference_look_recipe",
    "validate_recipe",
    "validate_reference_composition",
]

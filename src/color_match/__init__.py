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
from .fit import fit_reference_look
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
    "SUPPORTED_WORKING_SPACES",
    "ReferenceLookPolicy",
    "ReferenceLookRecipe",
    "ReferenceMatchDiagnostics",
    "ReferenceMatchResult",
    "ReferenceMatchContractError",
    "fit_reference_look",
    "load_reference_look_recipe",
    "recipe_from_json",
    "recipe_to_json",
    "replay_reference_batch",
    "render_reference_batch",
    "render_reference_look",
    "save_reference_look_recipe",
    "validate_recipe",
]

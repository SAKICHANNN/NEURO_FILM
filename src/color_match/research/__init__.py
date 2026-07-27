"""Isolated research challengers for uploaded-reference colour matching.

Nothing in this package is a promoted product renderer. Candidates must pass
the normal reference-match promotion gates before moving into
``src.color_match``.
"""

from .cfsm import (
    CFSM_ALGORITHM_ID,
    CFSM_BATCH_ALGORITHM_ID,
    CFSM_CANDIDATE_SCHEMA_ID,
    CFSMCandidate,
    CFSMFitDiagnostics,
    CFSMProjectionPolicy,
    cfsm_candidate_from_json,
    cfsm_candidate_to_json,
    compute_cfsm_candidate_id,
    fit_cfsm_batch_candidate,
    fit_cfsm_candidate,
    render_cfsm_batch,
    render_cfsm_candidate,
    validate_cfsm_candidate,
)

__all__ = [
    "CFSM_ALGORITHM_ID",
    "CFSM_BATCH_ALGORITHM_ID",
    "CFSM_CANDIDATE_SCHEMA_ID",
    "CFSMCandidate",
    "CFSMFitDiagnostics",
    "CFSMProjectionPolicy",
    "cfsm_candidate_from_json",
    "cfsm_candidate_to_json",
    "compute_cfsm_candidate_id",
    "fit_cfsm_batch_candidate",
    "fit_cfsm_candidate",
    "render_cfsm_batch",
    "render_cfsm_candidate",
    "validate_cfsm_candidate",
]

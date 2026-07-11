"""FilmCase-specific data governance and bounded-routing components.

This package is intentionally isolated from the legacy renderer and historical
training scripts. Its first module audits source lineage before any FilmCase
case memory or routing experiment is allowed to use a reference image.
"""

from .lineage import (
    AuditResult,
    ManifestAuditError,
    audit_manifest,
    write_audit_outputs,
)

__all__ = [
    "AuditResult",
    "ManifestAuditError",
    "audit_manifest",
    "write_audit_outputs",
]

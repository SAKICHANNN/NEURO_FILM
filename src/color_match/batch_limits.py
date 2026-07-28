"""Shared resource limits for one-reference/N-source product batches."""

from __future__ import annotations


# Runtime verification, decoded MatchView construction, and product staging all
# use this ceiling.  Enforce it at the first consumer batch boundary so an
# oversized request cannot become valid upstream and fail only after expensive
# output materialization.
MAX_REFERENCE_MATCH_BATCH_SOURCES = 64


__all__ = ["MAX_REFERENCE_MATCH_BATCH_SOURCES"]

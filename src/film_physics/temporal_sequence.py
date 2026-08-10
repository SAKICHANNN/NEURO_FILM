"""Explicit sequence identities for deterministic temporal physical streams."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SEQUENCE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_STREAM_ROLE = re.compile(r"[a-z][a-z0-9_]{0,63}")


class TemporalSequenceIdentityError(ValueError):
    """Raised when a temporal stream identity is ambiguous or invalid."""


@dataclass(frozen=True)
class TemporalSequenceIdentity:
    sequence_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not _SEQUENCE_ID.fullmatch(
            self.sequence_id
        ):
            raise TemporalSequenceIdentityError(
                "sequence_id must be 1-128 lowercase ASCII identity characters"
            )


def derive_temporal_stream_seed(
    *, base_seed: int, sequence: TemporalSequenceIdentity, stream_role: str
) -> int:
    """Derive a signed-63-bit, role-separated seed for one explicit sequence."""
    if not isinstance(base_seed, int) or not 0 <= base_seed < 2**64:
        raise TemporalSequenceIdentityError("base_seed must be unsigned 64-bit")
    if not isinstance(stream_role, str) or not _STREAM_ROLE.fullmatch(stream_role):
        raise TemporalSequenceIdentityError("stream_role is invalid")
    payload = (
        b"neuro-film.temporal-sequence-stream.v1\0"
        + base_seed.to_bytes(8, "little")
        + len(sequence.sequence_id).to_bytes(2, "little")
        + sequence.sequence_id.encode("ascii")
        + b"\0"
        + stream_role.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (2**63 - 1)

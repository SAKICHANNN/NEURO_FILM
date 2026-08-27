from __future__ import annotations

import hashlib

import pytest

from src.real_film.negicc_it8_source import (
    NegiccIt8SourceError,
    _git_blob_sha1,
    _parse_raw,
    _parse_train,
)


def test_git_blob_sha1_matches_git_object_rule() -> None:
    payload = b"patch r g b\na1 1 2 3\n"
    expected = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
    ).hexdigest()
    assert _git_blob_sha1(payload) == expected


def test_raw_and_train_parsers_preserve_exact_patch_values() -> None:
    raw = _parse_raw(b"patch r g b\na1 1 2 3\ngs0 4 5 6\n")
    train = _parse_train(
        b"patch,r,g,b,refR,refG,refB,refX,refY,refZ\n"
        b"a1,1,2,3,0,0,0,4,5,6\n"
        b"gs0,4,5,6,0,0,0,7,8,9\n"
    )
    assert raw == {patch: value[0] for patch, value in train.items()}
    assert train["gs0"][1] == (7.0, 8.0, 9.0)


def test_train_parser_rejects_duplicate_patch() -> None:
    with pytest.raises(NegiccIt8SourceError, match="duplicate"):
        _parse_train(
            b"patch,r,g,b,refR,refG,refB,refX,refY,refZ\n"
            b"a1,1,2,3,0,0,0,4,5,6\n"
            b"a1,1,2,3,0,0,0,4,5,6\n"
        )

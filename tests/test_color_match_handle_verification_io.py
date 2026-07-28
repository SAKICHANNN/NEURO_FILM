from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from src.color_match.contracts import ReferenceMatchContractError
import src.color_match.handle_verification_io as handle_io
from src.color_match.handle_verification_io import (
    StableFileHandleLease,
    open_stable_file_handle,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_stable_handle_double_reads_one_open_file_without_writing(
    tmp_path: Path,
) -> None:
    payload = b"handle-bound-report\n"
    target = (tmp_path / "report.json").resolve()
    target.write_bytes(payload)
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }

    with open_stable_file_handle(target, label="test report") as lease:
        verified = lease.verify(
            expected_sha256=_sha256(payload),
            maximum_bytes=1024,
            capture_bytes=True,
        )
        lease.final_check()

    assert verified.path == str(target)
    assert verified.raw_bytes == payload
    assert verified.sha256 == _sha256(payload)
    assert verified.size_bytes == len(payload)
    assert len(verified.handle_identity_id) == 64
    assert {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    } == before


def test_read_pass_rejects_growth_as_soon_as_bound_is_crossed() -> None:
    lease = object.__new__(StableFileHandleLease)
    lease.label = "growing report"
    lease.stream = io.BytesIO(b"x" * 9)

    with pytest.raises(ReferenceMatchContractError, match="bounded"):
        lease._read_pass(capture_bytes=True, maximum_bytes=8)


def test_double_read_divergence_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"stable-payload"
    target = (tmp_path / "target.bin").resolve()
    target.write_bytes(payload)

    with open_stable_file_handle(target, label="target") as lease:
        original = lease._read_pass
        calls = 0

        def divergent_read(
            *,
            capture_bytes: bool,
            maximum_bytes: int,
        ) -> tuple[str, bytes | None, int]:
            nonlocal calls
            calls += 1
            digest, raw, size = original(
                capture_bytes=capture_bytes,
                maximum_bytes=maximum_bytes,
            )
            if calls == 2:
                digest = "0" * 64
            return digest, raw, size

        monkeypatch.setattr(lease, "_read_pass", divergent_read)
        with pytest.raises(
            ReferenceMatchContractError,
            match="changed between handle read passes",
        ):
            lease.verify(
                expected_sha256=_sha256(payload),
                maximum_bytes=1024,
                capture_bytes=True,
            )


def test_verify_never_reopens_data_after_lease_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"single-open"
    target = (tmp_path / "target.bin").resolve()
    target.write_bytes(payload)

    with open_stable_file_handle(target, label="target") as lease:
        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("data path reopened")

        monkeypatch.setattr(handle_io.os, "open", forbidden)
        monkeypatch.setattr(Path, "open", forbidden)
        monkeypatch.setattr(Path, "read_bytes", forbidden)
        verified = lease.verify(
            expected_sha256=_sha256(payload),
            maximum_bytes=1024,
            capture_bytes=True,
        )
        lease.final_check()

    assert verified.raw_bytes == payload


def test_namespace_replacement_after_open_is_blocked_or_rejected(
    tmp_path: Path,
) -> None:
    trusted = b"trusted"
    target = (tmp_path / "target.bin").resolve()
    foreign = (tmp_path / "foreign.bin").resolve()
    target.write_bytes(trusted)
    foreign.write_bytes(b"foreign")

    with open_stable_file_handle(target, label="race target") as lease:
        try:
            os.replace(foreign, target)
            replaced = True
        except OSError:
            replaced = False
        if os.name == "nt":
            assert not replaced
            assert lease.verify(
                expected_sha256=_sha256(trusted),
                maximum_bytes=1024,
                capture_bytes=True,
            ).raw_bytes == trusted
            lease.final_check()
        else:
            assert replaced
            with pytest.raises(
                ReferenceMatchContractError,
                match="final handle path mismatch|namespace identity changed",
            ):
                lease.verify(
                    expected_sha256=_sha256(trusted),
                    maximum_bytes=1024,
                    capture_bytes=True,
                )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows share-deny prevents same-inode overwrite",
)
def test_same_tick_same_size_overwrite_is_caught_by_final_rehash(
    tmp_path: Path,
) -> None:
    trusted = b"AAAA"
    target = (tmp_path / "target.bin").resolve()
    target.write_bytes(trusted)

    with open_stable_file_handle(target, label="target") as lease:
        lease.verify(
            expected_sha256=_sha256(trusted),
            maximum_bytes=1024,
            capture_bytes=False,
        )
        target.write_bytes(b"EVIL")
        with pytest.raises(
            ReferenceMatchContractError,
            match="content changed before verification closure",
        ):
            lease.final_check()


def test_hard_links_have_the_same_handle_identity(tmp_path: Path) -> None:
    target = (tmp_path / "target.bin").resolve()
    alias = (tmp_path / "alias.bin").resolve()
    target.write_bytes(b"same-object")
    os.link(target, alias)

    with (
        open_stable_file_handle(target, label="target") as target_lease,
        open_stable_file_handle(alias, label="alias") as alias_lease,
    ):
        target_file = target_lease.verify(
            expected_sha256=_sha256(b"same-object"),
            maximum_bytes=1024,
            capture_bytes=False,
        )
        alias_file = alias_lease.verify(
            expected_sha256=_sha256(b"same-object"),
            maximum_bytes=1024,
            capture_bytes=False,
        )

    assert target_file.handle_identity_id == alias_file.handle_identity_id


def test_directory_and_final_symlink_fail_closed(tmp_path: Path) -> None:
    directory = (tmp_path / "directory").resolve()
    directory.mkdir()
    with pytest.raises(ReferenceMatchContractError):
        with open_stable_file_handle(directory, label="directory"):
            pass

    target = (tmp_path / "target.bin").resolve()
    link = (tmp_path / "link.bin").resolve()
    target.write_bytes(b"target")
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ReferenceMatchContractError, match="symlink|reparse"):
        with open_stable_file_handle(link, label="link"):
            pass


def test_exception_path_closes_handle_and_allows_rename(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "target.bin").resolve()
    renamed = (tmp_path / "renamed.bin").resolve()
    target.write_bytes(b"target")

    with pytest.raises(RuntimeError, match="forced"):
        with open_stable_file_handle(target, label="target"):
            raise RuntimeError("forced")

    target.rename(renamed)
    assert renamed.read_bytes() == b"target"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup path")
def test_base_exception_during_windows_acquisition_closes_native_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "target.bin").resolve()
    renamed = (tmp_path / "renamed.bin").resolve()
    target.write_bytes(b"target")
    native_closes = 0
    original_close = handle_io._CloseHandle

    def tracked_close(handle: int) -> object:
        nonlocal native_closes
        native_closes += 1
        return original_close(handle)

    def cancelled(_handle: int) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(handle_io, "_CloseHandle", tracked_close)
    monkeypatch.setattr(handle_io, "_windows_attribute_info", cancelled)
    with pytest.raises(KeyboardInterrupt):
        with open_stable_file_handle(target, label="target"):
            pass

    assert native_closes == 1
    target.rename(renamed)
    assert renamed.exists()

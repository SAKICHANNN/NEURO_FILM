"""Cross-process cooperative lock for atomic multi-file transactions."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Any, Iterator, Sequence

from .contracts import ReferenceMatchContractError


_LOCK_GUARD = threading.Lock()
_HELD_LOCK_KEYS: set[str] = set()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(
        attributes
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _reject_reparse_components(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in (absolute, *absolute.parents):
        if _is_reparse_point(component):
            raise ReferenceMatchContractError(
                f"{label} must not traverse a symlink or reparse point"
            )
    return absolute


def _lock_key(path: Path) -> str:
    canonical = os.path.normcase(os.path.abspath(os.fspath(path)))
    return hashlib.sha256(
        b"NeuroFilmReferenceMatchTransactionPathV1\0"
        + canonical.encode("utf-8")
    ).hexdigest()


def _acquire_platform_lock(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
    except OSError as exc:
        raise ReferenceMatchContractError(
            "reference-match transaction destination is already locked"
        ) from exc


def _release_platform_lock(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def target_transaction_lock(paths: Sequence[Path]) -> Iterator[None]:
    keys = tuple(sorted(_lock_key(path) for path in paths))
    if not keys:
        raise ReferenceMatchContractError(
            "reference-match transaction paths must not be empty"
        )
    if len(set(keys)) != len(keys):
        raise ReferenceMatchContractError(
            "reference-match transaction paths must be unique"
        )
    temp_root = _reject_reparse_components(
        Path(tempfile.gettempdir()),
        label="reference-match transaction temporary root",
    )
    if not temp_root.is_dir():
        raise ReferenceMatchContractError(
            "reference-match transaction temporary root must be a directory"
        )
    lock_root = temp_root / "neuro-film-reference-match-target-locks-v2"
    with _LOCK_GUARD:
        if any(key in _HELD_LOCK_KEYS for key in keys):
            raise ReferenceMatchContractError(
                "reference-match transaction destination is already locked"
            )
        _HELD_LOCK_KEYS.update(keys)
    handles: list[Any] = []
    try:
        _reject_reparse_components(
            lock_root,
            label="reference-match transaction lock root",
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        _reject_reparse_components(
            lock_root,
            label="reference-match transaction lock root",
        )
        for key in keys:
            lock_path = lock_root / f"{key}.lock"
            if _is_reparse_point(lock_path):
                raise ReferenceMatchContractError(
                    "reference-match transaction lock is a reparse point"
                )
            handle = lock_path.open("a+b")
            try:
                if (
                    _is_reparse_point(lock_path)
                    or not stat.S_ISREG(os.fstat(handle.fileno()).st_mode)
                ):
                    raise ReferenceMatchContractError(
                        "reference-match transaction lock must be a "
                        "non-reparse regular file"
                    )
                _acquire_platform_lock(handle)
            except Exception:
                handle.close()
                raise
            handles.append(handle)
        yield
    finally:
        try:
            for handle in reversed(handles):
                try:
                    try:
                        _release_platform_lock(handle)
                    except OSError:
                        pass
                finally:
                    try:
                        handle.close()
                    except OSError:
                        pass
        finally:
            with _LOCK_GUARD:
                _HELD_LOCK_KEYS.difference_update(keys)


__all__ = ["target_transaction_lock"]

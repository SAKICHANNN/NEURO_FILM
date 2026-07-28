"""Single-open, handle-bound verification for immutable staging artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Any, BinaryIO, Iterator

from .canonical import canonical_sha256
from .contracts import ReferenceMatchContractError


HANDLE_POLICY_ID = (
    "neuro-film.single-open-double-read-final-rehash.v1"
)
WINDOWS_HANDLE_IDENTITY_SCHEME = "windows-volume-file-id.v1"
POSIX_HANDLE_IDENTITY_SCHEME = "posix-device-inode.v1"
MAX_STAGED_OUTPUT_BYTES = 1024 * 1024 * 1024
MAX_STAGED_OUTPUT_AGGREGATE_BYTES = 8 * 1024 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_HASH_CHARS = frozenset("0123456789abcdef")


def _checked_hash(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HASH_CHARS for char in value)
    ):
        raise ReferenceMatchContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _canonical_absolute_path(
    path: Path | str,
    *,
    label: str,
) -> Path:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            f"{label} must be a canonical absolute path"
        ) from exc
    if not isinstance(raw, str) or not raw:
        raise ReferenceMatchContractError(
            f"{label} must be a canonical absolute path"
        )
    if "\x00" in raw:
        raise ReferenceMatchContractError(
            f"{label} must not contain a null byte"
        )
    try:
        absolute = os.path.abspath(raw)
    except (OSError, ValueError) as exc:
        raise ReferenceMatchContractError(
            f"{label} must be a canonical absolute path"
        ) from exc
    if raw != absolute:
        raise ReferenceMatchContractError(
            f"{label} must be a canonical absolute path"
        )
    if os.name == "nt":
        _drive, tail = os.path.splitdrive(absolute)
        if ":" in tail or absolute.startswith(
            ("\\\\.\\", "\\\\?\\", "\\??\\")
        ):
            raise ReferenceMatchContractError(
                f"{label} must not use an alternate stream or device path"
            )
    return Path(absolute)


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


def _fail_fast_reparse_check(path: Path, *, label: str) -> None:
    for component in (path, *path.parents):
        try:
            is_reparse = _is_reparse_point(component)
        except OSError as exc:
            raise ReferenceMatchContractError(
                f"{label} path components cannot be inspected"
            ) from exc
        if is_reparse:
            raise ReferenceMatchContractError(
                f"{label} must not traverse a symlink or reparse point"
            )


def _stable_stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _identity_id(
    *,
    scheme: str,
    components: dict[str, str],
) -> str:
    return canonical_sha256(
        {
            "schema_id": "neuro-film.open-handle-identity.v1",
            "scheme": scheme,
            "components": components,
        }
    )


@dataclass(frozen=True)
class VerifiedHandleFile:
    path: str
    sha256: str
    size_bytes: int
    handle_identity_id: str
    handle_identity_scheme: str
    raw_bytes: bytes | None


class StableFileHandleLease:
    """One open artifact retained through a complete multi-file verification."""

    def __init__(
        self,
        *,
        path: Path,
        label: str,
        stream: BinaryIO,
        identity_scheme: str,
        identity_id: str,
        initial_stat: os.stat_result,
        platform_handle: int | None,
        parent_fd: int | None,
        leaf_name: str | None,
    ) -> None:
        self.path = path
        self.label = label
        self.stream = stream
        self.identity_scheme = identity_scheme
        self.identity_id = identity_id
        self.initial_stat = initial_stat
        self.platform_handle = platform_handle
        self.parent_fd = parent_fd
        self.leaf_name = leaf_name
        self._closed = False
        self._verified: VerifiedHandleFile | None = None

    def __enter__(self) -> StableFileHandleLease:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stream.close()
        finally:
            if self.parent_fd is not None:
                os.close(self.parent_fd)

    def _current_identity_id(self) -> str:
        if self.identity_scheme == WINDOWS_HANDLE_IDENTITY_SCHEME:
            if self.platform_handle is None:
                raise ReferenceMatchContractError(
                    f"{self.label} lost its Windows handle"
                )
            return _windows_identity_id(self.platform_handle)
        value = os.fstat(self.stream.fileno())
        return _posix_identity_id(value)

    def _current_final_path(self) -> Path:
        if self.identity_scheme == WINDOWS_HANDLE_IDENTITY_SCHEME:
            if self.platform_handle is None:
                raise ReferenceMatchContractError(
                    f"{self.label} lost its Windows handle"
                )
            return Path(_windows_final_path(self.platform_handle))
        return Path(_posix_final_path(self.stream.fileno()))

    def _assert_namespace_binding(self) -> None:
        if self._current_identity_id() != self.identity_id:
            raise ReferenceMatchContractError(
                f"{self.label} handle identity changed"
            )
        if _path_key(self._current_final_path()) != _path_key(self.path):
            raise ReferenceMatchContractError(
                f"{self.label} final handle path mismatch"
            )
        if self.parent_fd is not None and self.leaf_name is not None:
            try:
                named = os.stat(
                    self.leaf_name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ReferenceMatchContractError(
                    f"{self.label} namespace binding is unavailable"
                ) from exc
            if (
                not stat.S_ISREG(named.st_mode)
                or _posix_identity_id(named) != self.identity_id
            ):
                raise ReferenceMatchContractError(
                    f"{self.label} namespace identity changed"
                )

    def _read_pass(
        self,
        *,
        capture_bytes: bool,
        maximum_bytes: int,
    ) -> tuple[str, bytes | None, int]:
        try:
            self.stream.seek(0)
        except OSError as exc:
            raise ReferenceMatchContractError(
                f"{self.label} is not seekable"
            ) from exc
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        total = 0
        while True:
            try:
                chunk = self.stream.read(_CHUNK_BYTES)
            except OSError as exc:
                raise ReferenceMatchContractError(
                    f"{self.label} is unreadable"
                ) from exc
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ReferenceMatchContractError(
                    f"{self.label} exceeds its bounded size contract"
                )
            if chunks is not None:
                chunks.append(chunk)
        raw = b"".join(chunks) if chunks is not None else None
        return digest.hexdigest(), raw, total

    def verify(
        self,
        *,
        expected_sha256: str,
        maximum_bytes: int,
        capture_bytes: bool,
    ) -> VerifiedHandleFile:
        expected = _checked_hash(expected_sha256, "expected file SHA-256")
        if (
            not stat.S_ISREG(self.initial_stat.st_mode)
            or self.initial_stat.st_size <= 0
            or self.initial_stat.st_size > maximum_bytes
        ):
            raise ReferenceMatchContractError(
                f"{self.label} violates the regular bounded-file contract"
            )
        read_limit = int(self.initial_stat.st_size)
        self._assert_namespace_binding()
        first_digest, raw, first_size = self._read_pass(
            capture_bytes=capture_bytes,
            maximum_bytes=read_limit,
        )
        middle = os.fstat(self.stream.fileno())
        second_digest, _discarded, second_size = self._read_pass(
            capture_bytes=False,
            maximum_bytes=read_limit,
        )
        final = os.fstat(self.stream.fileno())
        if (
            _stable_stat_tuple(self.initial_stat)
            != _stable_stat_tuple(middle)
            or _stable_stat_tuple(middle) != _stable_stat_tuple(final)
            or first_size != self.initial_stat.st_size
            or second_size != self.initial_stat.st_size
        ):
            raise ReferenceMatchContractError(
                f"{self.label} changed during handle verification"
            )
        if first_digest != second_digest:
            raise ReferenceMatchContractError(
                f"{self.label} changed between handle read passes"
            )
        if first_digest != expected:
            raise ReferenceMatchContractError(
                f"{self.label} hash mismatch"
            )
        self._assert_namespace_binding()
        result = VerifiedHandleFile(
            path=str(self.path),
            sha256=first_digest,
            size_bytes=int(final.st_size),
            handle_identity_id=self.identity_id,
            handle_identity_scheme=self.identity_scheme,
            raw_bytes=raw,
        )
        self._verified = result
        return result

    def final_check(self) -> None:
        if self._verified is None:
            raise ReferenceMatchContractError(
                f"{self.label} was not handle-verified"
            )
        before = os.fstat(self.stream.fileno())
        if _stable_stat_tuple(before) != _stable_stat_tuple(
            self.initial_stat
        ):
            raise ReferenceMatchContractError(
                f"{self.label} changed before verification closure"
            )
        digest, _discarded, size = self._read_pass(
            capture_bytes=False,
            maximum_bytes=int(self.initial_stat.st_size),
        )
        after = os.fstat(self.stream.fileno())
        if (
            digest != self._verified.sha256
            or size != self.initial_stat.st_size
            or _stable_stat_tuple(before) != _stable_stat_tuple(after)
        ):
            raise ReferenceMatchContractError(
                f"{self.label} content changed before verification closure"
            )
        self._assert_namespace_binding()


def _posix_identity_id(value: os.stat_result) -> str:
    if int(value.st_dev) == 0 or int(value.st_ino) == 0:
        raise ReferenceMatchContractError(
            "POSIX filesystem does not expose a stable file identity"
        )
    return _identity_id(
        scheme=POSIX_HANDLE_IDENTITY_SCHEME,
        components={
            "device": str(int(value.st_dev)),
            "inode": str(int(value.st_ino)),
        },
    )


def _posix_final_path(fd: int) -> str:
    if sys.platform.startswith(("linux", "android")):
        try:
            return os.readlink(f"/proc/self/fd/{fd}")
        except OSError as exc:
            raise ReferenceMatchContractError(
                "POSIX final handle path is unavailable"
            ) from exc
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        buffer = ctypes.create_string_buffer(4096)
        f_getpath = 50
        if libc.fcntl(fd, f_getpath, buffer) != 0:
            error = ctypes.get_errno()
            raise ReferenceMatchContractError(
                "Darwin final handle path is unavailable"
            ) from OSError(error, os.strerror(error))
        return os.fsdecode(buffer.value)
    raise ReferenceMatchContractError(
        "platform does not expose a supported final handle path"
    )


def _open_posix_lease(
    path: Path,
    *,
    label: str,
) -> StableFileHandleLease:
    components = path.parts
    if not path.is_absolute() or len(components) < 2:
        raise ReferenceMatchContractError(
            f"{label} must name a regular file"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    directory_fd: int | None = None
    file_fd: int | None = None
    stream: BinaryIO | None = None
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for component in components[1:-1]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=directory_fd,
            )
            try:
                os.close(directory_fd)
            except BaseException:
                try:
                    os.close(next_fd)
                except BaseException:
                    pass
                raise
            directory_fd = next_fd
        leaf_name = components[-1]
        file_fd = os.open(
            leaf_name,
            file_flags,
            dir_fd=directory_fd,
        )
        value = os.fstat(file_fd)
        if not stat.S_ISREG(value.st_mode):
            raise ReferenceMatchContractError(
                f"{label} must be a regular file"
            )
        identity_id = _posix_identity_id(value)
        stream = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = None
        lease = StableFileHandleLease(
            path=path,
            label=label,
            stream=stream,
            identity_scheme=POSIX_HANDLE_IDENTITY_SCHEME,
            identity_id=identity_id,
            initial_stat=value,
            platform_handle=None,
            parent_fd=directory_fd,
            leaf_name=leaf_name,
        )
        directory_fd = None
        stream = None
        return lease
    except BaseException as exc:
        if stream is not None:
            try:
                stream.close()
            except BaseException:
                pass
        if file_fd is not None:
            try:
                os.close(file_fd)
            except BaseException:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException:
                pass
        if isinstance(exc, ReferenceMatchContractError):
            raise
        if isinstance(exc, OSError):
            raise ReferenceMatchContractError(
                f"{label} cannot be securely opened"
            ) from exc
        raise


if os.name == "nt":
    from ctypes import wintypes
    import msvcrt

    class _FILE_ID_128(ctypes.Structure):
        _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

    class _FILE_ID_INFO(ctypes.Structure):
        _fields_ = [
            ("VolumeSerialNumber", ctypes.c_ulonglong),
            ("FileId", _FILE_ID_128),
        ]

    class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", wintypes.DWORD),
            ("ReparseTag", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL
    _GetFileInformationByHandleEx = (
        _kernel32.GetFileInformationByHandleEx
    )
    _GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _GetFileInformationByHandleEx.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _GetFileType = _kernel32.GetFileType
    _GetFileType.argtypes = [wintypes.HANDLE]
    _GetFileType.restype = wintypes.DWORD

    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_TYPE_DISK = 0x0001
    _FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
    _FILE_ID_INFO_CLASS = 18
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _windows_error(message: str) -> ReferenceMatchContractError:
    return ReferenceMatchContractError(
        f"{message}: {ctypes.WinError(ctypes.get_last_error())}"
    )


def _windows_attribute_info(handle: int) -> Any:
    info = _FILE_ATTRIBUTE_TAG_INFO()
    if not _GetFileInformationByHandleEx(
        handle,
        _FILE_ATTRIBUTE_TAG_INFO_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise _windows_error("Windows file attributes are unavailable")
    return info


def _windows_identity_id(handle: int) -> str:
    info = _FILE_ID_INFO()
    if not _GetFileInformationByHandleEx(
        handle,
        _FILE_ID_INFO_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise _windows_error("Windows file identity is unavailable")
    file_id = bytes(info.FileId.Identifier).hex()
    if int(info.VolumeSerialNumber) == 0 or file_id == "0" * 32:
        raise ReferenceMatchContractError(
            "Windows filesystem does not expose a stable file identity"
        )
    return _identity_id(
        scheme=WINDOWS_HANDLE_IDENTITY_SCHEME,
        components={
            "volume_serial": f"{int(info.VolumeSerialNumber):016x}",
            "file_id_128": file_id,
        },
    )


def _windows_final_path(handle: int) -> str:
    size = 32768
    buffer = ctypes.create_unicode_buffer(size)
    length = _GetFinalPathNameByHandleW(handle, buffer, size, 0)
    if length == 0:
        raise _windows_error("Windows final handle path is unavailable")
    if length >= size:
        size = length + 1
        buffer = ctypes.create_unicode_buffer(size)
        length = _GetFinalPathNameByHandleW(handle, buffer, size, 0)
        if length == 0 or length >= size:
            raise _windows_error(
                "Windows final handle path exceeds the supported bound"
            )
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _open_windows_lease(
    path: Path,
    *,
    label: str,
) -> StableFileHandleLease:
    handle = _CreateFileW(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_READ,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_OPEN_REPARSE_POINT | _FILE_FLAG_SEQUENTIAL_SCAN,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise _windows_error(f"{label} cannot be securely opened")
    fd: int | None = None
    stream: BinaryIO | None = None
    try:
        attributes = _windows_attribute_info(handle)
        if (
            int(attributes.FileAttributes)
            & (
                _FILE_ATTRIBUTE_DIRECTORY
                | _FILE_ATTRIBUTE_REPARSE_POINT
            )
            or int(attributes.ReparseTag) != 0
            or _GetFileType(handle) != _FILE_TYPE_DISK
        ):
            raise ReferenceMatchContractError(
                f"{label} must be a non-reparse regular disk file"
            )
        identity_id = _windows_identity_id(handle)
        final_path = _windows_final_path(handle)
        if _path_key(final_path) != _path_key(path):
            raise ReferenceMatchContractError(
                f"{label} final handle path mismatch"
            )
        fd = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        handle = None
        stream = os.fdopen(fd, "rb", closefd=True)
        fd = None
        value = os.fstat(stream.fileno())
        lease = StableFileHandleLease(
            path=path,
            label=label,
            stream=stream,
            identity_scheme=WINDOWS_HANDLE_IDENTITY_SCHEME,
            identity_id=identity_id,
            initial_stat=value,
            platform_handle=msvcrt.get_osfhandle(stream.fileno()),
            parent_fd=None,
            leaf_name=None,
        )
        stream = None
        return lease
    except BaseException:
        if stream is not None:
            try:
                stream.close()
            except BaseException:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except BaseException:
                pass
        if handle not in (None, _INVALID_HANDLE_VALUE):
            _CloseHandle(handle)
        raise


@contextmanager
def open_stable_file_handle(
    path: Path | str,
    *,
    label: str,
) -> Iterator[StableFileHandleLease]:
    canonical = _canonical_absolute_path(path, label=label)
    _fail_fast_reparse_check(canonical, label=label)
    lease = (
        _open_windows_lease(canonical, label=label)
        if os.name == "nt"
        else _open_posix_lease(canonical, label=label)
    )
    try:
        yield lease
    finally:
        lease.close()


__all__ = [
    "HANDLE_POLICY_ID",
    "MAX_STAGED_OUTPUT_AGGREGATE_BYTES",
    "MAX_STAGED_OUTPUT_BYTES",
    "POSIX_HANDLE_IDENTITY_SCHEME",
    "WINDOWS_HANDLE_IDENTITY_SCHEME",
    "StableFileHandleLease",
    "VerifiedHandleFile",
    "open_stable_file_handle",
]

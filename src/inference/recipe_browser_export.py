"""One-shot loopback browser bridge for strict recipe replay."""

from __future__ import annotations

import hashlib
import html
import secrets
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import parse_qs, urlsplit

from .recipe_export_request import (
    DEFAULT_MAXIMUM_REQUEST_BYTES,
    RecipeExportRequestError,
    build_recipe_export_request_set,
    export_recipe_request,
)

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_MAXIMUM_FORM_BYTES = 4096
RecipeBrowserPageRenderer = Callable[[list[dict[str, Any]], str], bytes]


class RecipeBrowserExportError(ValueError):
    """Raised when a local browser export session cannot be used."""


class _LoopbackHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        if isinstance(sys.exc_info()[1], ConnectionResetError):
            return
        super().handle_error(request, client_address)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _page(rows: list[dict[str, Any]], token: str) -> bytes:
    cards = "".join(
        "<form method=\"post\" action=\"/export\"><input type=\"hidden\" "
        "name=\"token\" value=\"{token}\"><input type=\"hidden\" "
        "name=\"request_file\" value=\"{request}\"><p class=\"eyebrow\">"
        "{style}</p><h2>{format} · {depth}-bit</h2><p>{label} · {grade}</p>"
        "<button type=\"submit\">Export this look</button></form>".format(
            token=html.escape(token, quote=True),
            request=html.escape(str(row["request_file"]), quote=True),
            style=html.escape(str(row["style"])),
            format=html.escape(str(row["output_format"])),
            depth=row["output_bit_depth"],
            label=html.escape(str(row["output_label"])),
            grade=html.escape(str(row["evidence_grade"])),
        )
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'">
<title>Export a film look · K-MCFM</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#101210;color:#f2f4ef;font:16px/1.5 system-ui,"Segoe UI",sans-serif}}main{{width:min(900px,calc(100vw - 24px));margin:0 auto;padding:42px 0 64px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}form{{padding:22px;border:1px solid #3a403a;border-radius:16px;background:#191c19}}.eyebrow{{color:#d7ff74;font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}h2{{margin:.3rem 0}}p{{color:#aab2aa}}button{{margin-top:8px;border:0;color:#101210;background:#d7ff74;padding:10px 14px;border-radius:10px;font:inherit;font-weight:800;cursor:pointer}}button:focus-visible{{outline:3px solid #fff;outline-offset:3px}}
</style></head><body><main><p class="eyebrow">K-MCFM local session</p><h1>Export a verified film look</h1><p>Selecting a look directly invokes its existing strict recipe replay. The browser cannot choose source, profile, or filesystem paths.</p><section class="grid" aria-label="Verified export recipes">{cards}</section><p>This one-shot loopback session ends after one successful export. All entries remain film-inspired Look Approximations.</p></main></body></html>""".encode()


def _result_page(style: str, output_name: str, output_sha256: str) -> bytes:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Export complete · K-MCFM</title><style>:root{{color-scheme:dark}}body{{margin:3rem;background:#101210;color:#f2f4ef;font:16px/1.5 system-ui}}code{{overflow-wrap:anywhere;color:#d7ff74}}</style></head><body><main><h1>Export complete</h1><p>{html.escape(style)} was published as <code>{html.escape(output_name)}</code>.</p><p>SHA-256 <code>{output_sha256}</code></p><p>The local one-shot session is now closed.</p></main></body></html>""".encode()


@dataclass(frozen=True)
class BrowserExportResult:
    request_file: str
    style: str
    output_path: Path
    output_sha256: str
    receipt: dict[str, Any]


class RecipeBrowserExportSession:
    """A bounded one-success HTTP session listening only on IPv4 loopback."""

    def __init__(
        self,
        history_root: Path,
        output_root: Path,
        *,
        profile_path: Path,
        root: Path,
        tile_size: int | None = None,
        maximum_recipe_files: int = 10_000,
        maximum_recipe_bytes: int = 2 * 1024 * 1024,
        maximum_request_bytes: int = DEFAULT_MAXIMUM_REQUEST_BYTES,
        maximum_form_bytes: int = DEFAULT_MAXIMUM_FORM_BYTES,
        page_renderer: RecipeBrowserPageRenderer = _page,
    ) -> None:
        if type(maximum_form_bytes) is not int or maximum_form_bytes < 1:
            raise RecipeBrowserExportError("maximum_form_bytes must be positive")
        if not callable(page_renderer):
            raise RecipeBrowserExportError("page_renderer must be callable")
        if output_root.exists() and not output_root.is_dir():
            raise RecipeBrowserExportError("output root must be a directory")
        output_root.mkdir(parents=True, exist_ok=True)
        request_set = build_recipe_export_request_set(
            history_root,
            maximum_recipe_files=maximum_recipe_files,
            maximum_recipe_bytes=maximum_recipe_bytes,
        )
        rows = request_set["receipt"]["requests"]
        if not rows:
            raise RecipeBrowserExportError("no valid export recipes are available")
        self._history_root = history_root
        self._output_root = output_root
        self._profile_path = profile_path
        self._root = root
        self._tile_size = tile_size
        self._maximum_recipe_files = maximum_recipe_files
        self._maximum_recipe_bytes = maximum_recipe_bytes
        self._maximum_request_bytes = maximum_request_bytes
        self._maximum_form_bytes = maximum_form_bytes
        self._page_renderer = page_renderer
        self._rows = {str(row["request_file"]): row for row in rows}
        self._payloads = {
            name: request_set["files"][name] for name in self._rows
        }
        self._token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._result: BrowserExportResult | None = None
        self._scratch = tempfile.TemporaryDirectory(prefix="neuro-film-browser-export-")
        self._server = _LoopbackHTTPServer((LOOPBACK_HOST, 0), self._handler_type())
        self._thread: threading.Thread | None = None

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        session = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "KMCFMRecipeExport/1"

            def log_message(self, format: str, *args: object) -> None:
                return

            def _write(self, status: HTTPStatus, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                if parsed.path != "/" or query != {"token": [session._token]}:
                    self._write(HTTPStatus.NOT_FOUND, b"not found\n")
                    return
                payload = session._page_renderer(
                    list(session._rows.values()), session._token
                )
                if not isinstance(payload, bytes) or not payload:
                    self._write(HTTPStatus.INTERNAL_SERVER_ERROR, b"page unavailable\n")
                    return
                self._write(HTTPStatus.OK, payload)

            def do_POST(self) -> None:
                if self.path != "/export":
                    self._write(HTTPStatus.NOT_FOUND, b"not found\n")
                    return
                try:
                    raw_length = self.headers.get("Content-Length", "")
                    length = int(raw_length)
                    if length < 1 or length > session._maximum_form_bytes:
                        raise RecipeBrowserExportError("form byte limit exceeded")
                    payload = self.rfile.read(length)
                    fields = parse_qs(
                        payload.decode("ascii"),
                        keep_blank_values=True,
                        strict_parsing=True,
                    )
                    if set(fields) != {"token", "request_file"} or any(
                        len(values) != 1 for values in fields.values()
                    ):
                        raise RecipeBrowserExportError("form fields drift")
                    if not secrets.compare_digest(fields["token"][0], session._token):
                        raise RecipeBrowserExportError("session token mismatch")
                    result = session._export(fields["request_file"][0])
                except (UnicodeDecodeError, ValueError, RecipeExportRequestError) as exc:
                    self._write(
                        HTTPStatus.BAD_REQUEST,
                        f"<h1>Export rejected</h1><p>{html.escape(str(exc))}</p>".encode(),
                    )
                    return
                self._write(
                    HTTPStatus.OK,
                    _result_page(
                        result.style,
                        result.output_path.name,
                        result.output_sha256,
                    ),
                )

        return Handler

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self._server.server_port}/?token={self._token}"

    @property
    def port(self) -> int:
        return int(self._server.server_port)

    @property
    def result(self) -> BrowserExportResult | None:
        return self._result

    @property
    def request_files(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def start(self) -> None:
        if self._thread is not None:
            raise RecipeBrowserExportError("session already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="recipe-browser-export",
            daemon=True,
        )
        self._thread.start()

    def wait(self, timeout: float | None = None) -> BrowserExportResult | None:
        self._done.wait(timeout)
        return self._result

    def _export(self, request_file: str) -> BrowserExportResult:
        with self._lock:
            if self._result is not None:
                raise RecipeBrowserExportError("one-shot session is closed")
            if request_file not in self._rows:
                raise RecipeBrowserExportError("unknown export request")
            row = self._rows[request_file]
            extension = str(row["output_format"]).lower()
            if extension not in {"png", "jpg", "jpeg", "tif", "tiff"}:
                raise RecipeBrowserExportError("unsupported output extension")
            output = self._output_root / f"{row['style']}.{extension}"
            if output.exists() or output.is_symlink():
                raise RecipeBrowserExportError("output already exists")
            request_path = Path(self._scratch.name) / request_file
            request_path.write_bytes(self._payloads[request_file])
            try:
                receipt = export_recipe_request(
                    self._history_root,
                    request_path,
                    profile_path=self._profile_path,
                    output_path=output,
                    root=self._root,
                    maximum_recipe_files=self._maximum_recipe_files,
                    maximum_recipe_bytes=self._maximum_recipe_bytes,
                    maximum_request_bytes=self._maximum_request_bytes,
                    tile_size=self._tile_size,
                )
            finally:
                request_path.unlink(missing_ok=True)
            result = BrowserExportResult(
                request_file=request_file,
                style=str(row["style"]),
                output_path=output,
                output_sha256=_sha256(output.read_bytes()),
                receipt=receipt,
            )
            self._result = result
            self._done.set()
            return result

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._scratch.cleanup()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_MAXIMUM_FORM_BYTES",
    "LOOPBACK_HOST",
    "BrowserExportResult",
    "RecipeBrowserExportError",
    "RecipeBrowserExportSession",
    "RecipeBrowserPageRenderer",
]

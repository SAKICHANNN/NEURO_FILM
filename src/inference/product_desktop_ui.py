"""Tkinter view for the bounded new-input Look Approximation workflow."""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from queue import Empty, SimpleQueue
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .product_desktop import (
    PRODUCT_LOOKS,
    PRODUCT_OUTPUT_FORMATS,
    PRODUCT_PREVIEW_DISPLAY_SIZE,
    DesktopBatchInput,
    DesktopBatchReceipt,
    DesktopExportReceipt,
    DesktopPreviewState,
    ProductDesktopError,
    ProductDesktopWorkflow,
    product_output_format,
)

_LOOK_IDS = tuple(row["style_id"] for row in PRODUCT_LOOKS)


class ProductDesktopApp:
    """Single-window native selector, preview surface and export action."""

    def __init__(
        self,
        root: tk.Tk,
        workflow: ProductDesktopWorkflow,
        *,
        initial_input: Path | None = None,
    ) -> None:
        self.root = root
        self.workflow = workflow
        self.input_path: Path | None = None
        self.input_paths: tuple[Path, ...] = ()
        self.batch_inputs: tuple[DesktopBatchInput, ...] | None = None
        self.preview_images: list[ImageTk.PhotoImage] = []
        self.busy = False
        self.batch_active = False
        self._closing = False
        self._batch_cancel = threading.Event()
        self._batch_thread: threading.Thread | None = None
        self._batch_outcome: dict[str, Any] = {}
        self._batch_progress_queue: SimpleQueue[tuple[int, int, str]] = SimpleQueue()
        self._foreground_thread: threading.Thread | None = None
        self._foreground_outcome: dict[str, Any] = {}
        self._foreground_success: Callable[[Any], None] | None = None
        self.preview_ready = False
        self.style = tk.StringVar(value="")
        self.amount = tk.DoubleVar(value=1.0)
        self.output_format = tk.StringVar(value="png16")
        self.output_format_text = tk.StringVar(value="Single-photo output")
        self.input_text = tk.StringVar(value="No photo selected")
        self.status = tk.StringVar(
            value="Choose a photo to begin. No image leaves this computer."
        )
        self._configure_window()
        self._build()
        if initial_input is not None:
            self._set_input(Path(initial_input))

    def _configure_window(self) -> None:
        self.root.title("K-MCFM Look Approximation")
        self.root.geometry("1180x760")
        self.root.minsize(920, 680)
        self.root.configure(background="#0e110e")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0e110e")
        style.configure("Panel.TFrame", background="#191d19")
        style.configure(
            "Title.TLabel",
            background="#0e110e",
            foreground="#f2f5ef",
            font=("Segoe UI", 28, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#0e110e",
            foreground="#abb4a9",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabel",
            background="#191d19",
            foreground="#f2f5ef",
            font=("Segoe UI", 10),
        )
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(14, 9))
        style.configure(
            "Primary.TButton",
            background="#d7ff74",
            foreground="#11150f",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#e4ff9e"), ("disabled", "#555b50")],
            foreground=[("disabled", "#a2a79d")],
        )
        style.configure(
            "Look.TRadiobutton",
            background="#191d19",
            foreground="#f2f5ef",
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "Look.TRadiobutton",
            background=[("active", "#242a23")],
            foreground=[("selected", "#d7ff74")],
        )

        outer = ttk.Frame(self.root, padding=24, style="Root.TFrame")
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Choose the look, keep the photograph",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Deterministic, content-safe, film-inspired Look Approximations. "
                "Not calibrated stock responses."
            ),
            style="Body.TLabel",
        ).pack(anchor="w", pady=(4, 18))

        controls = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        controls.pack(fill="x")
        self.choose_button = ttk.Button(
            controls,
            text="Choose photos",
            command=self.choose_input,
            style="Secondary.TButton",
            takefocus=True,
        )
        self.choose_button.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 14))
        ttk.Label(controls, textvariable=self.input_text, style="Panel.TLabel").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(controls, text="Look strength", style="Panel.TLabel").grid(
            row=0, column=2, sticky="e", padx=(18, 8)
        )
        self.amount_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.amount,
            command=self._amount_changed,
            takefocus=True,
        )
        self.amount_scale.grid(row=0, column=3, sticky="ew")
        self.amount_label = ttk.Label(controls, text="100%", style="Panel.TLabel")
        self.amount_label.grid(row=0, column=4, padx=(8, 0))
        self.preview_button = ttk.Button(
            controls,
            text="Render three previews",
            command=self.render_previews,
            style="Secondary.TButton",
            takefocus=True,
        )
        self.preview_button.grid(
            row=1, column=3, columnspan=2, sticky="e", pady=(10, 0)
        )
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        self.cards = ttk.Frame(outer, style="Root.TFrame")
        self.cards.pack(fill="both", expand=True, pady=18)
        self.preview_labels: dict[str, ttk.Label] = {}
        self.look_buttons: dict[str, ttk.Radiobutton] = {}
        for column, row in enumerate(PRODUCT_LOOKS):
            card = ttk.Frame(self.cards, style="Panel.TFrame", padding=12)
            card.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 7, 0 if column == 2 else 7),
            )
            image_label = ttk.Label(
                card,
                text="Preview not rendered",
                anchor="center",
                style="Panel.TLabel",
            )
            image_label.pack(fill="both", expand=True)
            self.preview_labels[row["style_id"]] = image_label
            look_button = ttk.Radiobutton(
                card,
                text=row["display_name"],
                value=row["style_id"],
                variable=self.style,
                command=self._style_changed,
                state="disabled",
                style="Look.TRadiobutton",
                takefocus=True,
            )
            look_button.pack(anchor="w", pady=(10, 2))
            self.look_buttons[row["style_id"]] = look_button
            ttk.Label(card, text=row["process"], style="Panel.TLabel").pack(anchor="w")
            ttk.Label(card, text=row["claim"], style="Panel.TLabel").pack(
                anchor="w", pady=(3, 0)
            )
            self.cards.columnconfigure(column, weight=1, uniform="look")
        self.cards.rowconfigure(0, weight=1)

        footer = ttk.Frame(outer, style="Root.TFrame")
        footer.pack(fill="x")
        self.status_label = ttk.Label(
            footer,
            textvariable=self.status,
            style="Body.TLabel",
            wraplength=760,
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        self.format_frame = ttk.Frame(footer, style="Root.TFrame")
        self.format_frame.pack(side="right", padx=(12, 0))
        self.format_label = ttk.Label(
            self.format_frame,
            textvariable=self.output_format_text,
            style="Body.TLabel",
        )
        self.format_label.pack(anchor="e")
        self.output_format_buttons: dict[str, ttk.Radiobutton] = {}
        format_choices = ttk.Frame(self.format_frame, style="Root.TFrame")
        format_choices.pack(anchor="e")
        for row in PRODUCT_OUTPUT_FORMATS:
            button = ttk.Radiobutton(
                format_choices,
                text=row.display_name,
                value=row.format_id,
                variable=self.output_format,
                command=self._output_format_changed,
                takefocus=True,
            )
            button.pack(side="left", padx=(8, 0))
            self.output_format_buttons[row.format_id] = button
        self.export_button = ttk.Button(
            footer,
            text="Export PNG16 + recipe",
            command=self.export,
            state="disabled",
            style="Primary.TButton",
            takefocus=True,
        )
        self.export_button.pack(side="right", padx=(16, 0))
        self.cancel_button = ttk.Button(
            footer,
            text="Cancel batch",
            command=self.cancel_batch,
            state="disabled",
            style="Secondary.TButton",
            takefocus=True,
        )
        self.cancel_button.pack(side="right")
        self._update_output_format_controls()

    def _set_inputs(self, paths: tuple[Path, ...]) -> None:
        if not 1 <= len(paths) <= 100:
            raise ProductDesktopError("choose between one and 100 photos")
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if any(not path.is_file() for path in resolved):
            raise ProductDesktopError("input must be an existing file")
        if self.workflow.preview_state is not None:
            if not self.workflow.close():
                self.preview_ready = False
                self.export_button.configure(state="disabled")
                raise ProductDesktopError(
                    "preview workspace ownership changed; preserved for inspection"
                )
            self._clear_preview_widgets()
        self.input_paths = resolved
        self.input_path = resolved[0]
        self.batch_inputs = None
        self.input_text.set(
            resolved[0].name
            if len(resolved) == 1
            else f"{len(resolved)} photos · preview representative pending"
        )
        self.style.set("")
        self.preview_ready = False
        self.export_button.configure(state="disabled")
        if len(resolved) > 1:
            self.output_format.set("png16")
        self._update_export_label()
        self._update_output_format_controls()
        self.status.set(
            "Photo selected. Render previews to compare the three looks."
            if len(resolved) == 1
            else (
                f"{len(resolved)} photos selected. Render previews to bind the "
                "canonical representative and compare looks."
            )
        )

    def _set_input(self, path: Path) -> None:
        """Retain the existing one-photo setup helper."""

        self._set_inputs((Path(path),))

    def choose_input(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Choose one or more photos"
        )
        if selected:
            try:
                self._set_inputs(tuple(Path(path) for path in selected))
            except (OSError, ProductDesktopError) as exc:
                self._show_error(exc)

    def _amount_changed(self, _value: object = None) -> None:
        self.amount_label.configure(text=f"{round(self.amount.get() * 100):d}%")
        if self.workflow.preview_state is not None:
            self._invalidate_previews(
                "Strength changed. Render new previews before export."
            )

    def _invalidate_previews(self, message: str) -> None:
        self.style.set("")
        self.batch_inputs = None
        self.preview_ready = False
        self.export_button.configure(state="disabled")
        if self.workflow.preview_state is not None and not self.workflow.close():
            self._show_error(
                ProductDesktopError(
                    "preview workspace ownership changed; preserved for inspection"
                )
            )
            return
        self._clear_preview_widgets()
        self.status.set(message)

    def _clear_preview_widgets(self) -> None:
        self.preview_images.clear()
        for label in self.preview_labels.values():
            label.configure(image="", text="Preview not rendered")

    def _update_export_label(self) -> None:
        count = len(self.input_paths)
        output = product_output_format(self.output_format.get())
        self.export_button.configure(
            text=(
                f"Export {output.display_name} + recipe"
                if count <= 1
                else f"Export {count} PNG16 + recipes"
            )
        )

    def _update_output_format_controls(self) -> None:
        count = len(self.input_paths)
        single_photo = count == 1
        enabled = single_photo and not self.busy
        if not single_photo:
            self.output_format.set("png16")
        if count == 0:
            label = "Choose one photo for output format"
        elif single_photo:
            label = "Single-photo output"
        else:
            label = "Batch output fixed: PNG16"
        self.output_format_text.set(label)
        state = "normal" if enabled else "disabled"
        for button in self.output_format_buttons.values():
            button.configure(state=state)
        self._update_export_label()

    def _output_format_changed(self) -> None:
        output = product_output_format(self.output_format.get())
        self._update_export_label()
        if self.preview_ready:
            self.status.set(
                f"Output changed to {output.display_name}. Existing previews remain valid."
            )

    def _set_busy(self, busy: bool, message: str) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.choose_button.configure(state=state)
        self.preview_button.configure(state=state)
        self.amount_scale.configure(state=state)
        self._update_output_format_controls()
        look_state = "normal" if not busy and self.preview_ready else "disabled"
        for button in self.look_buttons.values():
            button.configure(state=look_state)
        self.export_button.configure(
            state=(
                "normal"
                if not busy
                and self.preview_ready
                and self.workflow.preview_state
                and self.style.get() in _LOOK_IDS
                else "disabled"
            )
        )
        self.cancel_button.configure(
            state="normal" if self.batch_active and busy else "disabled"
        )
        self.status.set(message)

    def _style_changed(self) -> None:
        if self.style.get() not in _LOOK_IDS:
            self.export_button.configure(state="disabled")
            return
        if self.preview_ready and not self.busy:
            self.export_button.configure(state="normal")
            self.status.set(
                "Look selected. Export a new "
                f"{product_output_format(self.output_format.get()).display_name} "
                "+ recipe pair."
                if len(self.input_paths) <= 1
                else (
                    f"Look selected for all {len(self.input_paths)} photos. "
                    "Export one new atomic batch folder."
                )
            )

    def _background(
        self, action: Callable[[], Any], success: Callable[[Any], None]
    ) -> None:
        if self._foreground_thread is not None:
            raise ProductDesktopError("a foreground worker is already active")
        self._foreground_outcome = {}
        self._foreground_success = success

        def worker() -> None:
            try:
                self._foreground_outcome["result"] = action()
            except BaseException as exc:  # noqa: BLE001 - returned to Tk thread
                self._foreground_outcome["error"] = exc

        self._foreground_thread = threading.Thread(
            target=worker,
            name="kmcfm-desktop-foreground",
            daemon=False,
        )
        self._foreground_thread.start()
        self.root.after(50, self._poll_foreground)

    def _poll_foreground(self) -> None:
        worker = self._foreground_thread
        if worker is None:
            return
        if worker.is_alive():
            self.root.after(50, self._poll_foreground)
            return
        worker.join(timeout=0)
        self._foreground_thread = None
        error = self._foreground_outcome.pop("error", None)
        has_result = "result" in self._foreground_outcome
        result = self._foreground_outcome.pop("result", None)
        success = self._foreground_success
        self._foreground_success = None
        if self._closing:
            self._finish_close()
            return
        if error is not None:
            self._show_error(error)
            return
        if not has_result or success is None:
            self._show_error(ProductDesktopError("foreground worker returned no result"))
            return
        try:
            success(result)
        except Exception as exc:  # noqa: BLE001 - report UI completion failures
            self._show_error(exc)

    def render_previews(self) -> None:
        if self.busy:
            return
        if self.input_path is None:
            self._show_error(
                ProductDesktopError("choose a photo before rendering previews")
            )
            return
        amount = self.amount.get()
        sources = self.input_paths
        self._set_busy(True, "Rendering three bounded previews…")
        self._background(
            lambda: self.workflow.render_batch_previews(sources, amount),
            self._batch_preview_complete,
        )

    def _batch_preview_complete(
        self,
        result: tuple[DesktopPreviewState, tuple[DesktopBatchInput, ...]],
    ) -> None:
        state, inputs = result
        self.batch_inputs = inputs
        self.input_paths = tuple(row.path for row in inputs)
        self.input_path = inputs[0].path
        self.input_text.set(
            inputs[0].basename
            if len(inputs) == 1
            else f"{len(inputs)} photos · previewing {inputs[0].basename}"
        )
        self._update_export_label()
        self._preview_complete(state)

    def _preview_complete(self, state: DesktopPreviewState) -> None:
        if self.batch_inputs is None:
            self.batch_inputs = self.workflow.bind_batch_inputs((state.input_path,))
            self.input_paths = (state.input_path,)
            self.input_path = state.input_path
        self.preview_images.clear()
        preview_bytes = self.workflow.preview_bytes()
        for look in PRODUCT_LOOKS:
            with Image.open(BytesIO(preview_bytes[look["style_id"]])) as opened:
                image = opened.copy()
            image.thumbnail(PRODUCT_PREVIEW_DISPLAY_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview_images.append(photo)
            self.preview_labels[look["style_id"]].configure(image=photo, text="")
        self.preview_ready = True
        self._set_busy(
            False,
            (
                "Previews ready. Select one look and export a new "
                f"{product_output_format(self.output_format.get()).display_name} "
                "+ recipe pair."
                if len(self.input_paths) <= 1
                else (
                    f"Representative preview ready for {len(self.input_paths)} photos. "
                    "Select one look for the whole batch."
                )
            ),
        )

    def export(self) -> None:
        if self.busy or not self.preview_ready or self.workflow.preview_state is None:
            return
        state = self.workflow.preview_state
        selected = self.style.get()
        if selected not in _LOOK_IDS:
            self._show_error(ProductDesktopError("select one look before export"))
            return
        if self.batch_inputs is None:
            self._show_error(ProductDesktopError("render previews before exporting"))
            return
        if len(self.batch_inputs) > 1:
            destination = filedialog.asksaveasfilename(
                title="Choose a new batch folder",
                initialfile=f"kmcfm-{selected}-{len(self.batch_inputs)}-photos",
                filetypes=(("Batch folder name", "*"),),
            )
            if not destination:
                return
            self._batch_cancel.clear()
            self.batch_active = True
            self._set_busy(
                True,
                f"Rendering 0/{len(self.batch_inputs)} photos…",
            )
            inputs = self.batch_inputs
            self._background_batch(
                lambda: self.workflow.export_batch(
                    inputs,
                    selected,
                    Path(destination),
                    cancel_event=self._batch_cancel,
                    progress=self._batch_progress,
                ),
                self._batch_complete,
            )
            return
        output = product_output_format(self.output_format.get())
        destination = filedialog.asksaveasfilename(
            title="Export Look Approximation",
            defaultextension=output.canonical_extension,
            filetypes=(
                (
                    output.display_name,
                    " ".join(f"*{suffix}" for suffix in output.accepted_extensions),
                ),
            ),
            initialfile=(
                f"{state.input_path.stem}-{selected}{output.canonical_extension}"
            ),
        )
        if not destination:
            return
        output_format_id = output.format_id
        self._set_busy(
            True,
            f"Rendering full-resolution {output.display_name} and strict recipe…",
        )
        self._background(
            lambda: self.workflow.export(
                selected,
                Path(destination),
                output_format_id=output_format_id,
            ),
            self._export_complete,
        )

    def _batch_progress(self, completed: int, total: int, basename: str) -> None:
        self._batch_progress_queue.put((completed, total, basename))

    def _drain_batch_progress(self) -> None:
        latest: tuple[int, int, str] | None = None
        while True:
            try:
                latest = self._batch_progress_queue.get_nowait()
            except Empty:
                break
        if latest is not None:
            completed, total, basename = latest
            self.status.set(f"Rendered {completed}/{total}: {basename}")

    def _background_batch(
        self,
        action: Callable[[], DesktopBatchReceipt],
        success: Callable[[DesktopBatchReceipt], None],
    ) -> None:
        if self._batch_thread is not None:
            raise ProductDesktopError("a batch worker is already active")
        self._batch_outcome = {}
        self._drain_batch_progress()

        def worker() -> None:
            try:
                self._batch_outcome["result"] = action()
            except BaseException as exc:  # noqa: BLE001 - returned to Tk thread
                self._batch_outcome["error"] = exc

        self._batch_thread = threading.Thread(
            target=worker,
            name="kmcfm-desktop-batch",
            daemon=False,
        )
        self._batch_thread.start()
        self.root.after(50, lambda: self._poll_batch(success))

    def _poll_batch(
        self,
        success: Callable[[DesktopBatchReceipt], None],
    ) -> None:
        worker = self._batch_thread
        if worker is None:
            return
        self._drain_batch_progress()
        if worker.is_alive():
            self.root.after(50, lambda: self._poll_batch(success))
            return
        worker.join(timeout=0)
        self._batch_thread = None
        self.batch_active = False
        error = self._batch_outcome.pop("error", None)
        result = self._batch_outcome.pop("result", None)
        if error is not None:
            if self._closing:
                self._finish_close()
            else:
                self._show_error(error)
            return
        if not isinstance(result, DesktopBatchReceipt):
            if self._closing:
                self._finish_close()
            else:
                self._show_error(ProductDesktopError("batch worker returned no receipt"))
            return
        if self._closing:
            self._finish_close()
            return
        success(result)

    def cancel_batch(self) -> None:
        if self._batch_thread is None or not self._batch_thread.is_alive():
            return
        self._batch_cancel.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Stopping safely after the current photo…")

    def _batch_complete(self, receipt: DesktopBatchReceipt) -> None:
        self._set_busy(
            False,
            (
                f"Batch complete: {receipt.job_count} photos in "
                f"{receipt.output_directory.name}"
            ),
        )
        messagebox.showinfo(
            "Batch export complete",
            (
                f"Created {receipt.job_count} PNG16 + recipe pairs and "
                f"{receipt.receipt_path.name}.\n\n"
                "film-inspired / Look Approximation"
            ),
        )

    def _export_complete(self, receipt: DesktopExportReceipt) -> None:
        self._set_busy(
            False,
            f"Export complete: {receipt.output_path.name} + {receipt.recipe_path.name}",
        )
        messagebox.showinfo(
            "Export complete",
            (
                f"Created {receipt.output_path.name} and "
                f"{receipt.recipe_path.name}.\n\n"
                "film-inspired / Look Approximation"
            ),
        )

    def _show_error(self, exc: BaseException) -> None:
        self.style.set("")
        self.preview_ready = False
        self._set_busy(False, f"Stopped safely: {exc}")
        messagebox.showerror("K-MCFM stopped safely", str(exc))

    def close(self) -> None:
        if self._foreground_thread is not None:
            self._closing = True
            self._set_busy(True, "Closing safely after the current operation…")
            self.cancel_button.configure(state="disabled")
            if not self._foreground_thread.is_alive():
                self._poll_foreground()
            return
        if self._batch_thread is not None:
            if self._batch_thread.is_alive():
                self._closing = True
                self._batch_cancel.set()
                self._set_busy(True, "Closing safely after the current photo…")
                self.cancel_button.configure(state="disabled")
                return
            self._batch_thread.join(timeout=0)
            self._batch_thread = None
            self.batch_active = False
        self._finish_close()

    def _finish_close(self) -> None:
        self.workflow.close()
        if self.root.winfo_exists():
            # Release Tk-owned PhotoImage objects while their interpreter is
            # still alive.  Retaining them across root destruction can make a
            # subsequent desktop session race their delayed finalizers.
            self._clear_preview_widgets()
            self.root.destroy()


def build_product_desktop_app(
    root_window: tk.Tk,
    workflow: ProductDesktopWorkflow,
    *,
    initial_input: Path | None = None,
) -> ProductDesktopApp:
    """Build the native product app without adding a second rendering path."""

    return ProductDesktopApp(
        root_window,
        workflow,
        initial_input=initial_input,
    )


__all__ = [
    "ProductDesktopApp",
    "build_product_desktop_app",
]

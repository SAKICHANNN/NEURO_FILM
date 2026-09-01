"""Tkinter view for the bounded new-input Look Approximation workflow."""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from .product_desktop import (
    PRODUCT_LOOKS,
    DesktopExportReceipt,
    DesktopPreviewState,
    ProductDesktopError,
    ProductDesktopWorkflow,
)

_LOOK_IDS = tuple(row["style_id"] for row in PRODUCT_LOOKS)
_PREVIEW_DISPLAY_SIZE = (300, 260)


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
        self.preview_images: list[ImageTk.PhotoImage] = []
        self.busy = False
        self.preview_ready = False
        self.style = tk.StringVar(value=_LOOK_IDS[0])
        self.amount = tk.DoubleVar(value=1.0)
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
            text="Choose photo",
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
            ttk.Radiobutton(
                card,
                text=row["display_name"],
                value=row["style_id"],
                variable=self.style,
                style="Look.TRadiobutton",
                takefocus=True,
            ).pack(anchor="w", pady=(10, 2))
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
        self.export_button = ttk.Button(
            footer,
            text="Export PNG16 + recipe",
            command=self.export,
            state="disabled",
            style="Primary.TButton",
            takefocus=True,
        )
        self.export_button.pack(side="right", padx=(16, 0))

    def _set_input(self, path: Path) -> None:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ProductDesktopError("input must be an existing file")
        if self.workflow.preview_state is not None:
            if not self.workflow.close():
                self.preview_ready = False
                self.export_button.configure(state="disabled")
                raise ProductDesktopError(
                    "preview workspace ownership changed; preserved for inspection"
                )
            self._clear_preview_widgets()
        self.input_path = resolved
        self.input_text.set(resolved.name)
        self.preview_ready = False
        self.export_button.configure(state="disabled")
        self.status.set("Photo selected. Render previews to compare the three looks.")

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(title="Choose a photo")
        if selected:
            try:
                self._set_input(Path(selected))
            except (OSError, ProductDesktopError) as exc:
                self._show_error(exc)

    def _amount_changed(self, _value: object = None) -> None:
        self.amount_label.configure(text=f"{round(self.amount.get() * 100):d}%")
        if self.workflow.preview_state is not None:
            self._invalidate_previews(
                "Strength changed. Render new previews before export."
            )

    def _invalidate_previews(self, message: str) -> None:
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

    def _set_busy(self, busy: bool, message: str) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.choose_button.configure(state=state)
        self.preview_button.configure(state=state)
        self.amount_scale.configure(state=state)
        self.export_button.configure(
            state=(
                "normal"
                if not busy and self.preview_ready and self.workflow.preview_state
                else "disabled"
            )
        )
        self.status.set(message)

    def _background(
        self, action: Callable[[], Any], success: Callable[[Any], None]
    ) -> None:
        def worker() -> None:
            try:
                result = action()
            except Exception as exc:  # noqa: BLE001 - report worker failures in UI
                self.root.after(0, lambda error=exc: self._show_error(error))
            else:

                def finish_success() -> None:
                    try:
                        success(result)
                    except Exception as exc:  # noqa: BLE001 - report UI completion failures
                        self._show_error(exc)

                self.root.after(0, finish_success)

        threading.Thread(target=worker, daemon=True).start()

    def render_previews(self) -> None:
        if self.busy:
            return
        if self.input_path is None:
            self._show_error(
                ProductDesktopError("choose a photo before rendering previews")
            )
            return
        amount = self.amount.get()
        source = self.input_path
        self._set_busy(True, "Rendering three bounded previews…")
        self._background(
            lambda: self.workflow.render_previews(source, amount),
            self._preview_complete,
        )

    def _preview_complete(self, state: DesktopPreviewState) -> None:
        self.preview_images.clear()
        preview_bytes = self.workflow.preview_bytes()
        for look in PRODUCT_LOOKS:
            with Image.open(BytesIO(preview_bytes[look["style_id"]])) as opened:
                image = opened.copy()
            image.thumbnail(_PREVIEW_DISPLAY_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview_images.append(photo)
            self.preview_labels[look["style_id"]].configure(image=photo, text="")
        self.preview_ready = True
        self._set_busy(
            False,
            "Previews ready. Select one look and export a new PNG16 + recipe pair.",
        )

    def export(self) -> None:
        if self.busy or self.workflow.preview_state is None:
            return
        state = self.workflow.preview_state
        selected = self.style.get()
        destination = filedialog.asksaveasfilename(
            title="Export Look Approximation",
            defaultextension=".png",
            filetypes=(("16-bit PNG", "*.png"),),
            initialfile=f"{state.input_path.stem}-{selected}.png",
        )
        if not destination:
            return
        self._set_busy(True, "Rendering full-resolution PNG16 and strict recipe…")
        self._background(
            lambda: self.workflow.export(selected, Path(destination)),
            self._export_complete,
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
        self.preview_ready = False
        self._set_busy(False, f"Stopped safely: {exc}")
        messagebox.showerror("K-MCFM stopped safely", str(exc))

    def close(self) -> None:
        self.workflow.close()
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

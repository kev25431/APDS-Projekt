from __future__ import annotations

from datetime import date
from tkinter import ttk

import customtkinter as ctk
from tkcalendar import DateEntry

from .config import PALETTE


DARK_DATE_ENTRY_STYLE = "Dark.DateEntry"


def configure_dark_date_entry_style() -> str:
    style = ttk.Style()
    try:
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        DARK_DATE_ENTRY_STYLE,
        fieldbackground=PALETTE["surface_alt"],
        background=PALETTE["surface_alt"],
        foreground=PALETTE["text"],
        arrowcolor=PALETTE["text"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["border"],
        darkcolor=PALETTE["border"],
        insertcolor=PALETTE["text"],
        padding=4,
        relief="flat",
    )
    style.map(
        DARK_DATE_ENTRY_STYLE,
        fieldbackground=[
            ("readonly", PALETTE["surface_alt"]),
            ("focus", PALETTE["surface_alt"]),
            ("!disabled", PALETTE["surface_alt"]),
        ],
        background=[
            ("active", PALETTE["surface_alt"]),
            ("pressed", PALETTE["surface_alt"]),
            ("!disabled", PALETTE["surface_alt"]),
        ],
        foreground=[
            ("readonly", PALETTE["text"]),
            ("focus", PALETTE["text"]),
            ("!disabled", PALETTE["text"]),
        ],
        arrowcolor=[
            ("active", PALETTE["accent"]),
            ("pressed", PALETTE["accent"]),
            ("!disabled", PALETTE["text"]),
        ],
    )
    return DARK_DATE_ENTRY_STYLE


def dark_date_entry_options() -> dict[str, object]:
    return {
        "background": PALETTE["accent"],
        "foreground": "white",
        "disabledforeground": PALETTE["muted"],
        "bordercolor": PALETTE["border"],
        "headersbackground": PALETTE["surface_alt"],
        "headersforeground": PALETTE["text"],
        "normalbackground": PALETTE["surface"],
        "normalforeground": PALETTE["text"],
        "weekendbackground": PALETTE["surface_alt"],
        "weekendforeground": PALETTE["text"],
        "othermonthbackground": PALETTE["surface_soft"],
        "othermonthforeground": PALETTE["muted"],
        "othermonthwebackground": PALETTE["surface_soft"],
        "othermonthweforeground": PALETTE["muted"],
        "selectbackground": PALETTE["accent"],
        "selectforeground": "white",
        "borderwidth": 0,
        "style": configure_dark_date_entry_style(),
    }


class DatePicker(ctk.CTkFrame):
    def __init__(self, master, title: str, initial: date):
        super().__init__(master, fg_color=PALETTE["surface"], corner_radius=18, border_width=1, border_color=PALETTE["border"])
        style_name = configure_dark_date_entry_style()
        ctk.CTkLabel(
            self,
            text=title,
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        self.entry = DateEntry(
            self,
            date_pattern="yyyy-mm-dd",
            year=initial.year,
            month=initial.month,
            day=initial.day,
            **dark_date_entry_options(),
        )
        self.entry.configure(style=style_name, foreground=PALETTE["text"])
        self.entry.pack(fill="x", padx=14, pady=(0, 12))

    def get_date(self) -> date:
        return self.entry.get_date()

    def set_date(self, selected: date) -> None:
        self.entry.set_date(selected)


class AnimatedLineList(ctk.CTkFrame):
    def __init__(self, master, command=None, height: int = 150):
        super().__init__(master, fg_color=PALETTE["surface_alt"], corner_radius=18, border_width=1, border_color=PALETTE["border"])
        self.command = command
        self.items: list[str] = []
        self.buttons: dict[int, ctk.CTkButton] = {}
        self.selected_index = -1
        self.render_generation = 0
        self.keyboard_active = False

        self.top_gradient = ctk.CTkFrame(self, fg_color=PALETTE["surface_alt"], height=6, corner_radius=999)
        self.top_gradient.pack(fill="x", padx=10, pady=(8, 0))
        self.top_gradient.pack_propagate(False)

        self.scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=height,
            scrollbar_button_color=PALETTE["accent"],
            scrollbar_button_hover_color=PALETTE["accent_dark"],
        )
        self.scroll.pack(fill="both", expand=True, padx=10, pady=6)

        self.bottom_gradient = ctk.CTkFrame(self, fg_color=PALETTE["accent_soft"], height=8, corner_radius=999)
        self.bottom_gradient.pack(fill="x", padx=10, pady=(0, 8))
        self.bottom_gradient.pack_propagate(False)

        for widget in (self, self.scroll):
            widget.bind("<Enter>", self._activate_keyboard, add="+")
            widget.bind("<Leave>", self._deactivate_keyboard, add="+")
        self.scroll.bind("<MouseWheel>", lambda _event: self.after(20, self._update_gradients), add="+")
        self.after(0, self._bind_keyboard_events)

    def _bind_keyboard_events(self) -> None:
        root = self.winfo_toplevel()
        for sequence, handler in (
            ("<Up>", self._handle_up),
            ("<Down>", self._handle_down),
            ("<Return>", self._handle_enter),
            ("<Tab>", self._handle_down),
            ("<Shift-Tab>", self._handle_up),
        ):
            root.bind(sequence, handler, add="+")

    def set_items(self, items: list[str]) -> None:
        self.render_generation += 1
        generation = self.render_generation
        self.items = items
        self.buttons = {}
        self.selected_index = -1
        for child in self.scroll.winfo_children():
            child.destroy()
        for index, item in enumerate(items):
            self.after(index * 18, lambda idx=index, label=item, gen=generation: self._create_item(idx, label, gen))
        self.after(max(len(items), 1) * 18 + 40, self._update_gradients)

    def selected_item(self) -> str | None:
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def _create_item(self, index: int, item: str, generation: int) -> None:
        if generation != self.render_generation:
            return
        button = ctk.CTkButton(
            self.scroll,
            text=item,
            anchor="w",
            height=42,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            hover_color=PALETTE["accent_soft"],
            border_width=1,
            border_color=PALETTE["border"],
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            command=lambda idx=index: self._choose(idx),
        )
        button.pack(fill="x", pady=(0, 8))
        button.bind("<Enter>", lambda _event, idx=index: (self._activate_keyboard(_event), self._select(idx)), add="+")
        self.buttons[index] = button

    def _activate_keyboard(self, _event=None) -> None:
        self.keyboard_active = True
        self.focus_set()

    def _deactivate_keyboard(self, _event=None) -> None:
        self.keyboard_active = False

    def _select(self, index: int) -> None:
        if not 0 <= index < len(self.items):
            return
        self.selected_index = index
        for item_index, button in self.buttons.items():
            is_selected = item_index == index
            button.configure(
                fg_color=PALETTE["accent"] if is_selected else PALETTE["surface"],
                hover_color=PALETTE["accent_dark"] if is_selected else PALETTE["accent_soft"],
                border_color=PALETTE["accent"] if is_selected else PALETTE["border"],
                text_color="white" if is_selected else PALETTE["text"],
            )

    def _choose(self, index: int) -> None:
        self._select(index)
        if self.command and 0 <= index < len(self.items):
            self.command(self.items[index], index)

    def _move(self, delta: int) -> str:
        if not self.items:
            return "break"
        next_index = self.selected_index + delta if self.selected_index >= 0 else 0
        next_index = min(max(next_index, 0), len(self.items) - 1)
        self._select(next_index)
        self._scroll_to_index(next_index)
        return "break"

    def _handle_up(self, _event=None) -> str | None:
        if not self.keyboard_active:
            return None
        return self._move(-1)

    def _handle_down(self, _event=None) -> str | None:
        if not self.keyboard_active:
            return None
        return self._move(1)

    def _handle_enter(self, _event=None) -> str | None:
        if not self.keyboard_active:
            return None
        if 0 <= self.selected_index < len(self.items):
            self._choose(self.selected_index)
        return "break"

    def _scroll_to_index(self, index: int) -> None:
        try:
            canvas = self.scroll._parent_canvas
            fraction = index / max(len(self.items) - 1, 1)
            canvas.yview_moveto(min(max(fraction - 0.08, 0), 1))
        except Exception:
            pass
        self.after(20, self._update_gradients)

    def _update_gradients(self) -> None:
        try:
            first, last = self.scroll._parent_canvas.yview()
        except Exception:
            first, last = 0, 1
        self.top_gradient.configure(fg_color=PALETTE["accent_soft"] if first > 0.02 else PALETTE["surface_alt"])
        self.bottom_gradient.configure(fg_color=PALETTE["accent_soft"] if last < 0.98 else PALETTE["surface_alt"])

#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

class DataTab(ttk.Frame):
    """Placeholder tab for ‘Data’ actions and summaries."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._build_ui()

    def _build_ui(self):
        # Example layout: a label and a button
        lbl = ttk.Label(self, text="Data Tab", font=("Helvetica", 16, "bold"))
        lbl.pack(padx=20, pady=(20, 10))
        btn = ttk.Button(self, text="Refresh Data", command=self._on_refresh)
        btn.pack(padx=20, pady=10, anchor="w")

    def _on_refresh(self):
        # TODO: put your data‐loading logic here
        tk.messagebox.showinfo("DataTab", "Refresh Data clicked")
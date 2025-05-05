#!/usr/bin/env python3
"""
Dark theme styling module for Tkinter GUIs.

Usage:
    import tkinter as tk
    from dark_theme import apply_dark_theme

    root = tk.Tk()
    apply_dark_theme(root)
    # build the rest of your GUI...
"""
import tkinter as tk
from tkinter import ttk


def apply_dark_theme(root):
    """
    Apply a dark color scheme to the given Tkinter root or Toplevel window.
    """
    # Define theme colors
    bg_color       = '#1e1e1e'  # dark background
    fg_color       = '#e0e0e0'  # light text
    button_bg      = '#2d2d2d'
    button_fg      = '#e0e0e0'
    entry_bg       = '#2a2a2a'
    entry_fg       = '#ffffff'
    highlight_color= '#5e9cff'
    border_color   = '#444444'

    # Global widget defaults
    root.configure(bg=bg_color)
    root.option_add('*Background',     bg_color)
    root.option_add('*Foreground',     fg_color)
    root.option_add('*Font',           ('Segoe UI', 10))
    root.option_add('*Button.Background', button_bg)
    root.option_add('*Button.Foreground', button_fg)
    root.option_add('*Entry.Background',  entry_bg)
    root.option_add('*Entry.Foreground',  entry_fg)
    root.option_add('*Text.Background',   entry_bg)
    root.option_add('*Text.Foreground',   entry_fg)
    root.option_add('*Listbox.Background', entry_bg)
    root.option_add('*Listbox.Foreground', fg_color)
    root.option_add('*Scale.Background',   bg_color)
    root.option_add('*Scale.TroughColor',  border_color)
    root.option_add('*Checkbutton.Background', bg_color)
    root.option_add('*Radiobutton.Background', bg_color)
    root.option_add('*Menu.background',     button_bg)
    root.option_add('*Menu.foreground',     fg_color)

    # ttk styling
    style = ttk.Style(root)
    style.theme_use('clam')

    # TFrame
    style.configure('TFrame', background=bg_color)

    # TLabel
    style.configure('TLabel', background=bg_color, foreground=fg_color)

    # TButton
    style.configure('TButton', background=button_bg, foreground=button_fg, bordercolor=border_color)
    style.map('TButton',
              background=[('active', highlight_color)],
              foreground=[('disabled', '#888888')])

    # TEntry, TCombobox
    style.configure('TEntry', fieldbackground=entry_bg, foreground=entry_fg, bordercolor=border_color)
    style.configure('TCombobox', fieldbackground=entry_bg, background=entry_bg,
                    foreground=entry_fg, bordercolor=border_color)
    style.map('TCombobox',
              fieldbackground=[('readonly', entry_bg)],
              background=[('readonly', entry_bg)],
              foreground=[('readonly', entry_fg)])

    # TScale
    style.configure('Horizontal.TScale', background=bg_color)

    # TCheckbutton, TRadiobutton
    style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
    style.configure('TRadiobutton', background=bg_color, foreground=fg_color)

    # TMenubutton
    style.configure('TMenubutton', background=button_bg, foreground=button_fg)

    # Return style in case further tweaks are needed
    return style

# If imported directly, demonstrate usage
if __name__ == '__main__':
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    apply_dark_theme(root)
    ttk.Label(root, text="Dark Theme Demo").pack(padx=20, pady=20)
    ttk.Button(root, text="Click Me").pack(padx=20, pady=10)
    root.mainloop()

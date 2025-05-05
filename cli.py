import tkinter as tk
from tkinter import ttk

from gui.data_tab import DataTab
from gui.verification_tab import VerificationTab
from themes.dark_theme import apply_dark_theme
from gui.labeling_tab import LabelingTab
from gui.training_tab.training_tab import TrainingTab

def main():
    root = tk.Tk()
    apply_dark_theme(root)
    root.title("XP Labeler & Trainer")

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Labeling tab
    label_frame = ttk.Frame(notebook)
    notebook.add(label_frame, text='Labeling')
    LabelingTab(label_frame)

    # Training tab
    train_frame = ttk.Frame(notebook)
    notebook.add(train_frame, text='Training')
    train_tab = TrainingTab(train_frame)
    train_tab.pack(fill='both', expand=True)

    # Data tab
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text='Data')
    data_tab = DataTab(data_frame)
    data_tab.pack(fill='both', expand=True)

    # Verification tab
    verification_frame = ttk.Frame(notebook)
    notebook.add(verification_frame, text='Verification')
    verification_tab = VerificationTab(verification_frame)
    verification_tab.pack(fill='both', expand=True)

    root.mainloop()

if __name__ == '__main__':
    main()

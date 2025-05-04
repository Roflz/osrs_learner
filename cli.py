import tkinter as tk
from tkinter import ttk
from themes.dark_theme import apply_dark_theme
from gui.labeling_tab import LabelingTab
from gui.training_tab import TrainingTab

def main():
    root = tk.Tk()
    apply_dark_theme(root)
    root.title("XP Labeler & Trainer")

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    label_frame = ttk.Frame(notebook)
    train_frame = ttk.Frame(notebook)
    notebook.add(label_frame, text='Labeling')
    notebook.add(train_frame, text='Training')

    LabelingTab(label_frame)
    TrainingTab(train_frame)

    root.mainloop()

if __name__ == '__main__':
    main()
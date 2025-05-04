# gui/training_tab.py

import os
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

class TrainingTab(ttk.Frame):
    def __init__(self, parent, on_training_complete):
        """
        on_training_complete: callback to reload models in the labeling tab
        """
        super().__init__(parent)
        self.on_training_complete = on_training_complete
        self._build_ui()
        self._training_thread = None

    def _build_ui(self):
        # Inputs
        self.batch_size = tk.IntVar(value=16)
        self.epochs     = tk.IntVar(value=10)
        self.lr         = tk.DoubleVar(value=0.01)

        frm = ttk.Frame(self)
        frm.grid(row=0, column=0, pady=10, sticky='ew')
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Batch size:").grid(row=0, column=0, sticky='e')
        ttk.Entry(frm, textvariable=self.batch_size, width=6).grid(row=0, column=1, sticky='w')

        ttk.Label(frm, text="Epochs:").grid(row=1, column=0, sticky='e')
        ttk.Entry(frm, textvariable=self.epochs, width=6).grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text="Learning rate:").grid(row=2, column=0, sticky='e')
        ttk.Entry(frm, textvariable=self.lr, width=6).grid(row=2, column=1, sticky='w')

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=1, column=0, pady=5, sticky='ew')
        ttk.Button(btn_frame, text="Start YOLO Train", command=self._start_yolo).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Start CRNN Train", command=self._start_crnn).pack(side='left', padx=5)

        # Log output
        self.log_text = tk.Text(self, height=20, bg='#1e1e1e', fg='#e0e0e0')
        self.log_text.grid(row=2, column=0, sticky='nsew')
        self.rowconfigure(2, weight=1)

    def _append_log(self, line):
        self.log_text.insert('end', line)
        self.log_text.see('end')

    def _run_script(self, script_path):
        """
        Runs a .bat or shell script, streaming stdout/stderr into the text widget.
        """
        proc = subprocess.Popen(
            [script_path,
             str(self.batch_size.get()),
             str(self.epochs.get()),
             str(self.lr.get())],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        for line in proc.stdout:
            self.log_text.after(0, self._append_log, line)
        proc.wait()
        return proc.returncode

    def _start_yolo(self):
        if self._training_thread and self._training_thread.is_alive():
            messagebox.showwarning("Training in progress", "Please wait for the current training to finish.")
            return
        self.log_text.delete('1.0', 'end')
        self._training_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._training_thread.start()

    def _yolo_worker(self):
        self.log_text.after(0, self._append_log, "=== Starting YOLO training ===\n")
        ret = self._run_script("run.bat")
        if ret == 0:
            self.log_text.after(0, self._append_log, "\nYOLO training completed successfully.\n")
            self.on_training_complete('yolo')
        else:
            self.log_text.after(0, self._append_log, f"\nYOLO training failed (exit {ret}).\n")

    def _start_crnn(self):
        if self._training_thread and self._training_thread.is_alive():
            messagebox.showwarning("Training in progress", "Please wait for the current training to finish.")
            return
        self.log_text.delete('1.0', 'end')
        self._training_thread = threading.Thread(target=self._crnn_worker, daemon=True)
        self._training_thread.start()

    def _crnn_worker(self):
        self.log_text.after(0, self._append_log, "=== Starting CRNN training ===\n")
        ret = self._run_script("minimal_crnn_ctc.bat")
        if ret == 0:
            self.log_text.after(0, self._append_log, "\nCRNN training completed successfully.\n")
            self.on_training_complete('crnn')
        else:
            self.log_text.after(0, self._append_log, f"\nCRNN training failed (exit {ret}).\n")

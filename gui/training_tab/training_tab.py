import threading
from tkinter import ttk
from config import SKILLS
from .training_ui import build_training_ui
from .stats_utils import update_stats


class TrainingTab(ttk.Frame):
    def __init__(self, parent):
        """
        Initialize the unified training tab, set up threads and UI, and load initial statistics.
        """
        super().__init__(parent)
        # Thread and subprocess state
        self._thread = None
        self._current_process = None
        self._stop_requested = False
        self.parent = parent

        # Build the UI components (form, stats panel, controls, log)
        self._build_ui()
        # Load initial data statistics into the stats panel
        update_stats(self)
        self._pending_steps = []  # holds (title, func) tuples when using "Run All"

    def _build_ui(self):
        build_training_ui(self)

    def _on_mousewheel(self, event):
        self.form_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')

    def _on_run_step(self, step_title):
        mapping = {
            'Split Real Data': self._run_split_real,
            'Synthetic Numbers': self._run_synth_numbers,
            'Synthetic Skills': self._run_synth_skills,
            'Split Synth Numbers': self._run_split_synth,
            'Split Synth Skills': self._run_split_synth_skills,
            'YOLO Training': self._run_yolo_train,
            'CRNN Training': self._run_crnn_train
        }
        func = mapping.get(step_title)
        if func:
            threading.Thread(target=func, daemon=True).start()

    def _on_run_all(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_requested = False
        self.btn_run_all.config(state='disabled')
        self.btn_stop   .config(state='normal')
        self.log.delete('1.0', 'end')

        # enqueue the steps in order
        self._pending_steps = [
            ('Split Real Data',     self._run_split_real),
            ('Make Synth Numbers',  self._run_synth_numbers),
            ('Make Synth Skills',   self._run_synth_skills),
            ('Split Synth Numbers', self._run_split_synth),
            ('Split Synth Skills',  self._run_split_synth_skills),
            ('YOLO Training',       self._run_yolo_train),
            ('CRNN Training',       self._run_crnn_train),
        ]

        # kick off the first one
        title, func = self._pending_steps.pop(0)
        self.log.insert('end', f"=== Starting {title} ===\n")
        func()

    def _run_all_sequence(self):
        for title, func in self.all_funcs:
            if self._stop_requested:
                self.log.insert('end', f'=== Aborted before {title} ===\n')
                break

            # announce
            self.log.insert('end', f'=== Starting {title} ===\n')
            self.log.see('end')

            # run the step; it should block until complete
            try:
                func()
            except Exception as e:
                self.log.insert('end', f'Error in {title}: {e}\n')
                break

            # note completion
            self.log.insert('end', f'=== Done {title} ===\n\n')
            self.log.see('end')

        # reset buttons
        self.btn_stop.config(state='disabled')
        self.btn_run_all.config(state='normal')

    def _on_stop_pipeline(self):
        self._stop_requested = True
        if self._current_process and self._current_process.poll() is None:
            try: self._current_process.terminate()
            except: pass
        self.btn_stop.config(state='disabled')

    # ---- Individual step subprocess wrappers for training_tab.py ----
    def _run_split_real(self):
        from gui.training_tab.training_utils import build_split_real_cmd
        cmd = build_split_real_cmd(self.src_real.get(), self.pct_real.get())
        self._run_subprocess("Split Real", cmd)

    def _run_synth_numbers(self):
        from gui.training_tab.training_utils import build_synth_number_cmd
        cfg = {
            'num_images': self.syn_num.get(),
            'neg_ratio': self.syn_neg.get(),
            'min_seq': self.syn_min_seq.get(),
            'max_seq': self.syn_max_seq.get(),
            'min_digits': self.syn_min_digits.get(),
            'max_digits': self.syn_max_digits.get(),
            'use_weights': self.syn_use_weights.get(),
            'max_weight': self.syn_max_weight.get(),
        }
        cmd = build_synth_number_cmd(cfg)
        self._run_subprocess("Synth Numbers", cmd)

    def _run_synth_skills(self):
        from gui.training_tab.training_utils import build_synth_skill_cmd
        cfg = {
            'num_images': self.skill_num.get(),
            'max_icons': self.skill_max_icons.get(),
            'use_weights': self.skill_use_weights.get(),
            'max_weight': self.skill_max_weight.get(),
        }
        cmd = build_synth_skill_cmd(cfg)
        self._run_subprocess("Synth Skills", cmd)

    def _run_split_synth(self):
        from gui.training_tab.training_utils import build_split_synth_cmd
        cmd = build_split_synth_cmd(self.synth_split_pct.get(), mode='numbers')
        self._run_subprocess("Split Synth Numbers", cmd)

    def _run_split_synth_skills(self):
        from gui.training_tab.training_utils import build_split_synth_cmd
        cmd = build_split_synth_cmd(self.skill_split_pct.get(), mode='skills')
        self._run_subprocess("Split Synth Skills", cmd)

    def _run_yolo_train(self):
        from gui.training_tab.training_utils import build_yolo_cmd
        cfg = {
            'model_path': self.yolo_model_path.get(),
            'data_yaml': self.yolo_data.get(),
            'epochs': self.yolo_epochs.get(),
            'imgsz': self.yolo_imgsz.get(),
            'batch': self.yolo_batch.get(),
            'rect': self.yolo_rect.get(),
        }
        cmd = build_yolo_cmd(cfg)
        self._run_subprocess("YOLO Training", cmd)

    def _run_crnn_train(self):
        from gui.training_tab.training_utils import build_crnn_cmd
        cfg = {
            'epochs': self.crnn_epochs.get(),
            'batch': self.crnn_batch.get(),
            'lr': self.crnn_lr.get(),
            'sched': self.crnn_sched.get(),
        }
        cmd = build_crnn_cmd(cfg)
        self._run_subprocess("CRNN Training", cmd)

    def _clear_real_data_handler(self):
        from gui.training_tab.training_utils import clear_real_data
        clear_real_data()
        update_stats(self)

    def _clear_synth_numbers_handler(self):
        from gui.training_tab.training_utils import clear_synth_numbers
        clear_synth_numbers()
        update_stats(self)

    def _clear_synth_skills_handler(self):
        from gui.training_tab.training_utils import clear_synth_skills
        clear_synth_skills()
        update_stats(self)

    def _run_subprocess(self, name, cmd):
        import threading, subprocess, time, re

        def target():
            start_time = time.time()
            self._stop_requested = False

            # Header in the log
            self.log.after(0, lambda: self.log.insert('end', f"\n=== {name} ===\n"))
            self.log.after(0, self.log.see, 'end')
            # Reset progress/status/ETA
            self.pb.    after(0, lambda: self.pb.config(value=0))
            self.lbl_status.after(0, lambda: self.lbl_status.config(text=''))
            self.lbl_eta.  after(0, lambda: self.lbl_eta.config(text='ETA: N/A'))

            # Start subprocess
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            except Exception as e:
                self.log.after(0, lambda: self.log.insert('end', f"Failed to start {name}: {e}\n"))
                self.log.after(0, self._on_subprocess_complete)
                return

            self._current_process = proc
            prog_re = re.compile(r"PROGRESS\s+(\d+)\s*/\s*(\d+)")

            # Stream output
            for raw in proc.stdout:
                line = raw.rstrip('\n')
                if self._stop_requested:
                    break

                m = prog_re.search(line)
                if m:
                    done, total = map(int, m.groups())
                    pct = int(done / total * 100) if total else 0
                    elapsed = time.time() - start_time
                    eta = (elapsed * (total - done) / done) if done else 0
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))

                    # Update progress bar, status, ETA
                    self.pb.    after(0, lambda v=pct:    self.pb.config(value=v))
                    self.lbl_status.after(0, lambda d=done, t=total: self.lbl_status.config(text=f"{d}/{t}"))
                    self.lbl_eta.  after(0, lambda e=eta_str:      self.lbl_eta.config(text=f"ETA: {e}"))
                else:
                    # Regular log line
                    self.log.after(0, lambda l=line+'\n': self.log.insert('end', l))
                    self.log.after(0, self.log.see, 'end')

            proc.wait()
            exit_code = proc.returncode

            # Final status
            if self._stop_requested:
                self.log.after(0, lambda: self.log.insert('end', f"{name} aborted by user.\n"))
            else:
                self.pb.    after(0, lambda: self.pb.config(value=100))
                self.log.after(0, lambda: self.log.insert('end', f"=== {name} exited {exit_code} ===\n"))
                self.log.after(0, self.log.see, 'end')

            # Refresh stats panel when this step completes
            self.log.after(0, lambda: update_stats(self))
            # Re-enable buttons and clean up
            self.log.after(0, self._on_subprocess_complete)
            self._current_process = None

        threading.Thread(target=target, daemon=True).start()

    def _on_subprocess_complete(self):
        """
        Called on the main thread whenever a pipeline subprocess finishes,
        either via an individual step or as part of Run All.
        """
        # 1) Stop button off, Run All back on
        self.btn_stop.config(state='disabled')
        self.btn_run_all.config(state='normal')

        # 2) Clear/reset pipeline progress UI
        self.pb.config(value=0)
        self.lbl_status.config(text='')
        self.lbl_eta.config(text='ETA: N/A')

        # 3) Re‑enable each per‑step Run button
        for child in self.form_frame.winfo_children():
            # look for the 'Run' buttons in your steps list
            if isinstance(child, ttk.Button) and child.cget('text') == 'Run':
                child.config(state='normal')

        # 4) Refresh the stats table
        update_stats(self)

        if self._pending_steps and not self._stop_requested:
            next_title, next_func = self._pending_steps.pop(0)
            self.log.insert('end', f"\n=== Starting {next_title} ===\n")
            next_func()
        else:
            # all done (or aborted), reset Run All / Stop buttons
            self.btn_stop.config(state='disabled')
            self.btn_run_all.config(state='normal')

    # Mouse-wheel scroll handler
    def _on_mousewheel(self, event):
        self.form_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
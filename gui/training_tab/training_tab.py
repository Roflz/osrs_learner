import threading
from tkinter import ttk
from config import SKILLS
from gui.training_tab.training_ui import build_training_ui


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
        self._update_stats()
        self._pending_steps = []  # holds (title, func) tuples when using "Run All"

    def _build_ui(self):
        build_training_ui(self)

    def _update_stats(self):
        import os, json
        from collections import Counter

        st = self.stats_text
        st.config(state='normal', font=('Courier',10), bg='#1e1e1e', fg='#e0e0e0')
        st.delete('1.0', 'end')

        def color_for_ratio(r):
            if r < 0.5:
                r_, g = 200, int((r/0.5)*200)
            else:
                r_, g = int((1-(r-0.5)/0.5)*200), 200
            return f'#{r_:02x}{g:02x}00'

        def make_table(table_id, rows, headers, max_vals):
            widths = {h: len(h) for h in headers}
            for row in rows:
                for h in headers:
                    widths[h] = max(widths[h], len(str(row[h])))
            st.tag_configure('hdr', font=('Courier',10,'bold'), foreground='#ffffff')
            for h in headers:
                st.insert('end', h.ljust(widths[h]) + '  ', 'hdr')
            st.insert('end', '\n', 'hdr')
            for h in headers:
                st.insert('end', '-'*widths[h] + '  ', 'hdr')
            st.insert('end', '\n', 'hdr')
            for i, row in enumerate(rows, start=1):
                is_total = (str(row[headers[0]]) == 'Total')
                for j, h in enumerate(headers):
                    text = str(row[h]).ljust(widths[h])
                    if j == 0 or is_total:
                        st.insert('end', text + '  ')
                    else:
                        val   = row[h]
                        ratio = (val / max_vals[h]) if max_vals[h] else 0
                        fg    = color_for_ratio(ratio)
                        tag   = f'{table_id}_cell_{h}_{i}'
                        st.tag_configure(tag, foreground=fg)
                        st.insert('end', text + '  ', tag)
                st.insert('end', '\n')

        # ─── 1) SKILLS table ────────────────────────────────────────────────────
        rt, rv = Counter(), Counter()
        for split, ctr in [
            ('data/yolo/real/train/json', rt),
            ('data/yolo/real/val/json', rv),
        ]:
            if os.path.isdir(split):
                for fn in os.listdir(split):
                    if not fn.lower().endswith('.json'):
                        continue
                    data = json.load(open(os.path.join(split, fn)))
                    for skill_name in data.get('skills', []):
                        # map name→ID via your SKILLS list
                        cid = SKILLS.index(skill_name)
                        ctr[cid] += 1

        # 1b) count RealTotal via JSON skills
        real_json_dir = 'data/xp_crops_labeled'
        rt_json = Counter()
        if os.path.isdir(real_json_dir):
            for fn in os.listdir(real_json_dir):
                if fn.endswith('.json'):
                    data = json.load(open(os.path.join(real_json_dir, fn)))
                    for skill in data.get('skills', []):
                        rt_json[skill] += 1

        # 1c) count SynthTrain / SynthVal via synth_skill/train+val labels
        skl_tr, skl_val = Counter(), Counter()
        for path, ctr in [
            ('data/yolo/synth_skill/train/labels', skl_tr),
            ('data/yolo/synth_skill/val/labels', skl_val),
        ]:
            if os.path.isdir(path):
                for fn in os.listdir(path):
                    if fn.endswith('.txt'):
                        for ln in open(os.path.join(path, fn)):
                            parts = ln.split()
                            if parts:
                                ctr[int(parts[0])] += 1

        # 1d) count SynthTotal via ALL synth_skill labels
        synth_all_lbl = 'data/yolo/synth_skill/labels'
        skl_tot = Counter()
        if os.path.isdir(synth_all_lbl):
            for fn in os.listdir(synth_all_lbl):
                if fn.endswith('.txt'):
                    for ln in open(os.path.join(synth_all_lbl, fn)):
                        parts = ln.split()
                        if parts:
                            skl_tot[int(parts[0])] += 1

        # ─── 3) build skill_rows keyed by the *actual* class IDs ───────────────────
        skill_rows = []
        for cid, name in enumerate(SKILLS):
            if cid == 0:
                continue  # skip drop class
            rt_val = rt[cid]
            rv_val = rv[cid]
            st_val = skl_tr[cid]
            sv_val = skl_val[cid]
            skill_rows.append({
                'Skill': name,
                'RealTrain': rt_val,
                'RealVal': rv_val,
                'RealTotal': rt_val + rv_val,
                'SynthTrain': st_val,
                'SynthVal': sv_val,
                'SynthTotal': st_val + sv_val,
            })

        # ─── 4) append the Total row ───────────────────────────────────────────────
        total_skill = {
            'Skill': 'Total',
            'RealTrain': sum(r['RealTrain'] for r in skill_rows),
            'RealVal': sum(r['RealVal'] for r in skill_rows),
            'RealTotal': sum(r['RealTotal'] for r in skill_rows),
            'SynthTrain': sum(r['SynthTrain'] for r in skill_rows),
            'SynthVal': sum(r['SynthVal'] for r in skill_rows),
            'SynthTotal': sum(r['SynthTotal'] for r in skill_rows),
        }
        skill_rows.append(total_skill)

        # ─── 5) compute max_vals (excluding the Total row) ─────────────────────────
        data_skill = [r for r in skill_rows if r['Skill'] != 'Total']
        skill_max = {
            'RealTrain': max(r['RealTrain'] for r in data_skill) or 1,
            'RealVal': max(r['RealVal'] for r in data_skill) or 1,
            'RealTotal': max(r['RealTotal'] for r in data_skill) or 1,
            'SynthTrain': max(r['SynthTrain'] for r in data_skill) or 1,
            'SynthVal': max(r['SynthVal'] for r in data_skill) or 1,
            'SynthTotal': max(r['SynthTotal'] for r in data_skill) or 1,
        }

        # ─── 6) render the skills table ───────────────────────────────────────────
        st.insert('end', 'Skills\n', 'hdr')
        make_table(
            'skills',  # ← your table ID
            skill_rows,
            ['Skill', 'RealTrain', 'RealVal', 'RealTotal',
             'SynthTrain', 'SynthVal', 'SynthTotal'],
            skill_max
        )

        st.insert('end', '\n')

        # ─── 2) DIGIT INSTANCES table ──────────────────────────────────────────
        import os, csv, json
        from collections import Counter

        # 2a) count RealTrain / RealVal via split JSON xp_values
        real_train_json = 'data/yolo/real/train/json'
        real_val_json = 'data/yolo/real/val/json'
        digit_tr_real, digit_val_real = Counter(), Counter()
        for path, ctr in [(real_train_json, digit_tr_real), (real_val_json, digit_val_real)]:
            if os.path.isdir(path):
                for fn in os.listdir(path):
                    if fn.endswith('.json'):
                        data = json.load(open(os.path.join(path, fn)))
                        for xp in data.get('xp_values', []):
                            for ch in str(xp):
                                if ch.isdigit():
                                    ctr[int(ch)] += 1

        # 2b) count RealTotal via ALL JSON xp_values in xp_crops_labeled
        real_src_json = 'data/xp_crops_labeled'
        digit_total_real = Counter()
        if os.path.isdir(real_src_json):
            for fn in os.listdir(real_src_json):
                if fn.endswith('.json'):
                    data = json.load(open(os.path.join(real_src_json, fn)))
                    for xp in data.get('xp_values', []):
                        for ch in str(xp):
                            if ch.isdigit():
                                digit_total_real[int(ch)] += 1

        # 2c) count SynthTrain / SynthVal + SynthTotal via synth_map.csv
        csv_path = os.path.join('data', 'yolo', 'synth_numbers', 'synth_map.csv')
        train_dir = os.path.join('data', 'yolo', 'synth_numbers', 'train', 'images')
        val_dir = os.path.join('data', 'yolo', 'synth_numbers', 'val', 'images')

        train_screens = set(os.listdir(train_dir)) if os.path.isdir(train_dir) else set()
        val_screens = set(os.listdir(val_dir)) if os.path.isdir(val_dir) else set()

        digit_tr_syn = Counter()
        digit_val_syn = Counter()
        digit_total_syn = Counter()

        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as f:
                rdr = csv.reader(f)
                next(rdr, None)
                for crop, seq in rdr:
                    base = crop.split('_', 1)[0] + '.png'
                    # count for total
                    for ch in seq:
                        if ch.isdigit():
                            digit_total_syn[int(ch)] += 1
                    # count for train vs val
                    if base in train_screens:
                        for ch in seq:
                            if ch.isdigit():
                                digit_tr_syn[int(ch)] += 1
                    elif base in val_screens:
                        for ch in seq:
                            if ch.isdigit():
                                digit_val_syn[int(ch)] += 1

        # 2d) build digit rows
        digit_rows = []
        for d in range(10):
            digit_rows.append({
                'Digit': d,
                'RealTrain': digit_tr_real[d],
                'RealVal': digit_val_real[d],
                'RealTotal': digit_total_real[d],
                'SynthTrain': digit_tr_syn[d],
                'SynthVal': digit_val_syn[d],
                'SynthTotal': digit_total_syn[d],
            })

        # 2e) build digit rows
        digit_rows = []
        for d in range(10):
            digit_rows.append({
                'Digit': d,
                'RealTrain': digit_tr_real[d],
                'RealVal': digit_val_real[d],
                'RealTotal': digit_total_real[d],
                'SynthTrain': digit_tr_syn[d],
                'SynthVal': digit_val_syn[d],
                'SynthTotal': digit_total_syn[d],
            })

        # 2g) Totals row
        total_digits = {
            'Digit': 'Total',
            'RealTrain': sum(r['RealTrain'] for r in digit_rows),
            'RealVal': sum(r['RealVal'] for r in digit_rows),
            'RealTotal': sum(r['RealTotal'] for r in digit_rows),
            'SynthTrain': sum(r['SynthTrain'] for r in digit_rows),
            'SynthVal': sum(r['SynthVal'] for r in digit_rows),
            'SynthTotal': sum(r['SynthTotal'] for r in digit_rows),
        }
        digit_rows.append(total_digits)

        # 2h) compute maxima (exclude Total) and render
        data_digits = [r for r in digit_rows if r['Digit'] != 'Total']
        digit_max = {
            c: max(r[c] for r in data_digits) or 1
            for c in ['RealTrain', 'RealVal', 'RealTotal', 'SynthTrain', 'SynthVal', 'SynthTotal']
        }

        st.insert('end', 'Digit Instances\n', 'hdr')
        make_table(
            'digits',
            digit_rows,
            ['Digit', 'RealTrain', 'RealVal', 'RealTotal', 'SynthTrain', 'SynthVal', 'SynthTotal'],
            digit_max
        )

        st.config(state='disabled')

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
        self._update_stats()

    def _clear_synth_numbers_handler(self):
        from gui.training_tab.training_utils import clear_synth_numbers
        clear_synth_numbers()
        self._update_stats()

    def _clear_synth_skills_handler(self):
        from gui.training_tab.training_utils import clear_synth_skills
        clear_synth_skills()
        self._update_stats()

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
            self.log.after(0, self._update_stats)
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
        self._update_stats()

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

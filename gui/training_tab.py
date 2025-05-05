import os
import sys
import threading
import subprocess
import time
import tkinter as tk
from collections import Counter
from tkinter import ttk, messagebox
import re
import os, csv, json
from collections import Counter

import yaml

from config import SKILLS


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

        # Build the UI components (form, stats panel, controls, log)
        self._build_ui()
        # Load initial data statistics into the stats panel
        self._update_stats()
        self._pending_steps = []  # holds (title, func) tuples when using "Run All"

    def _build_ui(self):
        # ─── Horizontal split: left=pipeline, right=(stats above log) ────────────
        paned_h = tk.PanedWindow(self, orient='horizontal')  # use tk.PanedWindow here
        paned_h.pack(fill='both', expand=True)

        # ── Left pane: pipeline steps form (initially fitted) ───────────────────
        pipe_frame = ttk.Frame(paned_h)

        pipe_frame.columnconfigure(0, weight=1)
        pipe_frame.rowconfigure(0, weight=1)

        # scrollable canvas for steps
        self.form_canvas = tk.Canvas(pipe_frame)
        scroll_form = ttk.Scrollbar(pipe_frame, orient='vertical', command=self.form_canvas.yview)
        self.form_canvas.configure(yscrollcommand=scroll_form.set)
        self.form_canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        scroll_form.grid(row=0, column=1, sticky='ns', padx=(0, 5), pady=5)

        self.form_frame = ttk.Frame(self.form_canvas)
        self.form_canvas.create_window((0, 0), window=self.form_frame, anchor='nw')

        self.form_frame.bind(
            '<Configure>',
            lambda e: self.form_canvas.configure(scrollregion=self.form_canvas.bbox('all'))
        )
        self.form_canvas.bind('<Enter>', lambda e: self.form_canvas.bind_all('<MouseWheel>', self._on_mousewheel))
        self.form_canvas.bind('<Leave>', lambda e: self.form_canvas.unbind_all('<MouseWheel>'))

        # ── Define pipeline steps & their params ─────────────────────────────────
        self.steps = [
            ('Split Real Data', [
                ('Source Dir', 'src_real',             'data/xp_crops_labeled'),
                ('Train %',    'pct_real',             80),
            ], self._run_split_real),

            ('Synthetic Numbers', [
                ('Num Images',      'syn_num',         8000),
                ('Neg Ratio',       'syn_neg',         0.2),
                ('Use Inverse Freq','syn_use_weight',  True),
            ], self._run_synth_numbers),

            ('Synthetic Skills', [
                ('Num Images',      'skill_num',       8000),
                ('Max Icons/Image', 'skill_max',       5),
                ('Use Inverse Freq','skill_use_weight',True),
                ('Max Weight',      'skill_max_weight',''),
            ], self._run_synth_skills),

            ('Split Synth Numbers', [
                ('Train %', 'synth_split_pct', 80)
            ], self._run_split_synth),

            ('Split Synth Skills', [
                ('Train %', 'skill_split_pct', 80)
            ], self._run_split_synth_skills),

            ('YOLO Training', [
                ('Model Path',   'yolo_model_path', 'models/best.pt'),
                ('Data YAML',    'yolo_data',       'data.yaml'),
                ('Epochs',       'yolo_epochs',     40),
                ('Img Size',     'yolo_imgsz',      640),
                ('Batch Size',   'yolo_batch',      16),
                ('Rect Training','yolo_rect',       True),
            ], self._run_yolo_train),

            ('CRNN Training', [
                ('Epochs',        'crnn_epochs', 20),
                ('Batch Size',    'crnn_batch',  32),
                ('Learning Rate', 'crnn_lr',     1e-3),
                ('Scheduler',     'crnn_sched',  'onecycle'),
            ], self._run_crnn_train),
        ]
        # ────────────────────────────────────────────────────────────────────────────

        # ────────────────────────────────────────────────────────────────
        # Build each pipeline‐step UI inside form_frame, in 3 sections:
        #   1) Real Data     2) Synthetic Data     3) Training
        # ────────────────────────────────────────────────────────────────
        r = 0
        # 1) Real Data
        ttk.Label(self.form_frame, text='1) Real Data',
                  font=('Helvetica', 14, 'bold'), foreground='#4a90e2') \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 5))
        r += 1

        real_steps = [
            ('Split Real Data', [
                ('Source Dir', 'src_real', 'data/xp_crops_labeled'),
                ('Train %', 'pct_real', 80),
            ], self._run_split_real)
        ]
        for title, fields, func in real_steps:
            ttk.Label(self.form_frame, text=title, font=('Helvetica', 12, 'bold')) \
                .grid(row=r, column=0, columnspan=2, sticky='w', padx=20, pady=(5, 2))
            r += 1
            for label_text, attr, default in fields:
                var = getattr(self, attr, tk.StringVar(value=default))
                setattr(self, attr, var)
                ttk.Label(self.form_frame, text=label_text + ':') \
                    .grid(row=r, column=0, sticky='e', padx=30, pady=2)
                ttk.Entry(self.form_frame, textvariable=var, width=12) \
                    .grid(row=r, column=1, sticky='w')
                r += 1
            ttk.Button(self.form_frame, text=f'Run {title}', command=func) \
                .grid(row=r, column=0, columnspan=2, pady=(0, 5))
            r += 1

        # Real Data Utility: Clear Real Train/Val
        ttk.Label(self.form_frame, text='Real Data Utilities:',
                  font=('Helvetica', 12, 'bold')) \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=20, pady=(10, 2))
        r += 1
        ttk.Button(self.form_frame, text='Clear Real Train/Val',
                   command=self._clear_real_data) \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=30, pady=2)
        r += 1

        # 2) Synthetic Data
        ttk.Label(self.form_frame, text='2) Synthetic Data',
                  font=('Helvetica', 14, 'bold'), foreground='#50e3c2') \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=5, pady=(15, 5))
        r += 1
        synth_steps = [
            (
                'Make Synth Numbers',
                [
                    ('Num Images', 'syn_num', 8000),
                    ('Neg Ratio', 'syn_neg', 0.2),
                    ('Min Seq', 'syn_min_seq', 1),
                    ('Max Seq', 'syn_max_seq', 5),
                    ('Min Digits', 'syn_min_digits', 1),
                    ('Max Digits', 'syn_max_digits', 5),
                    ('Use Weights', 'syn_use_weights', True),
                    ('Cap Weight', 'syn_max_weight', ''),  # blank → no cap
                ],
                self._run_synth_numbers
            ),

            ('Make Synth Skills', [
                ('Num Images', 'skill_num', 8000),
                ('Max Icons/Image', 'skill_max_icons', 5),
                ('Use Weights', 'skill_use_weights', True),
                ('Cap Weight', 'skill_max_weight', ''),  # empty = no cap
            ], self._run_synth_skills),

            ('Split Synth Numbers', [
                ('Train %', 'synth_split_pct', 80)
            ], self._run_split_synth),

            ('Split Synth Skills', [
                ('Train %', 'skill_split_pct', 80)
            ], self._run_split_synth_skills),
        ]
        for title, fields, func in synth_steps:
            ttk.Label(self.form_frame, text=title, font=('Helvetica', 12, 'bold')) \
                .grid(row=r, column=0, columnspan=2, sticky='w', padx=20, pady=(5, 2))
            r += 1
            for label_text, attr, default in fields:
                if isinstance(default, bool):
                    var = getattr(self, attr, tk.BooleanVar(value=default))
                    setattr(self, attr, var)
                    ttk.Checkbutton(
                        self.form_frame,
                        text=label_text,
                        variable=var
                    ).grid(row=r, column=0, columnspan=2, sticky='w', padx=30, pady=2)
                else:
                    var = getattr(self, attr, tk.StringVar(value=str(default)))
                    setattr(self, attr, var)
                    ttk.Label(self.form_frame, text=label_text + ':') \
                        .grid(row=r, column=0, sticky='e', padx=30, pady=2)
                    ttk.Entry(self.form_frame, textvariable=var, width=12) \
                        .grid(row=r, column=1, sticky='w')
                r += 1
            ttk.Button(self.form_frame, text=f'Run {title}', command=func) \
                .grid(row=r, column=0, columnspan=2, pady=(0, 5))
            r += 1

        # ─── Synthetic Utilities: Clear Data ─────────────────────────────────
        ttk.Label(self.form_frame, text='Synthetic Utilities:',
                  font=('Helvetica',12,'bold')) \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=20, pady=(10,2))
        r += 1

        ttk.Button(self.form_frame, text='Clear Synth Numbers',
                   command=self._clear_synth_numbers) \
            .grid(row=r, column=0, sticky='w', padx=30, pady=2)
        ttk.Button(self.form_frame, text='Clear Synth Skills',
                   command=self._clear_synth_skills) \
            .grid(row=r, column=1, sticky='w', padx=30, pady=2)
        r += 1


        # 3) Training
        ttk.Label(self.form_frame, text='3) Training',
                  font=('Helvetica', 14, 'bold'), foreground='#f5a623') \
            .grid(row=r, column=0, columnspan=2, sticky='w', padx=5, pady=(15, 5))
        r += 1
        train_steps = [
            ('YOLO Training', [
                ('Model Path', 'yolo_model_path', 'models/best.pt'),
                ('Data YAML', 'yolo_data', 'data.yaml'),
                ('Epochs', 'yolo_epochs', 40),
                ('Img Size', 'yolo_imgsz', 640),
                ('Batch Size', 'yolo_batch', 16),
                ('Rect Training', 'yolo_rect', True),
            ], self._run_yolo_train),

            ('CRNN Training', [
                ('Epochs', 'crnn_epochs', 20),
                ('Batch Size', 'crnn_batch', 32),
                ('Learning Rate', 'crnn_lr', 1e-3),
                ('Scheduler', 'crnn_sched', 'onecycle'),
            ], self._run_crnn_train),
        ]
        for title, fields, func in train_steps:
            ttk.Label(self.form_frame, text=title, font=('Helvetica', 12, 'bold')) \
                .grid(row=r, column=0, columnspan=2, sticky='w', padx=20, pady=(5, 2))
            r += 1
            for label_text, attr, default in fields:
                # if default is a bool, render a checkbox
                if isinstance(default, bool):
                    var = getattr(self, attr, tk.BooleanVar(value=default))
                    setattr(self, attr, var)
                    ttk.Checkbutton(
                        self.form_frame,
                        text=label_text,
                        variable=var
                    ).grid(row=r, column=0, columnspan=2, sticky='w', padx=30, pady=2)
                else:
                    var = getattr(self, attr, tk.StringVar(value=str(default)))
                    setattr(self, attr, var)
                    ttk.Label(self.form_frame, text=label_text + ':') \
                        .grid(row=r, column=0, sticky='e', padx=30, pady=2)
                    ttk.Entry(self.form_frame, textvariable=var, width=12) \
                        .grid(row=r, column=1, sticky='w')
                r += 1
            ttk.Button(self.form_frame, text=f'Run {title}', command=func) \
                .grid(row=r, column=0, columnspan=2, pady=(0, 5))
            r += 1

        # make remaining space stretch
        self.form_frame.rowconfigure(r, weight=1)

        # After all widgets are added to self.form_frame
        self.update_idletasks()
        pane_width = self.form_frame.winfo_reqwidth() + scroll_form.winfo_reqwidth() + 20

        # Correctly add pane with minsize and no stretch initially
        paned_h.add(pipe_frame, minsize=pane_width, stretch='never')

        # ─── Right pane: vertical split of stats (top) and controls+log (bottom) ─
        paned_v = tk.PanedWindow(paned_h, orient='vertical')
        paned_h.add(paned_v, stretch='always')  # allow the right pane to stretch horizontally

        # ─── Stats panel in the top‐right ────────────────────────────────────────
        stats_frame = ttk.LabelFrame(paned_v, text='Stats Summary')
        paned_v.add(stats_frame, stretch='always')  # expand vertically

        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(
            stats_frame,
            bg='#1e1e1e', fg='#e0e0e0',
            wrap='none', borderwidth=0,
            font=('Courier', 10)
        )
        stats_scroll = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)

        self.stats_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        stats_scroll.grid(row=0, column=1, sticky='ns', padx=(0, 5), pady=5)

        refresh_btn = ttk.Button(
            stats_frame,
            text='Refresh',
            command=self._update_stats
        )
        refresh_btn.grid(row=1, column=0, columnspan=2, sticky='e', padx=5, pady=(0, 5))

        # ─── Bottom Frame (Pipeline Log and Controls) ──────────────────────────────
        bottom_frame = ttk.Frame(paned_v)

        # Explicitly set initial height to ~25% (adjust as needed, e.g., 200–250 pixels)
        paned_v.add(bottom_frame, stretch='never', height=200)

        bottom_frame.columnconfigure(0, weight=1)

        # Run/Stop buttons
        ctrl = ttk.Frame(bottom_frame)
        ctrl.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        ctrl.columnconfigure((0, 1), weight=1)
        self.btn_run_all = ttk.Button(ctrl, text='Run All', command=self._on_run_all)
        self.btn_stop = ttk.Button(ctrl, text='Stop', command=self._on_stop_pipeline, state='disabled')
        self.btn_run_all.grid(row=0, column=0, padx=5)
        self.btn_stop.grid(row=0, column=1, padx=5)

        # Progress / status / ETA
        self.pb = ttk.Progressbar(bottom_frame, mode='determinate')
        self.lbl_status = ttk.Label(bottom_frame, text='')
        self.lbl_eta = ttk.Label(bottom_frame, text='ETA: N/A')
        self.pb.grid(row=1, column=0, sticky='ew', padx=5)
        self.lbl_status.grid(row=1, column=1, sticky='w')
        self.lbl_eta.grid(row=2, column=0, columnspan=2, sticky='w', padx=5)

        # Pipeline Log
        ttk.Label(bottom_frame, text='Pipeline Log:', font=('Helvetica', 12, 'bold')) \
            .grid(row=3, column=0, columnspan=2, sticky='w', padx=5)
        self.log = tk.Text(bottom_frame, bg='#1e1e1e', fg='#e0e0e0', wrap='none')
        scroll_log = ttk.Scrollbar(bottom_frame, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=scroll_log.set)
        self.log.grid(row=4, column=0, sticky='nsew', padx=5, pady=5)
        scroll_log.grid(row=4, column=1, sticky='ns', padx=(0, 5), pady=5)

        bottom_frame.rowconfigure(4, weight=1)

    def _update_stats(self):
        import os, csv, json
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

    # ---- Individual step subprocess wrappers ----
    def _run_split_real(self):
        self._run_subprocess('Split Real', [sys.executable,
                                           'training/split_real.py',
                                           '--src', self.src_real.get(),
                                           '--pct', str(self.pct_real.get())])

    def _run_synth_numbers(self):
        # base command
        cmd = [
            'python', 'training/make_synth_numbers.py',
            '--num_images',    str(self.syn_num.get()),
            '--neg_ratio',     str(self.syn_neg.get()),
            '--min_seq',       str(self.syn_min_seq.get()),
            '--max_seq',       str(self.syn_max_seq.get()),
            '--min_digits',    str(self.syn_min_digits.get()),
            '--max_digits',    str(self.syn_max_digits.get()),
        ]

        # cap weight?
        max_w = self.syn_max_weight.get().strip()
        if max_w:
            cmd += ['--max_weight', max_w]

        # uniform vs inverse-frequency?
        if not self.syn_use_weights.get():
            cmd.append('--no_weights')

        self._run_subprocess('Make Synth Numbers', cmd)

    def _run_synth_skills(self):
        # build base command
        cmd = [
            'python', 'training/make_synth_skill.py',
            '--num_images', str(self.skill_num.get()),
            '--max_icons', str(self.skill_max_icons.get()),
        ]

        # cap weight?
        max_w = self.skill_max_weight.get().strip()
        if max_w:
            cmd += ['--max_weight', max_w]

        # uniform vs weighted?
        if not self.skill_use_weights.get():
            cmd.append('--no_weights')

        # specify output dirs if needed (optional)
        # cmd += ['--out_img_dir', DEFAULT_OUT_IMG_DIR, '--out_ann_dir', DEFAULT_OUT_ANN_DIR]

        # run it
        self._run_subprocess('Make Synth Skills', cmd)

    def _run_split_synth(self):
        self._run_subprocess('Split Synth Numbers',
                             [sys.executable, 'training/split_synth.py',
                              '--pct', str(self.synth_split_pct.get())])

    def _run_split_synth_skills(self):
        self._run_subprocess('Split Synth Skills',
                             [sys.executable, 'training/split_synth_skill.py',
                              '--pct', str(self.skill_split_pct.get())])

    def _run_yolo_train(self):
        cmd = ['yolo','detect','train',
               f'model={self.yolo_model_path.get()}',
               f'data={self.yolo_data.get()}',
               f'epochs={self.yolo_epochs.get()}',
               f'imgsz={self.yolo_imgsz.get()}',
               f'batch={self.yolo_batch.get()}',
               f'rect={self.yolo_rect.get()}']
        self._run_subprocess('YOLO Train', cmd)

    def _run_crnn_train(self):
        cmd = ['python', 'training/minimal_crnn_ctc.py',
               '--epochs', str(self.crnn_epochs.get()),
               '--batch', str(self.crnn_batch.get()),
               '--lr', str(self.crnn_lr.get()),
               '--scheduler', self.crnn_sched.get()]
        self._run_subprocess('CRNN Train', cmd)

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

    def _clear_synth_numbers(self):
        import os, shutil

        base = os.path.join('data','yolo','synth_numbers')
        # remove train/val folders
        for sub in ('train','val'):
            for kind in ('images','labels'):
                path = os.path.join(base, sub, kind)
                if os.path.isdir(path):
                    shutil.rmtree(path)
        # now delete the CSV map:
        csv_path = os.path.join(base, 'synth_map.csv')
        if os.path.isfile(csv_path):
            os.remove(csv_path)

        # recreate empty structure:
        for sub in ('train','val'):
            for kind in ('images','labels'):
                os.makedirs(os.path.join(base, sub, kind), exist_ok=True)

        # log & refresh
        self.log.insert('end', '🚮 Cleared Synth Numbers data & map\n')
        self._update_stats()

    def _clear_synth_skills(self):
        """
        Delete and recreate all synthetic‐skill directories:
        data/yolo/synth_skill/{images,labels,train,val}
        """
        import shutil, os

        base = os.path.join('data','yolo','synth_skill')
        for sub in ('images','labels','train','val'):
            path = os.path.join(base, sub)
            if os.path.isdir(path):
                shutil.rmtree(path)
        os.makedirs(os.path.join(base,'images'), exist_ok=True)
        os.makedirs(os.path.join(base,'labels'), exist_ok=True)
        for split in ('train','val'):
            os.makedirs(os.path.join(base, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base, split, 'labels'), exist_ok=True)

        self.log.insert('end', '🚮 Cleared synthetic‐skills data\n')
        self._update_stats()

    def _clear_real_data(self):
        """
        Delete and recreate data/yolo/real/train and data/yolo/real/val subdirs.
        """
        import shutil, os

        bases = [
            os.path.join('data','yolo','real','train'),
            os.path.join('data','yolo','real','val')
        ]
        for base in bases:
            # remove any existing subdirs and cache
            cache = os.path.join(base, 'labels.cache')
            if os.path.isfile(cache):
                os.remove(cache)
            for sub in ('images','json','labels'):
                path = os.path.join(base, sub)
                if os.path.isdir(path):
                    shutil.rmtree(path)

            # recreate fresh
            for sub in ('images','json','labels'):
                os.makedirs(os.path.join(base, sub), exist_ok=True)

        # log and refresh
        self.log.insert('end', '🚮 Cleared Real train/val data\n')
        self._update_stats()


    # Mouse-wheel scroll handler
    def _on_mousewheel(self, event):
        self.form_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')

# Usage:
# root = tk.Tk()
# tab = UnifiedTrainingTab(root)
# tab.pack(fill='both', expand=True)
# root.mainloop()

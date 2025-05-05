# training_ui.py
import tkinter as tk
from tkinter import ttk

# === Top-level UI Layout Entrypoint ===
def build_training_ui(self):
    """
    Create the main training UI layout: a horizontal paned window splitting
    pipeline controls (left) and stats/log (right).
    """
    paned_h = tk.PanedWindow(self, orient='horizontal')
    paned_h.pack(fill='both', expand=True)

    # Left: pipeline steps
    _build_pipeline_panel(self, paned_h)

    # Right: stats summary above log
    _build_stats_and_log_panels(self, paned_h)

# === Pipeline Panel ===
def _build_pipeline_panel(self, parent_paned):
    pipe_frame = ttk.Frame(parent_paned)
    pipe_frame.columnconfigure(0, weight=1)
    pipe_frame.rowconfigure(0, weight=1)
    parent_paned.add(pipe_frame, stretch='never')

    # scrollable canvas for steps
    self.form_canvas = tk.Canvas(pipe_frame)
    scroll_form = ttk.Scrollbar(pipe_frame, orient='vertical', command=self.form_canvas.yview)
    self.form_canvas.configure(yscrollcommand=scroll_form.set)
    self.form_canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
    scroll_form.grid(row=0, column=1, sticky='ns', padx=(0,5), pady=5)

    self.form_frame = ttk.Frame(self.form_canvas)
    self.form_canvas.create_window((0,0), window=self.form_frame, anchor='nw')
    self.form_frame.bind(
        '<Configure>',
        lambda e: self.form_canvas.configure(scrollregion=self.form_canvas.bbox('all'))
    )
    self.form_canvas.bind('<Enter>', lambda e: self.form_canvas.bind_all('<MouseWheel>', self._on_mousewheel))
    self.form_canvas.bind('<Leave>', lambda e: self.form_canvas.unbind_all('<MouseWheel>'))

    _populate_pipeline_steps(self)
    self.update_idletasks()
    # ensure min size based on content
    pane_width = self.form_frame.winfo_reqwidth() + scroll_form.winfo_reqwidth() + 20
    parent_paned.paneconfig(pipe_frame, minsize=pane_width)

# === Pipeline Steps Population ===
def _populate_pipeline_steps(self):
    # original step definitions and UI grid placement
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
               command=self._clear_real_data_handler) \
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
               command=self._clear_synth_numbers_handler) \
        .grid(row=r, column=0, sticky='w', padx=30, pady=2)
    ttk.Button(self.form_frame, text='Clear Synth Skills',
               command=self._clear_synth_skills_handler) \
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

# === Stats and Log Panels ===
def _build_stats_and_log_panels(self, parent_paned):
    paned_v = tk.PanedWindow(parent_paned, orient='vertical')
    parent_paned.add(paned_v, stretch='always')

    # Stats panel (expandable)
    self.stats_frame = ttk.LabelFrame(paned_v, text='Stats Summary')
    paned_v.add(self.stats_frame, stretch='always')
    _populate_stats_panel(self)

    # Log panel (fixed height)
    self.bottom_frame = ttk.Frame(paned_v)
    paned_v.add(self.bottom_frame, height=200, stretch='never')
    _populate_log_panel(self)

# === Populate Stats Panel ===
def _populate_stats_panel(self):
    self.stats_frame.columnconfigure(0, weight=1)
    self.stats_frame.rowconfigure(0, weight=1)
    self.stats_text = tk.Text(
        self.stats_frame, bg='#1e1e1e', fg='#e0e0e0', wrap='none', borderwidth=0, font=('Courier',10)
    )
    stats_scroll = ttk.Scrollbar(self.stats_frame, orient='vertical', command=self.stats_text.yview)
    self.stats_text.configure(yscrollcommand=stats_scroll.set)
    self.stats_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
    stats_scroll.grid(row=0, column=1, sticky='ns', padx=(0,5), pady=5)
    refresh_btn = ttk.Button(self.stats_frame, text='Refresh', command=self._update_stats)
    refresh_btn.grid(row=1, column=0, columnspan=2, sticky='e', padx=5, pady=(0,5))

# === Populate Log Panel ===
def _populate_log_panel(self):
    bf = self.bottom_frame
    bf.columnconfigure(0, weight=1)
    # controls row
    ctrl = ttk.Frame(bf)
    ctrl.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
    ctrl.columnconfigure((0,1), weight=1)
    self.btn_run_all = ttk.Button(ctrl, text='Run All', command=self._on_run_all)
    self.btn_stop    = ttk.Button(ctrl, text='Stop',   command=self._on_stop_pipeline, state='disabled')
    self.btn_run_all.grid(row=0, column=0, padx=5)
    self.btn_stop   .grid(row=0, column=1, padx=5)

    # progress/status row
    self.pb = ttk.Progressbar(bf, mode='determinate')
    self.lbl_status = ttk.Label(bf, text='')
    self.lbl_eta    = ttk.Label(bf, text='ETA: N/A')
    self.pb.grid(row=1, column=0, sticky='ew', padx=5)
    self.lbl_status.grid(row=1, column=1, sticky='w')
    self.lbl_eta.grid(row=2, column=0, columnspan=2, sticky='w', padx=5)

    # log text
    ttk.Label(bf, text='Pipeline Log:', font=('Helvetica',12,'bold')) \
        .grid(row=3, column=0, columnspan=2, sticky='w', padx=5)
    self.log = tk.Text(bf, bg='#1e1e1e', fg='#e0e0e0', wrap='none')
    scroll_log = ttk.Scrollbar(bf, orient='vertical', command=self.log.yview)
    self.log.configure(yscrollcommand=scroll_log.set)
    self.log.grid(row=4, column=0, sticky='nsew', padx=5, pady=5)
    scroll_log.grid(row=4, column=1, sticky='ns', padx=(0,5), pady=5)
    bf.rowconfigure(4, weight=1)

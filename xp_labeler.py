import json
import os
import random
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm

# === Paths & Constants ===
CROP_DIR = "data/xp_crops"
LABELED_DIR = "data/xp_crops_labeled"
SKIP_DIR = "data/xp_crops_skipped"
YOLO_MODEL_PATH = os.path.join("models", "best.pt")
CRNN_MODEL_PATH = os.path.join("models", "crnn_best.pt")  # your standalone CRNN

SKILLS = [
    "drop", "agility", "attack", "construction",
    "cooking", "crafting", "defence", "farming",
    "firemaking", "fishing", "fletching", "herblore",
    "hitpoints", "hunter", "magic", "mining",
    "prayer", "ranged", "runecrafting", "smithing",
    "slayer", "strength", "thieving", "woodcutting"
]


class XPLabeler:
    def __init__(self, root):
        self.root = root
        root.title("XP Labeler & Trainer")

        # --- Set up notebook for two tabs: Labeling & Training ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Labeling tab
        self.label_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.label_tab, text='Labeling')
        # Training tab
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text='Training')

        # === Build Labeling UI inside label_tab ===
        self.build_labeling_ui(self.label_tab)

        # === Build Training UI inside train_tab ===
        self.build_training_ui(self.train_tab)

        # finally load first image & stats
        if hasattr(self, 'files') and self.files:
            self.load_image()
            self.update_stats()

    def build_labeling_ui(self, parent):
        # --- Auto‑fill threshold defaults & misc state flags ---
        self.drop_threshold = 80.0
        self.xp_threshold = 80.0
        self.skill_threshold = 80.0
        self.box_opacity = 255
        self.random_order = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.fill_with_predicted = tk.BooleanVar(value=True)

        # === Dark Theme Colors ===
        self.bg_color = '#1e1e1e';
        self.fg_color = '#ffffff'
        self.button_bg = '#333333';
        self.button_fg = '#ffffff'
        self.entry_bg = '#2b2b2b';
        self.entry_fg = '#ffffff'
        self.highlight_color = '#5e9cff';
        self.shortcut_color = '#CCCC66'

        parent.option_add('*Font', ('Helvetica', 12))
        for opt in ['Background', 'Frame.background', 'Label.background']:
            parent.option_add(f'*{opt}', self.bg_color)
        for opt in ['Foreground', 'Label.foreground', 'Entry.foreground']:
            parent.option_add(f'*{opt}', self.fg_color)
        parent.option_add('*Entry.background', self.entry_bg)
        parent.option_add('*Spinbox.background', self.entry_bg)
        parent.option_add('*Button.background', self.button_bg)
        parent.option_add('*Button.foreground', self.button_fg)
        style = ttk.Style();
        style.theme_use('clam')
        style.configure('TCombobox',
                        fieldbackground=self.entry_bg,
                        background=self.entry_bg,
                        foreground=self.entry_fg)
        style.map('TCombobox',
                  fieldbackground=[('readonly', self.entry_bg)],
                  foreground=[('readonly', self.entry_fg)])

        # === Directories ===
        os.makedirs(CROP_DIR, exist_ok=True)
        os.makedirs(LABELED_DIR, exist_ok=True)
        os.makedirs(SKIP_DIR, exist_ok=True)

        # === Models ===
        self.device = torch.device('cpu')
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.ctc_model = torch.load(CRNN_MODEL_PATH, map_location=self.device)
        self.ctc_model.eval()
        self.ctc_preprocess = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # === State ===
        self.files = sorted(f for f in os.listdir(CROP_DIR) if f.lower().endswith('.png'))
        self.current_index = 0
        self.current_boxes = []
        self.skill_boxes = []
        self.xp_preds = []
        self.xp_confs = []
        self.xp_entries = []
        self.skill_selectors = []
        self.skill_boxes_raw = []
        self.skill_boxes_fallback = []
        self.use_raw_skill = []
        self.action_history = []
        self.defaults = {}
        self.last_value = ""
        self.last_count = 0
        self.last_nonzero_skills = []
        self.auto_skipped = 0
        self.labels = {}

        # === Layout: Canvas ===
        parent.rowconfigure(0, weight=1);
        parent.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(parent, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<Button-1>', self.on_click)
        for key, func in [('<Left>', self.move_skill_left),
                          ('<Right>', self.move_skill_right),
                          ('<Up>', self.move_skill_up),
                          ('<Down>', self.move_skill_down)]:
            self.canvas.bind(key, func)
        self.root.bind('<Return>', lambda e: self.save_label())
        self.root.bind('<Control-s>', lambda e: self.skip_image())
        self.root.bind('<Control-u>', lambda e: self.undo_label())

        # === Controls ===
        ctrl = tk.Frame(parent)
        ctrl.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        for c in range(3):
            ctrl.columnconfigure(c, weight=1)

        # Row 0: Remaining & Auto‑Prune
        self.count_label = tk.Label(ctrl, fg=self.highlight_color, anchor='w')
        self.count_label.grid(row=0, column=0, sticky='w')
        tk.Button(ctrl, text="Auto Prune Images", command=self.auto_prune_images) \
            .grid(row=0, column=2, sticky='e')

        # Row 1: Stats
        self.stats_label = tk.Label(ctrl, fg=self.highlight_color, anchor='w')
        self.stats_label.grid(row=1, column=0, sticky='w')

        # Row 2: XP values / Drop count / Skills
        frm = tk.Frame(ctrl)
        frm.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(5, 10))
        tk.Label(frm, text='XP values:').grid(row=0, column=0, sticky='e')
        self.xp_frame = tk.Frame(frm)
        self.xp_frame.grid(row=0, column=1, padx=(2, 10), sticky='w')
        tk.Label(frm, text='Drop count:').grid(row=0, column=2, sticky='e')
        self.count_spin = tk.Spinbox(frm, from_=0, to=20, width=5,
                                     command=self._on_count_changed)
        self.count_spin.grid(row=0, column=3, sticky='w', padx=(2, 10))
        tk.Label(frm, text='Skills:').grid(row=1, column=0, sticky='e')
        self.skill_frame = tk.Frame(frm)
        self.skill_frame.grid(row=1, column=1, columnspan=3, sticky='w')

        # Row 3: Fill‑mode toggle
        tk.Checkbutton(ctrl,
                       text="Fill with Predicted Values",
                       variable=self.fill_with_predicted) \
            .grid(row=3, column=1, pady=(5, 0))

        # Row 4: Prediction & confidence UI
        self.pred_frame = tk.Frame(ctrl)
        self.pred_frame.grid(row=4, column=1, pady=(5, 5))

        # Row 5: Navigation buttons
        nav = tk.Frame(ctrl)
        nav.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(0, 5))
        tk.Button(nav, text='Prev', command=self.prev_image).pack(side='left')
        tk.Button(nav, text='Next', command=self.next_image).pack(side='left', padx=5)
        tk.Button(nav, text='Skip', command=self.skip_image).pack(side='left', padx=5)
        self.save_btn = tk.Button(nav, text='Save Label', command=self.save_label)
        self.save_btn.pack(side='left', expand=True)
        tk.Button(nav, text='Undo', command=self.undo_label).pack(side='right')

        # Row 6: status messages
        self.status_label = tk.Label(ctrl, fg='green', anchor='w')
        self.status_label.grid(row=6, column=0, columnspan=3, sticky='w', pady=(0, 5))

        # Row 7: Show boxes & Random Order
        tk.Checkbutton(ctrl, text="Show YOLO Boxes",
                       variable=self.show_boxes,
                       command=self._draw_image) \
            .grid(row=7, column=0, sticky='w')
        tk.Checkbutton(ctrl, text="Random Order",
                       variable=self.random_order) \
            .grid(row=7, column=1)

        # Row 8: Threshold sliders
        thresh = tk.Frame(ctrl)
        thresh.grid(row=8, column=0, columnspan=3, sticky='ew', pady=(5, 5))
        tk.Label(thresh, text="Drop ≥").pack(side='left')
        self.drop_thresh_slider = tk.Scale(
            thresh, from_=0, to=100, orient='horizontal', length=250,
            command=self._on_drop_thresh_changed
        )
        self.drop_thresh_slider.set(self.drop_threshold)
        self.drop_thresh_slider.pack(side='left', padx=(5, 10))
        tk.Label(thresh, text="XP ≥").pack(side='left')
        self.xp_thresh_slider = tk.Scale(
            thresh, from_=0, to=100, orient='horizontal', length=250,
            command=self._on_xp_thresh_changed
        )
        self.xp_thresh_slider.set(self.xp_threshold)
        self.xp_thresh_slider.pack(side='left', padx=(5, 10))
        tk.Label(thresh, text="Skill ≥").pack(side='left')
        self.skill_thresh_slider = tk.Scale(
            thresh, from_=0, to=100, orient='horizontal', length=250,
            command=self._on_skill_thresh_changed
        )
        self.skill_thresh_slider.set(self.skill_threshold)
        self.skill_thresh_slider.pack(side='left')

        # Row 9: Shortcut hint
        tk.Label(ctrl,
                 text="Shortcuts: ←→↑↓ Move | Enter: Save | Ctrl+S: Skip | Ctrl+U: Undo",
                 font=('Helvetica', 11, 'bold'),
                 fg=self.shortcut_color,
                 anchor='center') \
            .grid(row=9, column=0, columnspan=3, sticky='ew', pady=(5, 0))

        # Bind fill‑mode changes
        self.fill_with_predicted.trace_add('write', self._apply_fill_mode)

        # finally: initial load
        if self.files:
            self.load_image()
            self.update_stats()

    def build_training_ui(self, parent):
        # Layout parameters & labels
        for c in range(2):
            parent.columnconfigure(c, weight=1)
        tk.Label(parent, text='Training with Labeled Data', font=('Helvetica', 14, 'bold')).grid(row=0, column=0,
                                                                                                 columnspan=2,
                                                                                                 pady=(10, 5))

        tk.Label(parent, text='Epochs:').grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.epochs_entry = tk.Entry(parent, width=6)
        self.epochs_entry.insert(0, '10')
        self.epochs_entry.grid(row=1, column=1, sticky='w', pady=5)

        tk.Label(parent, text='Batch size:').grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.batch_entry = tk.Entry(parent, width=6)
        self.batch_entry.insert(0, '16')
        self.batch_entry.grid(row=2, column=1, sticky='w', pady=5)

        tk.Label(parent, text='Learning rate:').grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.lr_entry = tk.Entry(parent, width=6)
        self.lr_entry.insert(0, '0.01')
        self.lr_entry.grid(row=3, column=1, sticky='w', pady=5)

        # Start button
        tk.Button(parent, text='Start Training', command=self.start_training).grid(row=4, column=0, columnspan=2,
                                                                                   pady=(10, 5))

        # Log output
        tk.Label(parent, text='Training Log:').grid(row=5, column=0, columnspan=2, sticky='w', padx=5)
        self.log_text = tk.Text(parent, height=15, bg='#2b2b2b', fg='#ffffff')
        self.log_text.grid(row=6, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        parent.rowconfigure(6, weight=1)

    def start_training(self):
        try:
            epochs = int(self.epochs_entry.get())
            batch = int(self.batch_entry.get())
            lr = float(self.lr_entry.get())
        except ValueError:
            messagebox.showerror('Invalid parameters',
                                 'Please enter valid numeric values for epochs, batch size, and learning rate.')
            return
        self.log_text.insert('end', f'Starting training: epochs={epochs}, batch={batch}, lr={lr}\n')
        self.log_text.see('end')
        threading.Thread(target=self._run_training, args=(epochs, batch, lr), daemon=True).start()

    def _run_training(self, epochs, batch_size, lr):
        # Prepare a YOLO-compatible data dict
        data_dict = {
            'train': LABELED_DIR,
            'val': LABELED_DIR,
            'nc': len(SKILLS),
            'names': SKILLS
        }
        self.yolo_model.train(data=data_dict,
                              epochs=epochs,
                              batch=batch_size,
                              lr0=lr)
        self.log_text.insert('end', 'Training completed.\n')
        self.log_text.see('end')

    def load_image(self):
        # clear canvas
        self.canvas.delete('all')

        if not self.files:
            self.count_label.config(text="✅ All images processed!")
            return

        # reset zoom & pan
        self.zoom, self.origin_x, self.origin_y = 1.0, 0, 0

        # pick file
        if self.random_order.get():
            self.current_index = random.randint(0, len(self.files) - 1)
        fn = self.files[self.current_index]
        img = Image.open(os.path.join(CROP_DIR, fn)).convert('RGB')
        self.current_img = img
        self.count_label.config(text=f"Remaining: {len(self.files)}")

        # YOLO
        results = self.yolo_model.predict(img, verbose=False)[0]
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = (results.boxes.conf.cpu().numpy() * 100).tolist()
        classes = results.boxes.cls.cpu().numpy().astype(int).tolist()

        # split drops vs skills
        self.current_boxes.clear()
        raw_skills = []
        for (x0, y0, x1, y1), conf, cls in zip(xyxy, confs, classes):
            if cls == 0:
                self.current_boxes.append(((x0, y0, x1, y1), conf))
            else:
                yc = (y0 + y1) / 2
                raw_skills.append({'cls': cls, 'conf': conf / 100.0, 'y': yc, 'xyxy': (x0, y0, x1, y1)})

        # sort & pair
        self.current_boxes.sort(key=lambda t: t[0][1])
        raw_skills.sort(key=lambda r: r['y'])
        drop_centers = [((y0 + y1) / 2) for ((x0, y0, x1, y1), _) in self.current_boxes]
        pairs, paired = [], [None] * len(drop_centers)
        for i, dy in enumerate(drop_centers):
            for j, sk in enumerate(raw_skills):
                pairs.append((abs(dy - sk['y']), i, j))
        for _, i, j in sorted(pairs):
            if paired[i] is None and j not in {p[1] for p in pairs if p[1] == i and p[2] != j}:
                paired[i] = raw_skills[j]
        self.raw_skills = raw_skills
        self.paired_skills = paired

        # build skill-box coords
        self.skill_boxes_raw = []
        self.skill_boxes_fallback = []
        self.use_raw_skill = []
        for idx, ((x0, y0, x1, y1), _) in enumerate(self.current_boxes):
            ix1, ix0 = x0 - 2.5, x0 - 28.0
            yc = (y0 + y1 + 3) / 2;
            iy0, iy1 = yc - 11.75, yc + 11.75
            fb = (ix0, iy0, ix1, iy1)
            raw = self.paired_skills[idx]['xyxy'] if self.paired_skills[idx] else None
            self.skill_boxes_fallback.append(fb)
            self.skill_boxes_raw.append(raw)
            self.use_raw_skill.append(bool(raw))
        self.skill_boxes = [
            (raw if use else fb)
            for raw, fb, use in zip(self.skill_boxes_raw,
                                    self.skill_boxes_fallback,
                                    self.use_raw_skill)
        ]

        # drop‑count
        if self.current_boxes and all(c >= self.drop_threshold for _, c in self.current_boxes):
            cnt = len(self.current_boxes)
        else:
            cnt = 0
        self.count_spin.delete(0, 'end')
        self.count_spin.insert(0, str(cnt))
        self.last_count = cnt
        self._on_count_changed()

        # CRNN
        self._compute_crnn_preds()

        # **apply** either predicted-auto-fill or last-saved
        self._apply_fill_mode()

        # redraw + refresh preds
        self._draw_image()
        self.load_pred_frame()

        # focus first
        if self.xp_entries:
            self.xp_entries[0].focus_set()
            self.xp_entries[0].selection_range(0, 'end')

    def _compute_crnn_preds(self):
        self.xp_preds.clear()
        self.xp_confs.clear()
        if not self.current_img or not self.current_boxes:
            return

        BLANK_IDX = 10  # your CTC “blank” token index

        for box, _ in self.current_boxes:
            # crop & preprocess
            crop = self.current_img.crop(tuple(map(int, box)))
            t = self.ctc_preprocess(crop).unsqueeze(0)  # (1,1,H,W)

            with torch.no_grad():
                logp = self.ctc_model(t)
                # normalize to (T, C)
                if logp.dim() == 3 and logp.shape[0] == 1:
                    probs = logp.squeeze(0).softmax(dim=1)  # (T, C)
                else:
                    probs = logp.permute(1, 0, 2)[0].softmax(dim=1)

                pred = probs.argmax(dim=1)  # (T,)

            # --- collapse repeats + remove blanks ---
            seq = []
            confs = []
            prev = BLANK_IDX
            for timestep, p in enumerate(pred):
                p = int(p.item())
                if p != prev and p != BLANK_IDX:
                    seq.append(p)
                    confs.append(probs[timestep, p].item())
                prev = p

            # build string & average confidence
            xp_str = ''.join(str(d) for d in seq)
            avg_conf = (sum(confs) / len(confs) * 100) if confs else 0.0

            self.xp_preds.append(xp_str)
            self.xp_confs.append(avg_conf)

    def _draw_image(self):
        if not self.current_img:
            return

        # 1) Canvas + image sizing
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base   = min(cw/iw, ch/ih)
        sc     = base * self.zoom
        nw, nh = int(iw*sc), int(ih*sc)
        if nw <= 0 or nh <= 0:
            return

        # 2) Prepare layers
        img2     = self.current_img.resize((nw, nh), Image.LANCZOS).convert("RGBA")
        overlay  = Image.new("RGBA", img2.size, (0, 0, 0, 0))
        draw     = ImageDraw.Draw(overlay)
        font     = ImageFont.load_default()
        big_font = ImageFont.truetype("arial.ttf", 20)
        bw       = max(1, int(self.zoom))

        for i, ((x0, y0, x1, y1), drop_conf) in enumerate(self.current_boxes):
            # build scaled drop coords for the rectangle (unchanged)
            sx0, sy0, sx1, sy1 = [v * sc for v in (x0, y0, x1, y1)]

            # draw drop rectangle & confidence as before
            if self.show_boxes.get():
                col = (0, 255, 0, 255) if drop_conf >= 80 else (255, 165, 0, 255) if drop_conf >= 70 else (
                255, 0, 0, 255)
                draw.rectangle([sx0, sy0, sx1, sy1], outline=col, width=bw)
                draw.text((sx0, max(0, sy0 - 12)), f"{drop_conf:.0f}%", fill=col, font=font)

            # —— NEW: get the i‑th skill box (raw or fallback) and scale it
            sb = self.skill_boxes[
                i]  # active skill box coords :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
            sk_x0, sk_y0, sk_x1, sk_y1 = [v * sc for v in sb]

            # (1) Draw predicted XP to the left of the skill box
            xp = str(self.xp_preds[i])
            xp_conf = self.xp_confs[i]
            xp_col = (0, 255, 0, 255) if xp_conf >= 80 else (255, 165, 0, 255) if xp_conf >= 70 else (255, 0, 0, 255)
            bb_xp = draw.textbbox((0, 0), xp, font=big_font)
            tw, th = bb_xp[2] - bb_xp[0], bb_xp[3] - bb_xp[1]
            tx = sk_x0 - 5 - tw
            ty = sk_y0
            draw.text((tx, ty), xp, fill=xp_col, font=big_font)

            # (2) Draw predicted skill name just below XP, also left of the skill box
            rec = self.paired_skills[i]
            label = rec and self.yolo_model.names[rec['cls']] or "N/A"
            sk_conf = rec['conf'] * 100 if rec else 0
            sk_col = (0, 255, 0, 255) if sk_conf >= 80 else (255, 165, 0, 255) if sk_conf >= 70 else (255, 0, 0, 255)
            bb_sk = draw.textbbox((0, 0), label, font=big_font)
            tw2 = bb_sk[2] - bb_sk[0]
            sk_tx = sk_x0 - 5 - tw2
            sk_ty = sk_y0 + th
            draw.text((sk_tx, sk_ty), label, fill=sk_col, font=big_font)

            # 4) Draw the skill/fallback boxes underneath as before
            if self.show_boxes.get():
                for j, use_raw in enumerate(self.use_raw_skill):
                    bx = (self.skill_boxes_raw[j] if use_raw and self.skill_boxes_raw[j]
                          else self.skill_boxes_fallback[j])
                    ssx0, ssy0, ssx1, ssy1 = [v * sc for v in bx]
                    outline = (255, 0, 255, self.box_opacity) if use_raw and self.skill_boxes_raw[j] else (
                    0, 255, 255, self.box_opacity)
                    draw.rectangle([ssx0, ssy0, ssx1, ssy1], outline=outline, width=bw)

        # 4) Composite & display
        comp       = Image.alpha_composite(img2, overlay).convert("RGB")
        self.tkimg = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.origin_x + cw//2,
            self.origin_y + ch//2,
            image=self.tkimg,
            anchor='center'
        )

    def load_pred_frame(self):
        # clear old widgets
        for w in self.pred_frame.winfo_children():
            w.destroy()

        big_font = ('Helvetica', 15, 'bold')
        med_font = ('Helvetica', 12)

        # DEBUG output
        print("[DEBUG] ===== Per-Drop Summary =====")
        for i, ((x0, y0, x1, y1), drop_conf) in enumerate(self.current_boxes, start=1):
            xp, xp_c = self.xp_preds[i - 1], self.xp_confs[i - 1]
            rec = self.paired_skills[i - 1]
            sk_name = self.yolo_model.names[rec['cls']] if rec else "N/A"
            sk_pct = (rec['conf'] * 100) if rec else 0.0
            print(f"  Drop {i}: Xp={xp} ({xp_c:.0f}%) | DropConf={drop_conf:.1f}% | Skill={sk_name} ({sk_pct:.0f}%)")

        # render rows
        for i, ((x0, y0, x1, y1), drop_conf) in enumerate(self.current_boxes):
            fr = tk.Frame(self.pred_frame)
            fr.pack(anchor='center', pady=2)

            # XP value
            xp, xp_c = self.xp_preds[i], self.xp_confs[i]
            xp_col = 'lime' if xp_c >= self.xp_threshold else 'orange' if xp_c >= self.xp_threshold - 10 else 'red'
            tk.Label(fr, text=f"Xp:", font=med_font).pack(side='left')
            tk.Label(fr, text=str(xp), fg=xp_col, font=big_font).pack(side='left')
            tk.Label(fr, text=f"({xp_c:.1f}%)", font=med_font).pack(side='left', padx=(0, 10))

            # Skill name
            if rec:
                sk_col = 'lime' if sk_pct >= self.skill_threshold else 'orange' if sk_pct >= self.skill_threshold - 10 else 'red'
            else:
                sk_col = 'red'
            tk.Label(fr, text="Skill:", font=med_font).pack(side='left')
            tk.Label(fr, text=sk_name, fg=sk_col, font=big_font).pack(side='left')
            tk.Label(fr, text=f"({sk_pct:.1f}%)", font=med_font).pack(side='left', padx=(0, 10))

            # Drop confidence
            dc_col = 'lime' if drop_conf >= self.drop_threshold else 'orange' if drop_conf >= self.drop_threshold - 10 else 'red'
            tk.Label(fr, text="Drop:", font=med_font).pack(side='left')
            tk.Label(fr, text=f"{drop_conf:.1f}%", fg=dc_col, font=big_font).pack(side='left', padx=(0, 10))

            # Remove & Use Predicted
            tk.Button(fr, text="Remove", fg='red',
                      command=lambda idx=i: self.remove_box(idx)
                      ).pack(side='left', padx=(0, 10))
            var = tk.BooleanVar(value=self.use_raw_skill[i])
            tk.Checkbutton(fr, text="Use Predicted", variable=var,
                           command=lambda idx=i, v=var: self._toggle_skill(idx, v)
                           ).pack(side='left')
    def _toggle_skill(self, idx, var):
        # flip between YOLO‐raw or fallback
        self.use_raw_skill[idx] = var.get()

        # rebuild for click/zoom
        self.skill_boxes = []
        for raw, fb, use in zip(self.skill_boxes_raw,
                                self.skill_boxes_fallback,
                                self.use_raw_skill):
            self.skill_boxes.append(raw if use and raw else fb)

        self._draw_image()

    def apply_random_order_logic(self):
        if self.random_order.get() and self.files:
            self.current_index = random.randint(0, len(self.files) - 1)
        else:
            self.current_index %= len(self.files)

    def on_click(self, event):
        if not self.current_img:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base = min(cw/iw, ch/ih)
        # map canvas coords back to image coords
        ix = (event.x - (cw - iw*base)/2) / base
        iy = (event.y - (ch - ih*base)/2) / base

        # check each active skill box
        for idx, (x0, y0, x1, y1) in enumerate(self.skill_boxes):
            if x0 <= ix <= x1 and y0 <= iy <= y1:
                self.drag_idx = idx
                self._zoom_to_box(x0, y0, x1, y1)
                self.canvas.focus_set()
                return

        # click outside any box resets zoom
        self.drag_idx = None
        self.zoom = 1.0
        self.origin_x = 0
        self.origin_y = 0
        self._draw_image()

    def move_skill_left(self, e):  self._move_selected_skill_box(-1,  0)
    def move_skill_right(self,e):  self._move_selected_skill_box( 1,  0)
    def move_skill_up(self,    e):  self._move_selected_skill_box( 0, -1)
    def move_skill_down(self,  e):  self._move_selected_skill_box( 0,  1)

    def _move_selected_skill_box(self, dx, dy):
        if self.drag_idx is None or not (0<=self.drag_idx<len(self.skill_boxes)):
            return
        x0,y0,x1,y1 = self.skill_boxes[self.drag_idx]
        self.skill_boxes[self.drag_idx] = (x0+dx, y0+dy, x1+dx, y1+dy)
        self._draw_image()

    def _zoom_to_box(self, x0,y0,x1,y1):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base = min(cw/iw, ch/ih)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        zx, zy = iw/(x1-x0), ih/(y1-y0)
        self.zoom = max(1.5, min(5.0, min(zx,zy)*2.0))
        sc = base*self.zoom
        sw, sh = iw*sc, ih*sc
        offx, offy = (cw-sw)/2, (ch-sh)/2
        scx, scy = cx*sc, cy*sc
        self.origin_x = cw/2 - (offx+scx)
        self.origin_y = ch/2 - (offy+scy)
        self._draw_image()

    def _on_opacity_changed(self, v):
        self.box_opacity = int(v)
        self._draw_image()

    def _on_canvas_resize(self, e):
        self._draw_image()

    def auto_prune_images(self):
        files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith('.png'))
        for fn in tqdm(files, desc="Auto-pruning"):
            img = Image.open(os.path.join(CROP_DIR, fn)).convert('RGB')
            res = self.yolo_model.predict(img, verbose=False)[0]
            if not len(res.boxes):
                shutil.move(os.path.join(CROP_DIR, fn), os.path.join(SKIP_DIR, fn))
                self.auto_skipped += 1
        self.files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith('.png'))
        self.current_index = 0
        self.load_image()

    def _on_value_changed(self, e):
        v = self.value_entry.get().strip()
        self.last_value = v
        self.defaults[self.last_count] = (v, self.last_nonzero_skills.copy())

    def _on_xp_changed(self, idx):
        # collect current XP inputs
        xp_vals = [e.get().strip() for e in self.xp_entries]
        # update defaults for this drop count
        self.defaults[self.last_count] = (xp_vals, self.last_nonzero_skills.copy())

    def _on_count_changed(self):
        try:
            cnt = int(self.count_spin.get())
        except ValueError:
            return
        self.last_count = cnt

        # clear everything if zero drops
        if cnt == 0:
            # clear XP boxes
            for w in self.xp_frame.winfo_children():
                w.destroy()
            # clear skill selectors
            for w in self.skill_frame.winfo_children():
                w.destroy()
            self.last_nonzero_skills = []
            self.defaults[0] = ([], [])
            return

        # restore or initialize defaults
        if cnt in self.defaults:
            xp_def, skills = self.defaults[cnt]
            xp_list = xp_def if isinstance(xp_def, list) else (xp_def.split(';') if xp_def else [])
        else:
            xp_list = []
            skills = []

        # render dynamic XP entries & skill selectors
        self.defaults[cnt] = (xp_list, skills.copy())
        self._render_xp_entries(cnt)
        self.last_nonzero_skills = skills.copy()
        self._render_skill_selectors(cnt)

    def _render_skill_selectors(self, cnt):
        # clear old widgets
        for w in self.skill_frame.winfo_children():
            w.destroy()
        self.skill_selectors = []

        for i in range(cnt):
            cb = ttk.Combobox(self.skill_frame, values=SKILLS, state='normal', width=12)
            cb.grid(row=0, column=i, padx=2)

            # if we have a stored skill, use it; otherwise default to 'None'
            default_skill = (
                self.last_nonzero_skills[i]
                if i < len(self.last_nonzero_skills)
                else 'None'
            )
            cb.set(default_skill)

            # bind events
            cb.bind('<<ComboboxSelected>>', lambda e, idx=i: self._on_skill_changed(idx))
            cb.bind('<KeyRelease>', lambda e, idx=i: self._on_skill_type(e, idx))

            self.skill_selectors.append(cb)

    def _on_skill_changed(self, idx):
        self.last_nonzero_skills[idx] = self.skill_selectors[idx].get()
        self.defaults[self.last_count] = (self.last_value, self.last_nonzero_skills.copy())

    def prev_image(self):
        if not self.files: return
        if self.random_order.get():
            self.current_index = random.randint(0, len(self.files) - 1)
        else:
            self.current_index = (self.current_index - 1) % len(self.files)
        self.load_image()

    def next_image(self):
        if not self.files:
            return
        if self.random_order.get():
            self.current_index = random.randint(0, len(self.files) - 1)
        else:
            self.current_index = (self.current_index + 1) % len(self.files)
        self.load_image()

    def skip_image(self):
        if not self.files: return
        idx, fn = self.current_index, self.files.pop(self.current_index)
        shutil.move(os.path.join(CROP_DIR, fn), os.path.join(SKIP_DIR, fn))
        self.action_history.append({'type': 'skip', 'filename': fn, 'idx': idx})
        if self.files:
            self.apply_random_order_logic()
        else:
            self.current_index = 0
        self.load_image()

    def save_label(self):
        if not self.files:
            return

        fn = self.files[self.current_index]
        cnt = self.last_count

        # gather inputs
        xp_vals = [e.get().strip() for e in self.xp_entries] if cnt > 0 else []
        skills = [cb.get().strip() for cb in self.skill_selectors] if cnt > 0 else []

        # validations
        if cnt > 0:
            if any(not v for v in xp_vals):
                self.status_label.config(text="Cannot save: XP missing", fg='red')
                return
            if len(skills) != cnt or any(s == "" or s.lower() == "none" for s in skills):
                self.status_label.config(text="Cannot save: All skills must be selected", fg='red')
                return
        if xp_vals and cnt == 0:
            self.status_label.config(text="Cannot save: XP without drops", fg='red')
            return

        # record defaults for this drop count
        self.defaults[cnt] = (xp_vals.copy(), skills.copy())

        # move & write files
        src = os.path.join(CROP_DIR, fn)
        dst = os.path.join(LABELED_DIR, fn)
        shutil.move(src, dst)

        # write .txt
        img = Image.open(dst)
        iw, ih = img.size
        base, _ = os.path.splitext(fn)
        with open(os.path.join(LABELED_DIR, base + ".txt"), 'w') as f:
            # drops
            for box, _ in self.current_boxes:
                x0, y0, x1, y1 = box
                cx, cy = ((x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih)
                w, h = ((x1 - x0) / iw, (y1 - y0) / ih)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            # skills
            for i in range(cnt):
                raw_ok = i < len(self.skill_boxes_raw) and self.skill_boxes_raw[i]
                fallback_ok = i < len(self.skill_boxes_fallback) and self.skill_boxes_fallback[i]
                use_raw = self.use_raw_skill[i] if i < len(self.use_raw_skill) else False
                if use_raw and raw_ok:
                    bx = self.skill_boxes_raw[i]
                elif fallback_ok:
                    bx = self.skill_boxes_fallback[i]
                else:
                    continue
                sx0, sy0, sx1, sy1 = bx
                name = skills[i]
                cid = SKILLS.index(name) if name in SKILLS else 0
                cx, cy = ((sx0 + sx1) / 2 / iw, (sy0 + sy1) / 2 / ih)
                w, h = ((sx1 - sx0) / iw, (sy1 - sy0) / ih)
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # write .json
        data = {
            "filename": fn,
            "drop_count": cnt,
            "xp_values": xp_vals,
            "skills": skills,
            "boxes": []
        }
        # drops
        for box, _ in self.current_boxes:
            x0, y0, x1, y1 = box
            data["boxes"].append({
                "class": 0,
                "center_x": float((x0 + x1) / 2 / iw),
                "center_y": float((y0 + y1) / 2 / ih),
                "width": float((x1 - x0) / iw),
                "height": float((y1 - y0) / ih)
            })
        # skills
        for i in range(cnt):
            raw_ok = i < len(self.skill_boxes_raw) and self.skill_boxes_raw[i]
            fallback_ok = i < len(self.skill_boxes_fallback) and self.skill_boxes_fallback[i]
            use_raw = self.use_raw_skill[i] if i < len(self.use_raw_skill) else False
            if use_raw and raw_ok:
                bx = self.skill_boxes_raw[i]
            elif fallback_ok:
                bx = self.skill_boxes_fallback[i]
            else:
                continue
            sx0, sy0, sx1, sy1 = bx
            name = skills[i]
            cid = SKILLS.index(name) if name in SKILLS else 0
            data["boxes"].append({
                "class": cid,
                "center_x": float((sx0 + sx1) / 2 / iw),
                "center_y": float((sy0 + sy1) / 2 / ih),
                "width": float((sx1 - sx0) / iw),
                "height": float((sy1 - sy0) / ih)
            })
        with open(os.path.join(LABELED_DIR, base + ".json"), 'w') as f:
            json.dump(data, f, indent=2)

        # finalize
        self.files.pop(self.current_index)
        self.status_label.config(text=f"Saved {fn}", fg='green')
        # immediately re-apply fill mode so unchecking restores saved defaults
        self._apply_fill_mode()

        if self.files:
            self.apply_random_order_logic()
        else:
            self.current_index = 0
        self.load_image()

    def remove_box(self, idx):
        # only if valid
        if not (0 <= idx < len(self.current_boxes)):
            return

        # 1) remove the drop box + confidences
        del self.current_boxes[idx]
        del self.xp_preds[idx]
        del self.xp_confs[idx]

        # 2) remove the paired‐skill entries
        del self.paired_skills[idx]
        del self.skill_boxes_raw[idx]
        del self.skill_boxes_fallback[idx]
        del self.use_raw_skill[idx]

        # 3) rebuild the active skill_boxes list for drawing & click
        self.skill_boxes = []
        for raw, fb, use in zip(self.skill_boxes_raw,
                                self.skill_boxes_fallback,
                                self.use_raw_skill):
            self.skill_boxes.append(raw if use and raw else fb)

        # 4) update the drop‐count spinbox and defaults
        new_count = len(self.current_boxes)
        self.count_spin.delete(0, 'end')
        self.count_spin.insert(0, str(new_count))
        self._on_count_changed()

        # 5) redraw image and repopulate the prediction frame
        self._draw_image()
        self.load_pred_frame()

        # 6) restore focus & selection on the XP entry
        self.value_entry.focus_set()
        self.value_entry.selection_range(0, 'end')

    def undo_label(self):
        if not self.action_history:
            return self.status_label.config(text="Nothing to undo.", fg='gray')
        act = self.action_history.pop()
        fn, idx = act['filename'], act['idx']
        if act['type'] == "save":
            shutil.move(os.path.join(LABELED_DIR, fn), os.path.join(CROP_DIR, fn))
        else:
            shutil.move(os.path.join(SKIP_DIR, fn), os.path.join(CROP_DIR, fn))
        self.files.insert(idx, fn)
        self.current_index = idx
        self.status_label.config(text=f"Undid {'save' if act['type'] == 'save' else 'skip'} of {fn}", fg='blue')
        self.load_image()
        self.update_stats()

    def update_stats(self):
        total = len(self.labels)
        rem   = len(self.files)
        drops = sum(c for *_,c in self.labels.values())
        self.stats_label.config(
            text=f"Labeled: {total} — Remaining: {rem} — Auto-skipped: {self.auto_skipped} | Total drops: {drops}"
        )

    def _on_skill_type(self, event, idx):
        w = event.widget; typed = w.get()
        for s in SKILLS:
            if s.lower().startswith(typed.lower()):
                w.delete(0,'end'); w.insert(0,s); w.select_range(len(typed),'end')
                w.icursor(len(s)); break
        if hasattr(self, 'skill_selectors'):
            self._on_skill_changed(idx)

    def _focus_first_skill(self, event):
        if hasattr(self, 'skill_selectors') and self.skill_selectors:
            self.skill_selectors[0].focus_set()
        return 'break'

    def _render_xp_entries(self, cnt):
        for w in self.xp_frame.winfo_children():
            w.destroy()
        self.xp_entries = []

        raw_def = self.defaults.get(cnt, ("", []))[0]
        xp_defaults = raw_def if isinstance(raw_def, list) else (raw_def.split(';') if raw_def else [])

        for i in range(cnt):
            e = tk.Entry(self.xp_frame, width=6)  # ← width=6
            e.grid(row=0, column=i, padx=2, sticky='w')  # ← sticky='w'
            if i < len(xp_defaults):
                e.insert(0, xp_defaults[i])
            e.bind('<KeyRelease>', lambda ev, idx=i: self._on_xp_changed(idx))
            if i == cnt - 1:
                e.bind('<Tab>', lambda ev: self._focus_first_skill(ev))
            self.xp_entries.append(e)

    def _on_drop_thresh_changed(self, v):
        try:
            self.drop_threshold = float(v)
        except ValueError:
            return

        # --- only re‑apply drop count auto‑fill ---
        if self.current_boxes and all(conf >= self.drop_threshold for _, conf in self.current_boxes):
            cnt = len(self.current_boxes)
        else:
            cnt = 0
        self.count_spin.delete(0, 'end')
        self.count_spin.insert(0, str(cnt))
        self._on_count_changed()  # rebuild XP/skill widgets for the new count

    def _on_xp_thresh_changed(self, v):
        try:
            self.xp_threshold = float(v)
        except ValueError:
            return

        # --- only re‑apply the XP auto‑fill block, without reloading the image ---
        if self.xp_confs and all(c >= self.xp_threshold for c in self.xp_confs):
            for i, val in enumerate(self.xp_preds):
                if i < len(self.xp_entries):
                    self.xp_entries[i].delete(0, 'end')
                    self.xp_entries[i].insert(0, val)
            # update defaults so Undo/etc still works
            self.defaults[self.last_count] = (
                [e.get().strip() for e in self.xp_entries],
                self.last_nonzero_skills.copy()
            )

    def _on_skill_thresh_changed(self, v):
        try:
            self.skill_threshold = float(v)
        except ValueError:
            return

        # --- only re‑apply the skill auto‑fill block ---
        cnt = self.last_count
        if cnt > 0 and hasattr(self, 'skill_selectors'):
            for i, rec in enumerate(self.paired_skills):
                if rec and rec['conf'] * 100 >= self.skill_threshold:
                    name = self.yolo_model.names[rec['cls']]
                    self.skill_selectors[i].set(name)
            self.last_nonzero_skills = [cb.get() for cb in self.skill_selectors]
            self.defaults[cnt] = (self.last_value, self.last_nonzero_skills.copy())

    def _snap_slider_to_5(self, slider, handler):
        """
        If slider value is within ±2.5 of a multiple of 5, snap it there
        and call the given handler with the new value.
        """
        try:
            v = float(slider.get())
        except ValueError:
            return

        rem = v % 5
        snap_tol = 2.5
        if rem <= snap_tol:
            new_v = v - rem
        elif rem >= 5 - snap_tol:
            new_v = v + (5 - rem)
        else:
            return

        # clamp 0–100
        new_v = max(0, min(100, new_v))
        slider.set(new_v)
        handler(str(new_v))

    def _apply_fill_mode(self, *args):
        cnt = self.last_count

        if self.fill_with_predicted.get():
            # auto-fill from predictions
            if self.xp_confs and all(c >= self.xp_threshold for c in self.xp_confs):
                for i, val in enumerate(self.xp_preds):
                    if i < len(self.xp_entries):
                        self.xp_entries[i].delete(0, 'end')
                        self.xp_entries[i].insert(0, val)
                self.defaults[cnt] = ([e.get().strip() for e in self.xp_entries],
                                      self.last_nonzero_skills.copy())
            if cnt > 0 and hasattr(self, 'skill_selectors'):
                for i, rec in enumerate(self.paired_skills):
                    if rec and rec['conf'] * 100 >= self.skill_threshold:
                        self.skill_selectors[i].set(self.yolo_model.names[rec['cls']])
                self.last_nonzero_skills = [cb.get() for cb in self.skill_selectors]
                self.defaults[cnt] = (self.last_value, self.last_nonzero_skills.copy())
        else:
            # restore from last saved defaults
            xp_def, sk_def = self.defaults.get(cnt, ([], []))
            for i, val in enumerate(xp_def):
                if i < len(self.xp_entries):
                    self.xp_entries[i].delete(0, 'end')
                    self.xp_entries[i].insert(0, val)
            for i, name in enumerate(sk_def):
                if i < len(self.skill_selectors):
                    self.skill_selectors[i].set(name)


if __name__ == '__main__':
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

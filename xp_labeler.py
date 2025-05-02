#!/usr/bin/env python3
import json
import os
import random
import shutil
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
CRNN_MODEL_PATH = os.path.join("models", "crnn.pt")  # your standalone CRNN

SKILLS = [
    "drop",         # class 0
    "agility",      # class 1
    "attack",       # class 2
    "construction", # class 3
    "cooking",      # class 4
    "crafting",     # class 5
    "defence",      # class 6
    "farming",      # class 7
    "firemaking",   # class 8
    "fishing",      # class 9
    "fletching",    # class 10
    "herblore",     # class 11
    "hitpoints",    # class 12
    "hunter",       # class 13
    "magic",        # class 14
    "mining",       # class 15
    "prayer",       # class 16
    "ranged",       # class 17
    "runecrafting", # class 18
    "smithing",     # class 19
    "slayer",       # class 20
    "strength",     # class 21
    "thieving",     # class 22
    "woodcutting"   # class 23
]


class XPLabeler:
    def __init__(self, root):
        self.root = root
        root.title("XP Labeler")

        # === Dark Theme Colors ===
        self.bg_color = '#1e1e1e'; self.fg_color = '#ffffff'
        self.button_bg = '#333333'; self.button_fg = '#ffffff'
        self.entry_bg = '#2b2b2b'; self.entry_fg = '#ffffff'
        self.highlight_color = '#5e9cff'; self.shortcut_color = '#CCCC66'
        root.option_add('*Font', ('Helvetica', 12))
        # Global style
        for opt in ['Background','Frame.background','Label.background']:
            root.option_add(f'*{opt}', self.bg_color)
        for opt in ['Foreground','Label.foreground','Entry.foreground']:
            root.option_add(f'*{opt}', self.fg_color)
        root.option_add('*Entry.background', self.entry_bg)
        root.option_add('*Spinbox.background', self.entry_bg)
        root.option_add('*Button.background', self.button_bg)
        root.option_add('*Button.foreground', self.button_fg)
        root.option_add('*Scale.troughColor', self.button_bg)
        style = ttk.Style(); style.theme_use('clam')
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
            transforms.Resize((64,128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        # === State ===
        self.files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith('.png'))
        self.labels = {}
        self.defaults = {}
        self.auto_skipped = 0
        self.last_value = ""
        self.last_count = 0
        self.last_nonzero_skills = []
        self.show_boxes = tk.BooleanVar(value=True)

        # === Layout ===
        root.rowconfigure(0, weight=1); root.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(root, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<Button-1>', self.on_click)
        self.zoom = 1.0; self.origin_x = 0; self.origin_y = 0
        self.canvas.bind('<Left>', self.move_skill_left)
        self.canvas.bind('<Right>', self.move_skill_right)
        self.canvas.bind('<Up>', self.move_skill_up)
        self.canvas.bind('<Down>', self.move_skill_down)

        # Shortcuts
        self.root.bind('<Return>', lambda e: self.save_label())
        self.root.bind('<Control-s>', lambda e: self.skip_image())
        self.root.bind('<Control-u>', lambda e: self.undo_label())

        # Controls
        ctrl = tk.Frame(root); ctrl.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        self.count_label = tk.Label(ctrl, anchor='w', fg=self.highlight_color); self.count_label.pack(fill='x')
        self.stats_label = tk.Label(ctrl, anchor='w', fg=self.highlight_color); self.stats_label.pack(fill='x', pady=(0,5))

        frm = tk.Frame(ctrl); frm.pack(fill='x', pady=(0,5))
        tk.Label(frm, text='XP values (semicolon):').grid(row=0, column=0, sticky='e')
        self.value_entry = tk.Entry(frm, width=20); self.value_entry.grid(row=0, column=1, padx=(2,10))
        self.value_entry.bind('<KeyRelease>', self._on_value_changed)
        self.value_entry.bind('<Tab>', self._focus_first_skill)

        tk.Label(frm, text='Drop count:').grid(row=0, column=2, sticky='e')
        self.count_spin = tk.Spinbox(frm, from_=0, to=20, width=5, command=self._on_count_changed)
        self.count_spin.grid(row=0, column=3, sticky='w')

        tk.Label(frm, text='Skills:').grid(row=1, column=0, sticky='e')
        self.skill_frame = tk.Frame(frm); self.skill_frame.grid(row=1, column=1, columnspan=3, sticky='w')

        nav = tk.Frame(ctrl);
        nav.pack(fill='x', pady=(0, 5))
        tk.Button(nav, text='Prev', command=self.prev_image).pack(side='left')
        tk.Button(nav, text='Next', command=self.next_image).pack(side='left', padx=5)  # ← NEW
        tk.Button(nav, text='Skip', command=self.skip_image).pack(side='left', padx=5)
        self.save_btn = tk.Button(nav, text='Save Label', command=self.save_label);
        self.save_btn.pack(side='left', padx=5)
        tk.Button(nav, text='Undo', command=self.undo_label).pack(side='left', padx=5)

        tk.Checkbutton(ctrl, text="Show YOLO Boxes", variable=self.show_boxes, command=self._draw_image).pack(pady=5)
        tk.Button(ctrl, text="Auto Prune Images", command=self.auto_prune_images).pack(pady=5)

        opacity_frame = tk.Frame(ctrl); opacity_frame.pack(fill='x', pady=(0,5))
        tk.Label(opacity_frame, text="Box Opacity:").pack(side='left')
        self.opacity_slider = tk.Scale(opacity_frame, from_=0, to=255, orient='horizontal',
                                       command=self._on_opacity_changed, length=100)
        self.opacity_slider.set(255); self.opacity_slider.pack(side='left', padx=(5,0))
        self.box_opacity = 255

        self.random_order = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Random Order", variable=self.random_order).pack(pady=5)

        self.status_label = tk.Label(ctrl, anchor='w'); self.status_label.pack(fill='x', pady=(0,5))
        self.pred_frame = tk.Frame(ctrl); self.pred_frame.pack(fill='x')
        tk.Label(ctrl,
                 text="Shortcuts: ←→↑↓ Move box | Enter: Save | Ctrl+S: Skip | Ctrl+U: Undo",
                 font=('Helvetica', 13, 'bold'),
                 fg=self.shortcut_color,
                 anchor='center',
                 justify='center').pack(fill='x', pady=(5,0))

        # Boxes & History
        self.current_img = None
        self.current_boxes = []
        self.skill_boxes = []
        self.xp_preds = []
        self.xp_confs = []
        self.drag_idx = None
        self.action_history = []
        self.current_index = 0

        if self.files:
            self.load_image()
            self.update_stats()

    def load_image(self):
        if not self.files:
            self.canvas.delete('all')
            self.count_label.config(text="✅ All images processed!")
            return

        # Reset pan/zoom
        self.zoom, self.origin_x, self.origin_y = 1.0, 0, 0

        # Pick next file (random or sequential)
        if self.random_order.get():
            self.current_index = random.randint(0, len(self.files) - 1)

        fn = self.files[self.current_index]
        img = Image.open(os.path.join(CROP_DIR, fn)).convert('RGB')
        self.current_img = img
        self.count_label.config(text=f"Remaining: {len(self.files)}")

        # YOLO inference
        results = self.yolo_model.predict(img, verbose=False)[0]
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Clear old
        self.current_boxes.clear()
        self.pred_skill_boxes = []  # raw YOLO skill detections
        # build drop‐boxes
        for (x0, y0, x1, y1), conf, cls in zip(xyxy, confs, classes):
            if cls == 0:
                # true drop
                self.current_boxes.append(((x0, y0, x1, y1), conf * 100))
            else:
                # only collect raw skill preds here
                self.pred_skill_boxes.append((x0, y0, x1, y1))

        # **DEBUG OUTPUT**
        print(f"[DEBUG] Detected {len(self.current_boxes)} drop box(es):")
        for i, ((x0, y0, x1, y1), conf) in enumerate(self.current_boxes, 1):
            print(f"  Drop {i}: ({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}), conf={conf:.1f}%")
        print(f"[DEBUG] Detected {len(self.pred_skill_boxes)} raw skill box(es):")
        for i, (x0, y0, x1, y1) in enumerate(self.pred_skill_boxes, 1):
            print(f"  SkillBox {i}: ({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")

        # Sort drops top-to-bottom
        self.current_boxes.sort(key=lambda item: item[0][1])

        # Now build one hard-coded skill box per drop (cyan)
        self.skill_boxes = []
        for (x0, y0, x1, y1), _ in self.current_boxes:
            ix1 = x0 - 2.5
            ix0 = ix1 - 25.5
            yc = (y0 + y1 + 3) / 2
            iy0 = yc - 11.75
            iy1 = yc + 11.75
            self.skill_boxes.append((ix0, iy0, ix1, iy1))

        # Autofill drop count
        cnt = len(self.current_boxes)
        self.count_spin.delete(0, 'end')
        self.count_spin.insert(0, str(cnt))
        self.last_count = cnt
        self._on_count_changed()

        # Compute CRNN preds + confidences
        self._compute_crnn_preds()

        # Auto-fill XP if confidences are high
        if self.xp_confs and all(c >= 90.0 for c in self.xp_confs):
            vals = ";".join(self.xp_preds)
            self.value_entry.delete(0, 'end')
            self.value_entry.insert(0, vals)
            self.last_value = vals

        # Draw & update
        self._draw_image()
        self.load_pred_frame()

        # Focus entry
        self.value_entry.focus_set()
        self.value_entry.selection_range(0, 'end')

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
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base = min(cw / iw, ch / ih)
        sc = base * self.zoom
        nw, nh = int(iw * sc), int(ih * sc)
        if nw <= 0 or nh <= 0:
            return

        # Prepare
        img2 = self.current_img.resize((nw, nh), Image.LANCZOS).convert("RGBA")
        overlay = Image.new('RGBA', img2.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        bw = max(1, int(self.zoom))

        if self.show_boxes.get():
            # 1) XP drop boxes (green/orange/red)
            for box, conf in self.current_boxes:
                x0, y0, x1, y1 = [v * sc for v in box]
                col = (
                    (0, 255, 0, 255) if conf >= 80 else
                    (255, 165, 0, 255) if conf >= 70 else
                    (255, 0, 0, 255)
                )
                draw.rectangle([x0, y0, x1, y1], outline=col, width=bw)
                draw.text((x0, max(0, y0 - 12)), f"{conf:.0f}%", fill=col, font=font)

            # 2) raw YOLO skill preds (magenta)
            for x0, y0, x1, y1 in self.pred_skill_boxes:
                sx0, sy0, sx1, sy1 = [v * sc for v in (x0, y0, x1, y1)]
                draw.rectangle(
                    [sx0, sy0, sx1, sy1],
                    outline=(255, 0, 255, self.box_opacity),
                    width=bw
                )

            # 3) hard-coded skill boxes (cyan)
            for x0, y0, x1, y1 in self.skill_boxes:
                sx0, sy0, sx1, sy1 = [v * sc for v in (x0, y0, x1, y1)]
                draw.rectangle(
                    [sx0, sy0, sx1, sy1],
                    fill=(0, 255, 255, 0),
                    outline=(0, 255, 255, self.box_opacity),
                    width=bw
                )

        # Composite & show
        comp = Image.alpha_composite(img2, overlay).convert("RGB")
        self.tkimg = ImageTk.PhotoImage(comp)
        self.canvas.delete('all')
        self.canvas.create_image(
            self.origin_x + cw // 2,
            self.origin_y + ch // 2,
            image=self.tkimg,
            anchor='center'
        )

    def load_pred_frame(self):
        # clear out old widgets
        for w in self.pred_frame.winfo_children():
            w.destroy()

        # 1) run YOLO once and collect ALL skill detections with their y‐centers
        results = self.yolo_model.predict(self.current_img, verbose=False)[0]
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        xyxy = results.boxes.xyxy.cpu().numpy()

        raw_skills = []
        for cls, conf, (x0, y0, x1, y1) in zip(classes, confs, xyxy):
            if int(cls) > 0:
                yc = (y0 + y1) / 2.0
                raw_skills.append({
                    'cls': int(cls),
                    'conf': float(conf),
                    'y': yc
                })

        # 2) compute drop centers (already sorted top→bottom)
        drop_centers = [((y0 + y1) / 2.0) for ((x0, y0, x1, y1), _) in self.current_boxes]
        n_drops = len(drop_centers)
        n_skills = len(raw_skills)

        # 3) build all possible (drop_i, skill_j, vertical distance)
        pairs = []
        for i, dy in enumerate(drop_centers):
            for j, sk in enumerate(raw_skills):
                dist = abs(dy - sk['y'])
                pairs.append((dist, i, j))
        pairs.sort(key=lambda x: x[0])

        # 4) greedy match: for each smallest-distance pair, assign if both unpaired
        paired_skills = [None] * n_drops
        used_skills = set()
        for dist, i, j in pairs:
            if paired_skills[i] is None and j not in used_skills:
                paired_skills[i] = raw_skills[j]
                used_skills.add(j)

        # 5) DEBUG: print per-drop summary
        print("[DEBUG] ===== Per-Drop Summary =====")
        for i, ((x0, y0, x1, y1), drop_conf) in enumerate(self.current_boxes, start=1):
            xp = self.xp_preds[i - 1]
            xp_c = self.xp_confs[i - 1]
            rec = paired_skills[i - 1]

            if rec is not None:
                sk_name = self.yolo_model.names[rec['cls']]
                sk_pct = rec['conf'] * 100
                sk_str = f"{sk_name} ({sk_pct:.0f}%)"
            else:
                sk_str = "N/A"

            print(f"  Drop {i}: Xp={xp} ({xp_c:.0f}%) | "
                  f"DropConf={drop_conf:.1f}% | Skill={sk_str}")
        print("[DEBUG] =============================")

        # 6) now render one GUI row per drop, with its paired skill
        for i, ((x0, y0, x1, y1), drop_conf) in enumerate(self.current_boxes):
            fr = tk.Frame(self.pred_frame)
            fr.pack(anchor='w', pady=2, fill='x')

            # --- CRNN XP prediction ---
            xp = self.xp_preds[i]
            xp_c = self.xp_confs[i]
            xp_col = 'lime' if xp_c >= 80 else \
                'orange' if xp_c >= 70 else \
                    'red'
            tk.Label(fr,
                     text=f"Xp value {i + 1}: {xp} ({xp_c:.0f}%)",
                     fg=xp_col).pack(side='left', padx=(0, 10))

            # --- drop confidence ---
            dc_col = 'lime' if drop_conf >= 80 else \
                'orange' if drop_conf >= 70 else \
                    'red'
            tk.Label(fr,
                     text=f"Drop {i + 1}: {drop_conf:.1f}%",
                     fg=dc_col).pack(side='left', padx=(0, 10))

            # --- matched skill prediction ---
            rec = paired_skills[i]
            if rec is not None:
                name = self.yolo_model.names[rec['cls']]
                pct = rec['conf'] * 100
                sk_col = 'lime' if pct >= 80 else \
                    'orange' if pct >= 70 else \
                        'red'
                tk.Label(fr,
                         text=f"Skill: {name} ({pct:.0f}%)",
                         fg=sk_col,
                         font=('Helvetica', 13, 'bold')
                         ).pack(side='left', padx=(0, 10))
            else:
                tk.Label(fr,
                         text="Skill: N/A",
                         fg='red',
                         font=('Helvetica', 13, 'bold')
                         ).pack(side='left', padx=(0, 10))

            # --- remove button ---
            tk.Button(fr,
                      text="Remove",
                      fg='red',
                      command=lambda idx=i: self.remove_box(idx)
                      ).pack(side='left')

    def apply_random_order_logic(self):
        if self.random_order.get() and self.files:
            self.current_index = random.randint(0, len(self.files) - 1)
        else:
            self.current_index %= len(self.files)

    def on_click(self, event):
        if not self.current_img: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base = min(cw/iw, ch/ih)
        ix = (event.x - (cw - iw*base)/2)/base
        iy = (event.y - (ch - ih*base)/2)/base
        for idx,(x0,y0,x1,y1) in enumerate(self.skill_boxes):
            if x0<=ix<=x1 and y0<=iy<=y1:
                self.drag_idx = idx
                self._zoom_to_box(x0,y0,x1,y1)
                self.canvas.focus_set()
                return
        self.drag_idx = None
        self.zoom = 1.0; self.origin_x = 0; self.origin_y = 0
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

    def _on_count_changed(self):
        try:
            cnt = int(self.count_spin.get())
        except ValueError:
            return
        self.last_count = cnt
        if cnt in self.defaults:
            xp_str, skills = self.defaults[cnt]
        else:
            xp_str = self.value_entry.get().strip()
            skills = [cb.get() for cb in getattr(self, 'skill_selectors', [])]
            skills = (skills + ['None']*cnt)[:cnt]
        self.value_entry.delete(0, 'end')
        self.value_entry.insert(0, xp_str)
        self.last_value = xp_str
        self.last_nonzero_skills = skills.copy()
        self.defaults[cnt] = (xp_str, skills.copy())
        self._render_skill_selectors(cnt)

    def _render_skill_selectors(self, cnt):
        for w in self.skill_frame.winfo_children():
            w.destroy()
        self.skill_selectors = []
        for i in range(cnt):
            cb = ttk.Combobox(self.skill_frame, values=SKILLS, state='normal', width=12)
            cb.grid(row=0, column=i, padx=2)
            cb.set(self.last_nonzero_skills[i])
            cb.bind('<<ComboboxSelected>>', lambda e, idx=i: self._on_skill_changed(idx))
            cb.bind('<KeyRelease>',      lambda e, idx=i: self._on_skill_type(e, idx))
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
        val = self.value_entry.get().strip()
        cnt = self.last_count

        # Build skills only if there are drops
        if cnt > 0:
            skills = [cb.get() for cb in self.skill_selectors]
        else:
            skills = []

        # Validation
        if cnt > 0 and not val:
            return self.status_label.config(text="Cannot save: XP missing", fg='red')
        if cnt > 0 and any(s == "None" for s in skills):
            return self.status_label.config(text="Cannot save: All skills needed", fg='red')
        if val and cnt == 0:
            return self.status_label.config(text="Cannot save: XP without drops", fg='red')

        # Record label
        self.labels[fn] = (val, skills, cnt)
        self.defaults[cnt] = (val, skills.copy())

        # Move file
        src = os.path.join(CROP_DIR, fn)
        dst = os.path.join(LABELED_DIR, fn)
        shutil.move(src, dst)

        # Write YOLO .txt (only drop boxes when cnt==0)
        img = Image.open(dst)
        iw, ih = img.size
        label_txt = os.path.splitext(fn)[0] + ".txt"
        with open(os.path.join(LABELED_DIR, label_txt), 'w') as f:
            # drop boxes
            for box, _ in self.current_boxes:
                x0, y0, x1, y1 = box
                cx, cy = ((x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih)
                w, h = ((x1 - x0) / iw, (y1 - y0) / ih)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # skill boxes only if cnt > 0
            if cnt > 0:
                for idx, (sx0, sy0, sx1, sy1) in enumerate(self.skill_boxes):
                    name = skills[idx]
                    cid = SKILLS.index(name) if name in SKILLS else 0
                    # shrink box by 1px
                    sx0 += 1;
                    sy0 += 1;
                    sx1 -= 1;
                    sy1 -= 1
                    cx, cy = ((sx0 + sx1) / 2 / iw, (sy0 + sy1) / 2 / ih)
                    w, h = ((sx1 - sx0) / iw, (sy1 - sy0) / ih)
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Write JSON metadata
        data = {
            "filename": fn,
            "drop_count": cnt,
            "xp_values": val.split(";") if val else [],
            "skills": skills,
            "boxes": []
        }
        # drop entries
        for box, _ in self.current_boxes:
            x0, y0, x1, y1 = box
            data["boxes"].append({
                "class": 0,
                "center_x": float((x0 + x1) / 2 / iw),
                "center_y": float((y0 + y1) / 2 / ih),
                "width": float((x1 - x0) / iw),
                "height": float((y1 - y0) / ih)
            })
        # skill entries only if cnt > 0
        if cnt > 0:
            for idx, (sx0, sy0, sx1, sy1) in enumerate(self.skill_boxes):
                name = skills[idx]
                cid = SKILLS.index(name) if name in SKILLS else 0
                sx0 += 1;
                sy0 += 1;
                sx1 -= 1;
                sy1 -= 1
                data["boxes"].append({
                    "class": cid,
                    "center_x": float((sx0 + sx1) / 2 / iw),
                    "center_y": float((sy0 + sy1) / 2 / ih),
                    "width": float((sx1 - sx0) / iw),
                    "height": float((sy1 - sy0) / ih)
                })

        with open(os.path.join(LABELED_DIR, os.path.splitext(fn)[0] + ".json"), 'w') as f:
            json.dump(data, f, indent=2)

        # Finalize
        self.files.pop(self.current_index)
        self.action_history.append({'type': 'save', 'filename': fn, 'idx': self.current_index})
        self.status_label.config(text=f"Saved {fn}", fg='green')

        if self.files:
            self.apply_random_order_logic()
        else:
            self.current_index = 0
        self.load_image()

    def remove_box(self, idx):
        if 0 <= idx < len(self.current_boxes):
            # remove the visual boxes
            del self.current_boxes[idx]
            del self.skill_boxes[idx]

            # **new**: remove the matching CRNN prediction and confidence
            del self.xp_preds[idx]
            del self.xp_confs[idx]

            # redraw & re-populate
            self._draw_image()
            self.load_pred_frame()

            # restore focus & selection
            self.value_entry.focus_set()
            self.value_entry.selection_range(0, 'end')

            # update the drop count spinbox to match
            new_count = len(self.current_boxes)
            self.count_spin.delete(0, 'end')
            self.count_spin.insert(0, str(new_count))
            self._on_count_changed()

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

if __name__ == '__main__':
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

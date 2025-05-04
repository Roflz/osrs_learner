import json
import os
import random
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from config import CROP_DIR, LABELED_DIR, SKILLS, ENTRY_BG, ENTRY_FG, SKIP_DIR, YOLO_MODEL_PATH, CRNN_MODEL_PATH
from io_utils import list_image_files, save_label_data, move_file
from models.yolo_model import YoloModel
from models.crnn_model import get_crnn_predictor
from utils import snap_text, format_stats

class LabelingTab:
    def __init__(self, parent):
        self.parent = parent
        self.files = list_image_files(CROP_DIR)
        self.index = 0
        self.yolo = YoloModel()
        self.crnn = get_crnn_predictor()
        self.build_ui()
        self.load_image()


    def build_ui(self):
        # --- Auto‑fill threshold defaults & misc state flags ---
        self.drop_threshold = 80.0
        self.xp_threshold = 80.0
        self.skill_threshold = 80.0
        self.box_opacity = 255
        self.random_order = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.fill_with_predicted = tk.BooleanVar(value=True)

        # === Directories ===
        os.makedirs(CROP_DIR, exist_ok=True)
        os.makedirs(LABELED_DIR, exist_ok=True)
        os.makedirs(SKIP_DIR, exist_ok=True)

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
        self.parent.rowconfigure(0, weight=1);
        self.parent.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.parent)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind('<Button-1>', self.on_click)
        for key, func in [('<Left>', self.move_skill_left),
                          ('<Right>', self.move_skill_right),
                          ('<Up>', self.move_skill_up),
                          ('<Down>', self.move_skill_down)]:
            self.canvas.bind(key, func)
        self.parent.bind('<Return>', lambda e: self.save_label())
        self.parent.bind('<Control-s>', lambda e: self.skip_image())
        self.parent.bind('<Control-u>', lambda e: self.undo_label())

        # === Controls ===
        ctrl = tk.Frame(self.parent)
        ctrl.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        for c in range(3):
            ctrl.columnconfigure(c, weight=1)

        # Row 0: Remaining & Auto‑Prune
        self.count_label = tk.Label(ctrl, anchor='w')
        self.count_label.grid(row=0, column=0, sticky='w')
        tk.Button(ctrl, text="Auto Prune Images", command=self.auto_prune_images) \
            .grid(row=0, column=2, sticky='e')

        # Row 1: Stats
        self.stats_label = tk.Label(ctrl, anchor='w')
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
                 anchor='center') \
            .grid(row=9, column=0, columnspan=3, sticky='ew', pady=(5, 0))

        # Bind fill‑mode changes
        self.fill_with_predicted.trace_add('write', self._apply_fill_mode)

        # finally: initial load
        if self.files:
            self.load_image()
            self.update_stats()

    def load_image(self):
        # ── Step 0: clear stale XP entry widgets to avoid bad path names ──
        # Destroy any existing widgets in the xp_frame and reset xp_entries
        for w in getattr(self, 'xp_frame', []).winfo_children():
            w.destroy()
        self.xp_entries = []

        # ── Step 1: clear the canvas ──
        self.canvas.delete('all')

        # ── Step 2: check for no files ──
        if not self.files:
            self.count_label.config(text="✅ All images processed!")
            return

        # ── Step 3: reset zoom & pan ──
        self.zoom, self.origin_x, self.origin_y = 1.0, 0, 0

        # ── Step 4: pick the next file ──
        if self.random_order.get():
            self.current_index = random.randint(0, len(self.files) - 1)
        fn = self.files[self.current_index]
        img_path = os.path.join(CROP_DIR, fn)
        img = Image.open(img_path).convert('RGB')
        self.current_img = img
        self.count_label.config(text=f"Remaining: {len(self.files)}")

        # 1) YOLO → pass the PIL.Image directly
        detections = self.yolo.predict(img)
        print(f"[DEBUG] YOLO returned {len(detections)} detections:")
        for d in detections:
            print("   ", d)

        # 2) SPLIT — no thresholds yet, just collect everything
        self.current_boxes.clear()
        raw_skills = []
        for det in detections:
            x0, y0, x1, y1 = det['bbox']
            cls            = det['class_id']
            name           = det.get('name', self.yolo.model.names[cls])
            conf_f         = det['confidence']       # 0–1
            conf_p         = conf_f * 100            # 0–100%

            if cls == 0:
                # drop
                self.current_boxes.append(((x0, y0, x1, y1), conf_p))
            else:
                # skill
                yc = (y0 + y1) / 2
                raw_skills.append({
                    'cls':  cls,
                    'name': name,
                    'conf': conf_f,
                    'y':    yc,
                    'xyxy': (x0, y0, x1, y1)
                })

        print(f"[DEBUG] Built current_boxes: {self.current_boxes}")
        print(f"[DEBUG] Built raw_skills:   {raw_skills}")

        # 3) Pair drops ↔ skills (same as before)
        self.current_boxes.sort(key=lambda t: t[0][1])
        raw_skills.sort(key=lambda r: r['y'])
        # Build lists of drop‐centers and skill‐centers/confidence
        drop_centers = [((y0 + y1) / 2) for ((x0, y0, x1, y1), _) in self.current_boxes]
        skill_centers = [sk['y'] for sk in raw_skills]
        skill_confs = [sk['conf'] for sk in raw_skills]

        if drop_centers and skill_centers:
            # cost = vertical distance / (confidence + ε)
            eps = 1e-6
            cost_matrix = np.zeros((len(drop_centers), len(skill_centers)), dtype=float)
            for i, dy in enumerate(drop_centers):
                for j, sy in enumerate(skill_centers):
                    dist = abs(dy - sy)
                    cost_matrix[i, j] = dist / (skill_confs[j] + eps)

            # solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # threshold for “too far apart” matches (tune as needed)
            MAX_Y_DIST = 20
            paired = [None] * len(drop_centers)
            for i, j in zip(row_ind, col_ind):
                if abs(drop_centers[i] - skill_centers[j]) <= MAX_Y_DIST:
                    paired[i] = raw_skills[j]
        else:
            paired = [None] * len(drop_centers)

        self.raw_skills = raw_skills
        self.paired_skills = paired

        # 4) Build skill‐box rectangles
        self.skill_boxes_raw      = []
        self.skill_boxes_fallback = []
        self.use_raw_skill        = []
        for idx, ((x0, y0, x1, y1), _) in enumerate(self.current_boxes):
            ix1, ix0 = x0 - 2.5, x0 - 28.0
            yc = (y0 + y1 + 3) / 2
            iy0, iy1 = yc - 11.75, yc + 11.75

            fb  = (ix0, iy0, ix1, iy1)
            raw = self.paired_skills[idx]['xyxy'] if self.paired_skills[idx] else None

            self.skill_boxes_fallback.append(fb)
            self.skill_boxes_raw.append(raw)
            self.use_raw_skill.append(bool(raw))

        self.skill_boxes = [
            (raw if use else fb)
            for raw, fb, use in zip(
                self.skill_boxes_raw,
                self.skill_boxes_fallback,
                self.use_raw_skill
            )
        ]

        # 5) Compute drop‐count
        if (self.current_boxes and
            all(conf >= self.drop_threshold for (_, conf) in self.current_boxes)):
            cnt = len(self.current_boxes)
        else:
            cnt = 0
        self.count_spin.delete(0, 'end')
        self.count_spin.insert(0, str(cnt))
        self.last_count = cnt
        self._on_count_changed()

        # 6) CRNN on every drop‐box
        box_coords     = [box for (box, _) in self.current_boxes]
        self.crnn_preds = self.crnn.predict_from_boxes(self.current_img, box_coords)
        self.xp_preds   = [t for (t, _) in self.crnn_preds]
        self.xp_confs   = [c for (_, c) in self.crnn_preds]

        # 7) Auto‑fill or use saved defaults
        self._apply_fill_mode()

        # ── Step 6: redraw & update the predictions panel ──
        self._draw_image()
        self.load_pred_frame()

        # ── Step 7: safely focus the first XP entry ──
        if self.xp_entries:
            try:
                self.xp_entries[0].focus_set()
                self.xp_entries[0].selection_range(0, 'end')
            except tk.TclError:
                # if that widget was destroyed/recreated in the interim, just skip
                pass

    def _draw_image(self):
        if not self.current_img:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.current_img.size
        base   = min(cw/iw, ch/ih)
        sc     = base * self.zoom
        nw, nh = int(iw*sc), int(ih*sc)
        if nw <= 0 or nh <= 0:
            return

        img2    = self.current_img.resize((nw,nh), Image.LANCZOS).convert("RGBA")
        overlay = Image.new("RGBA", img2.size, (0,0,0,0))
        draw    = ImageDraw.Draw(overlay)
        font    = ImageFont.load_default()
        big_f   = ImageFont.truetype("arial.ttf", 20)
        bw      = max(1, int(self.zoom))

        # draw drops
        for i, ((x0,y0,x1,y1), drop_conf) in enumerate(self.current_boxes):
            sx0, sy0, sx1, sy1 = [v*sc for v in (x0,y0,x1,y1)]
            if self.show_boxes.get():
                col = (0,255,0,255) if drop_conf>=80 else (255,165,0,255) if drop_conf>=70 else (255,0,0,255)
                draw.rectangle([sx0,sy0,sx1,sy1], outline=col, width=bw)
                draw.text((sx0, max(0,sy0-12)), f"{drop_conf:.0f}%", fill=col, font=font)

            # draw XP
            xp, xp_c = self.xp_preds[i], self.xp_confs[i]
            xp_col = (0,255,0,255) if xp_c>=self.xp_threshold else (255,165,0,255) if xp_c>=self.xp_threshold-10 else (255,0,0,255)
            sb = self.skill_boxes[i]
            sk_x0, sk_y0, *_ = [v*sc for v in sb]
            bb_xp = draw.textbbox((0,0), xp, font=big_f)
            tw, th = bb_xp[2]-bb_xp[0], bb_xp[3]-bb_xp[1]
            draw.text((sk_x0-5-tw, sk_y0), xp, fill=xp_col, font=big_f)

            # draw Skill label
            rec = self.paired_skills[i]
            if rec:
                sk_name = rec['name']
                sk_pct  = rec['conf']*100
                sk_col  = (0,255,0,255) if sk_pct>=self.skill_threshold else (255,165,0,255) if sk_pct>=self.skill_threshold-10 else (255,0,0,255)
            else:
                sk_name, sk_pct, sk_col = "N/A", 0, (255,0,0,255)
            bb_sk = draw.textbbox((0,0), sk_name, font=big_f)
            tw2 = bb_sk[2]-bb_sk[0]
            draw.text((sk_x0-5-tw2, sk_y0+th), sk_name, fill=sk_col, font=big_f)

            # draw skill/fallback boxes
            if self.show_boxes.get():
                for j,use_raw in enumerate(self.use_raw_skill):
                    bx = self.skill_boxes_raw[j] if use_raw and self.skill_boxes_raw[j] else self.skill_boxes_fallback[j]
                    ssx0, ssy0, ssx1, ssy1 = [v*sc for v in bx]
                    outline = (255,0,255,self.box_opacity) if use_raw else (0,255,255,self.box_opacity)
                    draw.rectangle([ssx0,ssy0,ssx1,ssy1], outline=outline, width=bw)

        comp      = Image.alpha_composite(img2, overlay).convert("RGB")
        self.tkimg = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.origin_x + cw//2,
            self.origin_y + ch//2,
            image=self.tkimg,
            anchor='center'
        )

    def load_pred_frame(self):
        # clear old
        for w in self.pred_frame.winfo_children():
            w.destroy()

        # debug log
        print("[DEBUG] ===== Per-Drop Summary =====")
        for i, ((x0,y0,x1,y1), drop_conf) in enumerate(self.current_boxes, start=1):
            xp, xp_c = self.xp_preds[i-1], self.xp_confs[i-1]
            rec = self.paired_skills[i-1]
            sk_name = rec['name'] if rec else "N/A"
            sk_pct  = (rec['conf']*100) if rec else 0
            print(f"  Drop {i}: Xp={xp} ({xp_c:.0f}%) | DropConf={drop_conf:.1f}% | Skill={sk_name} ({sk_pct:.0f}%)")

        # render rows
        for i, ((x0,y0,x1,y1), drop_conf) in enumerate(self.current_boxes):
            rec = self.paired_skills[i]
            sk_name = rec['name'] if rec else "N/A"
            sk_pct  = (rec['conf']*100) if rec else 0

            fr = tk.Frame(self.pred_frame)
            fr.pack(anchor='center', pady=2)

            # XP
            xp, xp_c = self.xp_preds[i], self.xp_confs[i]
            xp_col = 'lime' if xp_c>=self.xp_threshold else 'orange' if xp_c>=self.xp_threshold-10 else 'red'
            tk.Label(fr, text="Xp:", font=('Helvetica',12)).pack(side='left')
            tk.Label(fr, text=str(xp), fg=xp_col, font=('Helvetica',15,'bold')).pack(side='left')
            tk.Label(fr, text=f"({xp_c:.1f}%)", font=('Helvetica',12)).pack(side='left', padx=(0,10))

            # Skill
            sk_col = 'lime' if sk_pct>=self.skill_threshold else 'orange' if sk_pct>=self.skill_threshold-10 else 'red'
            tk.Label(fr, text="Skill:", font=('Helvetica',12)).pack(side='left')
            tk.Label(fr, text=sk_name, fg=sk_col, font=('Helvetica',15,'bold')).pack(side='left')
            tk.Label(fr, text=f"({sk_pct:.1f}%)", font=('Helvetica',12)).pack(side='left', padx=(0,10))

            # Drop conf
            dc_col = 'lime' if drop_conf>=self.drop_threshold else 'orange' if drop_conf>=self.drop_threshold-10 else 'red'
            tk.Label(fr, text="Drop:", font=('Helvetica',12)).pack(side='left')
            tk.Label(fr, text=f"{drop_conf:.1f}%", fg=dc_col, font=('Helvetica',15,'bold')).pack(side='left', padx=(0,10))

            # Remove & toggle
            tk.Button(fr, text="Remove", fg='red', command=lambda idx=i: self.remove_box(idx)).pack(side='left', padx=(0,10))
            var = tk.BooleanVar(value=self.use_raw_skill[i])
            tk.Checkbutton(fr, text="Use Predicted", variable=var,
                           command=lambda idx=i,v=var: self._toggle_skill(idx,v)
                           ).pack(side='left')

    def save_label(self):
        if not self.files:
            return

        fn = self.files[self.current_index]
        cnt = self.last_count

        # gather inputs
        xp_vals = [e.get().strip() for e in self.xp_entries] if cnt > 0 else []
        skills  = [cb.get().strip() for cb in self.skill_selectors] if cnt > 0 else []

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

        # log action for undo
        self.action_history.append({'type': 'save', 'filename': fn, 'idx': self.current_index})

        # move & write files
        src = os.path.join(CROP_DIR, fn)
        dst = os.path.join(LABELED_DIR, fn)
        shutil.move(src, dst)

        # open for size
        img = Image.open(dst)
        iw, ih = img.size
        base, _ = os.path.splitext(fn)

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
            self._apply_fill_mode()

            if self.files:
                self.apply_random_order_logic()
            else:
                self.current_index = 0
            self.load_image()

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
        if not self.files:
            return

        idx = self.current_index
        fn = self.files.pop(self.current_index)

        # log action for undo
        self.action_history.append({'type': 'skip', 'filename': fn, 'idx': idx})

        # move file to skip dir
        shutil.move(os.path.join(CROP_DIR, fn), os.path.join(SKIP_DIR, fn))

        self.status_label.config(text=f"Skipped {fn}", fg='orange')
        if self.files:
            self.apply_random_order_logic()
        else:
            self.current_index = 0
        self.load_image()

    def undo_label(self):
        """
        Undo the last save or skip action. Moves file back into CROP_DIR and reloads it.
        """
        # Debug: what’s in the action history?
        print("[DEBUG] action_history:", self.action_history)

        if not getattr(self, 'action_history', None):
            self.status_label.config(text="Nothing to undo.", fg='gray')
            return

        act = self.action_history.pop()
        print("[DEBUG] Popped action:", act)

        fn = act.get('filename')
        idx = act.get('idx', self.current_index)

        action_type = act.get('type')
        if action_type == 'save':
            src = os.path.join(LABELED_DIR, fn)
            dst = os.path.join(CROP_DIR, fn)
        elif action_type == 'skip':
            src = os.path.join(SKIP_DIR, fn)
            dst = os.path.join(CROP_DIR, fn)
        else:
            self.status_label.config(text=f"Unknown action '{action_type}'", fg='red')
            return

        print(f"[DEBUG] Moving {src} -> {dst}")
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"[DEBUG] Move failed:", e)
            self.status_label.config(text=f"Undo failed: {e}", fg='red')
            return

        # Reinsert filename at original index
        self.files.insert(idx, fn)
        self.current_index = idx
        print("[DEBUG] New files list:", self.files)

        # Refresh UI
        self.status_label.config(text=f"Undid {action_type} of {fn}", fg='blue')
        self.load_image()
        # If you track stats separately, call update_stats
        if hasattr(self, 'update_stats'):
            self.update_stats()

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

    def _on_canvas_resize(self, e):
        self._draw_image()

    def auto_prune_images(self):
        files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith('.png'))
        for fn in tqdm(files, desc="Auto-pruning"):
            img = Image.open(os.path.join(CROP_DIR, fn)).convert('RGB')
            res = self.yolo.predict(img, verbose=False)[0]
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
        name = self.skill_selectors[idx].get()
        # ensure our history list is long enough
        if idx < len(self.last_nonzero_skills):
            self.last_nonzero_skills[idx] = name
        else:
            # extend with empty strings up to idx, then set
            extension = [''] * (idx - len(self.last_nonzero_skills) + 1)
            self.last_nonzero_skills.extend(extension)
            self.last_nonzero_skills[idx] = name

        # persist defaults for undo/fill-mode
        self.defaults[self.last_count] = (self.last_value, self.last_nonzero_skills.copy())

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

    def update_stats(self):
        total = len(self.labels)
        rem = len(self.files)
        drops = sum(c for *_, c in self.labels.values())
        self.stats_label.config(
            text=f"Labeled: {total} — Remaining: {rem} — Auto-skipped: {self.auto_skipped} | Total drops: {drops}"
        )

    def _on_skill_type(self, event, idx):
        w = event.widget;
        typed = w.get()
        for s in SKILLS:
            if s.lower().startswith(typed.lower()):
                w.delete(0, 'end');
                w.insert(0, s);
                w.select_range(len(typed), 'end')
                w.icursor(len(s));
                break
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
                    name = self.yolo.model.names[rec['cls']]
                    self.skill_selectors[i].set(name)
            self.last_nonzero_skills = [cb.get() for cb in self.skill_selectors]
            self.defaults[cnt] = (self.last_value, self.last_nonzero_skills.copy())

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
                        self.skill_selectors[i].set(self.yolo.model.names[rec['cls']])
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
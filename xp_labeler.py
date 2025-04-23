#!/usr/bin/env python3
import os
import csv
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Grayscale, ToTensor

# ─── Paths & Constants ───────────────────────────────────────────────────────
CROP_DIR       = "data/xp_crops"
LABELED_DIR    = "data/xp_crops_labeled"
SKIP_DIR       = "data/xp_crops_skipped"
LABEL_CSV      = os.path.join("data", "xp_labels.csv")
REGRESSOR_PTH  = os.path.join("models", "xp_regressor.pth")
CLASSIFIER_PTH = os.path.join("models", "skill_classifier.pth")
DETECTOR_PTH   = os.path.join("models", "xp_detector.pth")

SKILLS = [
    "None",
    "woodcutting", "mining", "fishing", "cooking", "firemaking",
    "fletching", "thieving", "agility", "herblore", "crafting",
    "smithing", "runecrafting", "slayer", "farming", "construction",
    "hunter"
]

# ─── Model Definitions ───────────────────────────────────────────────────────
class XPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x): return self.net(x).squeeze(1)

class SkillClassifier(nn.Module):
    def __init__(self, n_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,n_skills)
        )
    def forward(self,x): return self.net(x)

class XPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).squeeze(1)

# ─── Labeler GUI ─────────────────────────────────────────────────────────────
class XPLabeler:
    def __init__(self, root):
        self.root = root
        root.title("XP Crop Labeler")

        # ensure directories
        os.makedirs(CROP_DIR, exist_ok=True)
        os.makedirs(LABELED_DIR, exist_ok=True)
        os.makedirs(SKIP_DIR,    exist_ok=True)

        # histories and default actions
        self.save_history = []
        self.skip_history = []
        self.last_button_action = self.save_label
        root.bind('<Return>', lambda e: self.last_button_action())

        # remember last inputs
        self.last_skill = "None"
        self.last_xp    = ""
        self.last_drop  = False

        # load existing labels
        self.labels = {}
        if os.path.exists(LABEL_CSV):
            with open(LABEL_CSV, newline='') as f:
                for row in csv.DictReader(f):
                    self.labels[row["filename"]] = (
                        row["value"], row["skill"], row["drop"]
                    )

        # pending files
        all_files = sorted(fn for fn in os.listdir(CROP_DIR)
                           if fn.lower().endswith(".png"))
        self.files = [fn for fn in all_files if fn not in self.labels]
        self.current_index = 0

        # ─── UI setup ────────────────────────────────────────────────────
        self.img_label = tk.Label(root)
        self.img_label.pack(padx=5, pady=5)

        self.count_label = tk.Label(root, anchor="w")
        self.count_label.pack(fill="x", padx=5)
        self.stats_label = tk.Label(root, anchor="w")
        self.stats_label.pack(fill="x", padx=5)

        frm = tk.Frame(root)
        frm.pack(fill="x", padx=5, pady=5)
        tk.Label(frm, text="XP amount:").grid(row=0, column=0, sticky="e")
        self.xp_entry = tk.Entry(frm, width=10)
        self.xp_entry.grid(row=0, column=1, padx=(2,10))
        self.xp_entry.bind("<KeyRelease>", self._on_xp_changed)

        tk.Label(frm, text="Skill:").grid(row=0, column=2, sticky="e")
        self.skill_combo = ttk.Combobox(frm, values=SKILLS, state="readonly")
        self.skill_combo.grid(row=0, column=3, padx=(0,10))
        self.skill_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: setattr(self, 'last_skill', self.skill_combo.get())
        )

        tk.Label(frm, text="Drop?").grid(row=0, column=4, sticky="e")
        self.drop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm, variable=self.drop_var).grid(row=0, column=5, sticky="w")

        nav = tk.Frame(root)
        nav.pack(fill="x", pady=5)
        tk.Button(nav, text="Prev",
                  command=self._track(self.prev_image)).pack(side="left")
        tk.Button(nav, text="Skip",
                  command=self._track(self.skip_image)).pack(side="left", padx=5)
        self.save_btn = tk.Button(nav, text="Save Label",
                                  command=self._track(self.save_label))
        self.save_btn.pack(side="left", padx=5)
        tk.Button(nav, text="Undo",
                  command=self._track(self.undo_label)).pack(side="left", padx=5)
        tk.Button(nav, text="Export CSV",
                  command=self._track(self.export_csv)).pack(side="right")

        self.status_label = tk.Label(root, anchor="w")
        self.status_label.pack(fill="x", padx=5, pady=(0,5))
        self.pred_label = tk.Label(root, fg="blue", anchor="w")
        self.pred_label.pack(fill="x", padx=5, pady=(0,5))

        # load ML models and transform
        self.device = torch.device("cpu")
        self._load_models()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            Grayscale(num_output_channels=1),
            ToTensor(),
        ])

        # initial display
        if not self.files:
            self.status_label.config(text="No new crops to label.", fg="blue")
        else:
            self.load_image()
            self.update_stats()

    def _track(self, fn):
        def wrapped():
            self.last_button_action = fn
            fn()
        return wrapped

    def _load_models(self):
        try:
            m = XPRegressor().to(self.device)
            m.load_state_dict(torch.load(REGRESSOR_PTH, map_location=self.device))
            m.eval()
            self.xp_model = m
        except:
            self.xp_model = None
        try:
            m = SkillClassifier(len(SKILLS)).to(self.device)
            m.load_state_dict(torch.load(CLASSIFIER_PTH, map_location=self.device))
            m.eval()
            self.skill_model = m
        except:
            self.skill_model = None
        try:
            m = XPDetector().to(self.device)
            m.load_state_dict(torch.load(DETECTOR_PTH, map_location=self.device))
            m.eval()
            self.detector = m
        except:
            self.detector = None

    def load_image(self):
        if not self.files:
            self.img_label.config(image="", text="(no images)")
            self.count_label.config(text="Remaining: 0")
            return

        fname = self.files[self.current_index]
        img = Image.open(os.path.join(CROP_DIR, fname))
        img = img.resize((300, int(300 * img.height / img.width)), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tkimg)

        self.count_label.config(text=f"Remaining: {len(self.files)}")

        # restore or default inputs
        if fname in self.labels:
            val, skl, drop = self.labels[fname]
            self.xp_entry.delete(0, tk.END)
            self.xp_entry.insert(0, val)
            self.skill_combo.set(skl)
            self.drop_var.set(drop == "yes")
            self.last_xp    = val
            self.last_skill = skl
            self.last_drop  = (drop == "yes")
        else:
            self.xp_entry.delete(0, tk.END)
            self.xp_entry.insert(0, self.last_xp)
            self.skill_combo.set(self.last_skill)
            self.drop_var.set(self.last_drop)

        self._update_prediction(fname)
        self.update_stats()
        self.save_btn.focus_set()

    def _on_xp_changed(self, event):
        text = self.xp_entry.get().strip()
        if text:
            self.drop_var.set(True)
        self.last_xp = text

    def _update_prediction(self, fname):
        path = os.path.join(CROP_DIR, fname)
        img = Image.open(path).convert("RGB")
        inp = self.transform(img).unsqueeze(0).to(self.device)
        parts = []
        if self.xp_model:
            parts.append(f"XP≈{self.xp_model(inp).item():.1f}")
        if self.skill_model:
            logits = self.skill_model(inp)
            probs = torch.softmax(logits, 1)
            p, idx = probs.max(1)
            skl = SKILLS[idx]
            parts.append(f"Skill={skl} ({p*100:.0f}%)")
        if self.detector:
            dp = self.detector(inp).item()
            parts.append(f"Drop={'Yes' if dp>0.5 else 'No'} ({dp*100:.0f}%)")
        self.pred_label.config(text=" ▶ ".join(parts))

    def prev_image(self):
        if not self.files:
            return
        self.current_index = max(0, self.current_index - 1)
        self.status_label.config(text="")
        self.load_image()

    def skip_image(self):
        if not self.files:
            return
        idx = self.current_index
        fname = self.files.pop(idx)
        shutil.move(os.path.join(CROP_DIR, fname),
                    os.path.join(SKIP_DIR, fname))
        self.skip_history.append((fname, idx))
        if idx >= len(self.files):
            self.current_index = max(0, len(self.files) - 1)
        self.status_label.config(text=f"Skipped {fname}", fg="orange")
        self.load_image()

    def save_label(self):
        if not self.files:
            return
        fname = self.files[self.current_index]
        drop  = self.drop_var.get()
        xp_val = self.xp_entry.get().strip()
        skl    = self.skill_combo.get().strip()

        # validation rules:
        if xp_val and skl == "None":
            self.status_label.config(
                text="Cannot save: XP entered with no skill selected.",
                fg="red"
            )
            return
        if (xp_val or skl != "None") and not drop:
            self.status_label.config(
                text="Cannot save: Drop must be checked if XP or skill given.",
                fg="red"
            )
            return

        # record
        self.save_history.append((fname, xp_val, skl, "yes" if drop else "no"))
        self.labels[fname] = (xp_val, skl, "yes" if drop else "no")

        # update memories
        self.last_xp    = xp_val
        self.last_skill = skl
        self.last_drop  = drop

        shutil.move(os.path.join(CROP_DIR, fname),
                    os.path.join(LABELED_DIR, fname))
        del self.files[self.current_index]
        if self.current_index >= len(self.files):
            self.current_index = max(0, len(self.files) - 1)
        self.status_label.config(text=f"Saved {fname}", fg="green")
        self.load_image()

    def undo_label(self):
        if self.save_history:
            fname, xp_val, skl, drop = self.save_history.pop()
            self.labels.pop(fname, None)
            shutil.move(os.path.join(LABELED_DIR, fname),
                        os.path.join(CROP_DIR, fname))
            self.files.insert(self.current_index+1, fname)
            self.current_index = self.files.index(fname)
            self.load_image()
            self.xp_entry.delete(0, tk.END)
            self.xp_entry.insert(0, xp_val)
            self.skill_combo.set(skl)
            self.drop_var.set(drop == "yes")
            self.status_label.config(text=f"Undid save {fname}", fg="blue")
            return

        if self.skip_history:
            fname, idx = self.skip_history.pop()
            shutil.move(os.path.join(SKIP_DIR, fname),
                        os.path.join(CROP_DIR, fname))
            self.files.insert(idx, fname)
            self.current_index = idx
            self.load_image()
            self.status_label.config(text=f"Undid skip {fname}", fg="blue")
            return

        self.status_label.config(text="Nothing to undo.", fg="orange")

    def export_csv(self):
        if not self.labels:
            messagebox.showinfo("Export CSV", "No labels to export.")
            return
        with open(LABEL_CSV, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["filename", "value", "skill", "drop"])
            for fn, (v, s, d) in sorted(self.labels.items()):
                if d.lower() != "yes":
                    # blank out value & skill for no-drop
                    w.writerow([fn, "", "", "no"])
                else:
                    w.writerow([fn, v, s, "yes"])
        self.status_label.config(
            text=f"Exported {len(self.labels)} labels", fg="darkgreen"
        )

    def update_stats(self):
        total     = len(self.labels)
        remaining = len(self.files)
        yes_count = sum(1 for *_, d in self.labels.values() if d == "yes")
        no_count  = total - yes_count
        self.stats_label.config(
            text=f"Labeled: {total} (Yes: {yes_count}, No: {no_count}) — Remaining: {remaining}"
        )

if __name__ == "__main__":
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

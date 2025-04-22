#!/usr/bin/env python3
import os
import csv
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import transforms as T

# ─── Configuration ──────────────────────────────────────────────────────────

CROP_DIR       = os.path.join("data", "xp_crops")
LABELED_DIR    = os.path.join("data", "xp_crops_labeled")
LABEL_CSV      = os.path.join("data", "xp_labels.csv")
REGRESSOR_PTH  = os.path.join("models", "xp_regressor.pth")
CLASSIFIER_PTH = os.path.join("models", "skill_classifier.pth")
DETECTOR_PTH = os.path.join("models", "xp_detector.pth")  # your trained detector checkpoint

SKILLS = [
    "woodcutting", "mining", "fishing", "cooking", "firemaking",
    "fletching", "thieving", "agility", "herblore", "crafting",
    "smithing", "runecrafting", "slayer", "farming", "construction",
    "hunter"
]

# ─── Model Definitions ─────────────────────────────────────────────────────

class XPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32→16
            nn.Flatten(),
            # flatten size = 32 channels × 16 × 16 = 8192
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class SkillClassifier(nn.Module):
    def __init__(self, n_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32→16
            nn.Flatten(),
            # flatten size = 32 × 16 × 16 = 8192
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_skills)
        )

    def forward(self, x):
        return self.net(x)

class XPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # conv layers: 64×64 → 32×32 → 16×16
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            # flatten dim = 32 channels × 16 × 16 = 8192
            nn.Linear(32 * 16 * 16, 64),  # hidden size = 64
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1),             # single‑logit output
            nn.Sigmoid()                  # map to [0,1]
        )

    def forward(self, x):
        return self.net(x).squeeze(1)




# ─── Labeler App ──────────────────────────────────────────────────────────

class XPLabeler:
    def __init__(self, root):
        self.root = root
        root.title("XP Crop Labeler")

        # ensure directories
        os.makedirs(CROP_DIR, exist_ok=True)
        os.makedirs(LABELED_DIR, exist_ok=True)

        # load existing labels
        self.labels = {}
        if os.path.exists(LABEL_CSV):
            with open(LABEL_CSV, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels[row["filename"]] = (row["value"], row["skill"])

        # undo history
        self.save_history = []

        # build file list (skip labeled)
        all_files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith(".png"))
        self.files = [fn for fn in all_files if fn not in self.labels]
        self.current_index = 0

        # ─── UI ────────────────────────────────────────────────────────────

        # image display
        self.img_label = tk.Label(root)
        self.img_label.pack(padx=5, pady=5)

        # form: XP + skill
        frm = tk.Frame(root)
        frm.pack(fill="x", padx=5)
        tk.Label(frm, text="XP amount:").grid(row=0, column=0, sticky="e")
        self.xp_entry = tk.Entry(frm, width=10)
        self.xp_entry.grid(row=0, column=1, padx=(2,10))
        tk.Label(frm, text="Skill:").grid(row=0, column=2, sticky="e")
        self.skill_combo = ttk.Combobox(frm, values=SKILLS, state="readonly")
        self.skill_combo.grid(row=0, column=3)

        # navigation buttons
        nav = tk.Frame(root)
        nav.pack(fill="x", pady=5)
        tk.Button(nav, text="Prev",       command=self.prev_image).pack(side="left")
        tk.Button(nav, text="Next",       command=self.next_image).pack(side="left")
        tk.Button(nav, text="Save Label", command=self.save_label).pack(side="left", padx=5)
        tk.Button(nav, text="Undo",       command=self.undo_label).pack(side="left", padx=5)
        tk.Button(nav, text="Export CSV", command=self.export_csv).pack(side="right")

        # status and prediction labels
        self.status_label = tk.Label(root, text="", anchor="w")
        self.status_label.pack(fill="x", padx=5, pady=(0,5))
        self.pred_label   = tk.Label(root, text="", anchor="w", fg="blue")
        self.pred_label.pack(fill="x", padx=5, pady=(0,5))

        # ─── Models & Transforms ─────────────────────────────────────────

        self.device = torch.device("cpu")
        self._load_models()

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((64, 64)),   # ← must match training
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        # initial display
        if not self.files:
            self.status_label.config(text="No new crops to label.", fg="blue")
        else:
            self.load_image()

    def _load_models(self):
        # ─── XP Regressor ───────────────────────────────────────────────────
        try:
            self.xp_model = XPRegressor().to(self.device)
            chk = torch.load(REGRESSOR_PTH, map_location=self.device)
            self.xp_model.load_state_dict(chk, strict=False)
            self.xp_model.eval()
        except Exception as e:
            messagebox.showwarning("Model Load", f"XP regressor not loaded:\n{e}")
            self.xp_model = None

        # ─── Skill Classifier (dynamic head size) ────────────────────────────
        try:
            chk = torch.load(CLASSIFIER_PTH, map_location=self.device)
            # find the final linear weight in the state dict
            head_key = next(k for k in chk.keys() if k.endswith("net.10.weight"))
            n_out = chk[head_key].shape[0]
            # build a classifier with exactly that many outputs
            self.skill_model = SkillClassifier(n_out).to(self.device)
            self.skill_model.load_state_dict(chk, strict=False)
            self.skill_model.eval()
            # map indexes back to your skill names (slice your SKILLS list)
            self.skill_names = SKILLS[:n_out]
        except Exception as e:
            messagebox.showwarning("Model Load", f"Skill classifier not loaded:\n{e}")
            self.skill_model = None
            self.skill_names = []

        # ─── XP Presence Detector ───────────────────────────────────────────────
        try:
            self.detector = XPDetector().to(self.device)
            chk = torch.load(DETECTOR_PTH, map_location=self.device)
            self.detector.load_state_dict(chk, strict=False)
            self.detector.eval()
        except Exception as e:
            messagebox.showwarning("Model Load", f"XP detector not loaded:\n{e}")
            self.detector = None

    def load_image(self):
        if not self.files:
            self.img_label.config(image="", text="(no images)")
            self.pred_label.config(text="")
            return

        fname = self.files[self.current_index]
        path = os.path.join(CROP_DIR, fname)
        img = Image.open(path)
        img = img.resize((300, int(300 * img.height / img.width)), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tkimg)

        # restore fields if labeled
        if fname in self.labels:
            val, skl = self.labels[fname]
            self.xp_entry.delete(0, tk.END); self.xp_entry.insert(0, val)
            self.skill_combo.set(skl)

        # update model prediction
        self._update_prediction(fname)

    def _update_prediction(self, fname):
        path = os.path.join(CROP_DIR, fname)
        img = Image.open(path).convert("RGB")
        inp = self.transform(img).unsqueeze(0).to(self.device)

        parts = []
        # XP regressor
        if self.xp_model:
            xp_pred = self.xp_model(inp).item()
            parts.append(f"XP≈{xp_pred:.1f}")
        # Skill classifier
        if self.skill_model:
            logits = self.skill_model(inp)
            probs = torch.softmax(logits, dim=1)
            top_p, top_i = probs.max(1)
            idx = top_i.item()
            name = self.skill_names[idx] if idx < len(self.skill_names) else f"class_{idx}"
            conf = top_p.item() * 100
            parts.append(f"Skill={name} ({conf:.0f}%)")
        # XP-drop detector
        if self.detector:
            drop_p = self.detector(inp).item()
            verdict = "Yes" if drop_p > 0.5 else "No"
            parts.append(f"Drop={verdict} ({drop_p * 100:.0f}%)")

        self.pred_label.config(text=" ▶ ".join(parts))

    def prev_image(self):
        if not self.files: return
        self.current_index = max(0, self.current_index - 1)
        self.status_label.config(text="")
        self.load_image()

    def next_image(self):
        if not self.files: return
        self.current_index = min(len(self.files) - 1, self.current_index + 1)
        self.status_label.config(text="")
        self.load_image()

    def save_label(self):
        if not self.files: return
        fname  = self.files[self.current_index]
        xp_val = self.xp_entry.get().strip()
        skl    = self.skill_combo.get().strip()
        if not xp_val or not skl:
            self.status_label.config(text="Enter both XP and skill first.", fg="red")
            return

        self.save_history.append((fname, xp_val, skl))
        self.labels[fname] = (xp_val, skl)
        shutil.move(os.path.join(CROP_DIR,   fname),
                    os.path.join(LABELED_DIR, fname))
        del self.files[self.current_index]
        if self.current_index >= len(self.files):
            self.current_index = len(self.files) - 1

        self.status_label.config(text=f"Saved {fname}", fg="green")
        self.load_image()

    def undo_label(self):
        if not self.save_history:
            self.status_label.config(text="Nothing to undo.", fg="orange")
            return

        fname, xp_val, skl = self.save_history.pop()
        self.labels.pop(fname, None)
        shutil.move(os.path.join(LABELED_DIR, fname),
                    os.path.join(CROP_DIR,    fname))

        self.files.insert(self.current_index + 1, fname)
        self.current_index = self.files.index(fname)
        self.load_image()
        self.xp_entry.delete(0, tk.END); self.xp_entry.insert(0, xp_val)
        self.skill_combo.set(skl)
        self.status_label.config(text=f"Undid {fname}", fg="blue")

    def export_csv(self):
        if not self.labels:
            messagebox.showinfo("Export CSV", "No labels to export.")
            return
        with open(LABEL_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "value", "skill"])
            for fn, (val, skl) in sorted(self.labels.items()):
                w.writerow([fn, val, skl])
        self.status_label.config(text=f"Exported {len(self.labels)} labels", fg="darkgreen")


if __name__ == "__main__":
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

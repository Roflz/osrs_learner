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
from collections import Counter

# ─── Configuration ──────────────────────────────────────────────────────────

CROP_DIR       = os.path.join("data", "xp_crops")
LABELED_DIR    = os.path.join("data", "xp_crops_labeled")
LABEL_CSV      = os.path.join("data", "xp_labels.csv")
REGRESSOR_PTH  = os.path.join("models", "xp_regressor.pth")
CLASSIFIER_PTH = os.path.join("models", "skill_classifier.pth")
DETECTOR_PTH   = os.path.join("models", "xp_detector.pth")

SKILLS = [
    "None",         # explicit “no skill” option for no-drop cases
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
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

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
    def forward(self, x):
        return self.net(x)

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
    def forward(self, x):
        return self.net(x).squeeze(1)


# ─── Labeler App ──────────────────────────────────────────────────────────

class XPLabeler:
    def __init__(self, root):
        self.root = root
        root.title("XP Crop Labeler")

        os.makedirs(CROP_DIR, exist_ok=True)
        os.makedirs(LABELED_DIR, exist_ok=True)

        # load existing labels
        self.labels = {}
        if os.path.exists(LABEL_CSV):
            with open(LABEL_CSV, newline='') as f:
                for row in csv.DictReader(f):
                    drop = row.get("drop","yes")
                    self.labels[row["filename"]] = (row["value"], row["skill"], drop)

        # histories
        self.save_history = []
        self.eval_history = []

        # files to label
        all_files = sorted(fn for fn in os.listdir(CROP_DIR) if fn.lower().endswith(".png"))
        self.files = [fn for fn in all_files if fn not in self.labels]
        self.current_index = 0

        # ─── UI ────────────────────────────────────────────────────────────

        # image
        self.img_label = tk.Label(root)
        self.img_label.pack(padx=5,pady=5)

        # stats
        self.count_label  = tk.Label(root, text="", anchor="w")
        self.count_label.pack(fill="x", padx=5, pady=(0,5))
        self.stats_label  = tk.Label(root, text="", anchor="w")
        self.stats_label.pack(fill="x", padx=5, pady=(0,5))
        self.stats2_label = tk.Label(root, text="", anchor="w", justify="left")
        self.stats2_label.pack(fill="x", padx=5, pady=(0,5))
        self.stats3_label = tk.Label(root, text="", anchor="w", justify="left")
        self.stats3_label.pack(fill="x", padx=5, pady=(0,10))

        # form
        frm = tk.Frame(root)
        frm.pack(fill="x", padx=5)
        tk.Label(frm, text="XP amount:").grid(row=0,column=0,sticky="e")
        self.xp_entry = tk.Entry(frm,width=10)
        self.xp_entry.grid(row=0,column=1,padx=(2,10))
        self.xp_entry.bind("<KeyRelease>", self._on_xp_changed)
        tk.Label(frm, text="Skill:").grid(row=0,column=2,sticky="e")
        self.skill_combo = ttk.Combobox(frm, values=SKILLS, state="readonly")
        self.skill_combo.grid(row=0,column=3,padx=(0,10))
        tk.Label(frm, text="Drop?").grid(row=0,column=4,sticky="e")
        self.drop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm, variable=self.drop_var).grid(row=0,column=5,sticky="w")

        # navigation
        nav = tk.Frame(root)
        nav.pack(fill="x",pady=5)
        tk.Button(nav, text="Prev",       command=self.prev_image).pack(side="left")
        tk.Button(nav, text="Next",       command=self.next_image).pack(side="left")
        tk.Button(nav, text="Save Label", command=self.save_label).pack(side="left",padx=5)
        tk.Button(nav, text="Undo",       command=self.undo_label).pack(side="left",padx=5)
        tk.Button(nav, text="Export CSV", command=self.export_csv).pack(side="right")

        # status & preds
        self.status_label = tk.Label(root, text="", anchor="w")
        self.status_label.pack(fill="x", padx=5, pady=(0,5))
        self.pred_label   = tk.Label(root, text="", anchor="w", fg="blue")
        self.pred_label.pack(fill="x", padx=5, pady=(0,5))

        # models & transforms
        self.device = torch.device("cpu")
        self._load_models()
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((64,64)),
            T.ToTensor(),
            T.Normalize((0.5,),(0.5,))
        ])

        # start
        if not self.files:
            self.status_label.config(text="No new crops to label.", fg="blue")
        else:
            self.load_image()
            self.update_stats()


    def _load_models(self):
        try:
            self.xp_model = XPRegressor().to(self.device)
            chk = torch.load(REGRESSOR_PTH, map_location=self.device)
            self.xp_model.load_state_dict(chk, strict=False)
            self.xp_model.eval()
        except Exception as e:
            messagebox.showwarning("Model Load", f"XP regressor not loaded:\n{e}")
            self.xp_model = None

        try:
            chk = torch.load(CLASSIFIER_PTH, map_location=self.device)
            head_key = next(k for k in chk if k.endswith("net.10.weight"))
            n_cls    = chk[head_key].shape[0]
            self.skill_model = SkillClassifier(n_cls).to(self.device)
            self.skill_model.load_state_dict(chk, strict=False)
            self.skill_model.eval()
            self.skill_names = SKILLS[:n_cls]
        except Exception as e:
            messagebox.showwarning("Model Load", f"Skill classifier not loaded:\n{e}")
            self.skill_model = None
            self.skill_names = []

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
            self.count_label.config(text="Images remaining: 0")
            self.update_stats()
            return

        fname = self.files[self.current_index]
        path  = os.path.join(CROP_DIR, fname)
        img   = Image.open(path).resize((300, int(300*Image.open(path).height/Image.open(path).width)), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tkimg)

        self.count_label.config(text=f"Images remaining: {len(self.files)}")

        if fname in self.labels:
            val, skl, drop = self.labels[fname]
            self.xp_entry.delete(0, tk.END); self.xp_entry.insert(0, val)
            self.skill_combo.set(skl)
            self.drop_var.set(drop.lower()=="yes")
        else:
            self.drop_var.set(False)
            self.skill_combo.set("None")

        self._update_prediction(fname)
        self.update_stats()


    def _on_xp_changed(self, event):
        if self.xp_entry.get().strip():
            self.drop_var.set(True)


    def _update_prediction(self, fname):
        path = os.path.join(CROP_DIR, fname)
        img  = Image.open(path).convert("RGB")
        inp  = self.transform(img).unsqueeze(0).to(self.device)

        parts = []
        if self.xp_model:
            parts.append(f"XP≈{self.xp_model(inp).item():.1f}")
        if self.skill_model:
            logits = self.skill_model(inp)
            probs  = torch.softmax(logits, dim=1)
            p, idx = probs.max(1)
            name   = self.skill_names[idx.item()] if idx < len(self.skill_names) else f"class_{idx}"
            parts.append(f"Skill={name} ({p.item()*100:.0f}%)")
        if self.detector:
            dp      = self.detector(inp).item()
            verdict = "Yes" if dp>0.5 else "No"
            parts.append(f"Drop={verdict} ({dp*100:.0f}%)")

        self.pred_label.config(text=" ▶ ".join(parts))


    def prev_image(self):
        if not self.files: return
        self.current_index = max(0, self.current_index-1)
        self.status_label.config(text="")
        self.load_image()


    def next_image(self):
        if not self.files: return
        self.current_index = min(len(self.files)-1, self.current_index+1)
        self.load_image()


    def save_label(self):
        if not self.files: return
        fname  = self.files[self.current_index]
        drop   = self.drop_var.get()
        xp_val = self.xp_entry.get().strip() if drop else ""
        skl    = self.skill_combo.get().strip()
        if drop:
            if skl == "None" or not xp_val:
                self.status_label.config(text="Select a real skill and enter XP for a drop.", fg="red")
                return
        else:
            xp_val = ""
            skl    = ""

        # record eval
        if self.xp_model or self.skill_model or self.detector:
            img = Image.open(os.path.join(CROP_DIR, fname)).convert("RGB")
            inp = self.transform(img).unsqueeze(0).to(self.device)
            rec = {
                "xp_true": float(xp_val) if xp_val else 0.0,
                "xp_pred": self.xp_model(inp).item() if self.xp_model else 0.0,
                "skill_true": skl,
                "skill_pred": None,
                "drop_true":  "yes" if drop else "no",
                "drop_pred":  None
            }
            if self.skill_model and skl:
                probs = torch.softmax(self.skill_model(inp), dim=1)
                rec["skill_pred"] = SKILLS[probs.argmax(1).item()]
            if self.detector:
                rec["drop_pred"] = "yes" if self.detector(inp).item()>0.5 else "no"
            self.eval_history.append(rec)

        # save
        self.save_history.append((fname, xp_val, skl, "yes" if drop else "no"))
        self.labels[fname] = (xp_val, skl, "yes" if drop else "no")
        shutil.move(os.path.join(CROP_DIR, fname), os.path.join(LABELED_DIR, fname))
        del self.files[self.current_index]
        if self.current_index >= len(self.files):
            self.current_index = max(0, len(self.files)-1)

        self.status_label.config(text=f"Saved {fname}", fg="green")
        self.load_image()


    def undo_label(self):
        if not self.save_history:
            self.status_label.config(text="Nothing to undo.", fg="orange")
            return
        fname, xp_val, skl, drop = self.save_history.pop()
        self.labels.pop(fname, None)
        shutil.move(os.path.join(LABELED_DIR, fname), os.path.join(CROP_DIR, fname))
        self.files.insert(self.current_index+1, fname)
        self.current_index = self.files.index(fname)
        self.load_image()
        self.xp_entry.delete(0, tk.END); self.xp_entry.insert(0, xp_val)
        self.skill_combo.set(skl); self.drop_var.set(drop=="yes")
        self.status_label.config(text=f"Undid {fname}", fg="blue")


    def export_csv(self):
        if not self.labels:
            messagebox.showinfo("Export CSV", "No labels to export.")
            return
        with open(LABEL_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "value", "skill", "drop"])
            for fn, (val, skl, drop) in sorted(self.labels.items()):
                w.writerow([fn, val, skl, drop])
        self.status_label.config(text=f"Exported {len(self.labels)} labels", fg="darkgreen")


    def update_stats(self):
        total     = len(self.labels)
        remaining = len(self.files)
        yes_count = sum(1 for *_,d in self.labels.values() if d=="yes")
        no_count  = total - yes_count

        # primary
        self.stats_label.config(
            text=f"Labeled: {total} (Yes: {yes_count}, No: {no_count}), Remaining: {remaining}"
        )
        # distributions
        xp_vals = [int(v) for v,_,d in self.labels.values() if v]
        skills  = [s for _,s,_ in self.labels.values() if s]
        xp_dist = Counter(xp_vals)
        sk_dist = Counter(skills)
        parts2 = []
        if xp_dist:
            parts2.append("XP counts: " + ", ".join(f"{xp}×{c}" for xp,c in xp_dist.most_common(5)))
        if sk_dist:
            parts2.append("Skills: " + ", ".join(f"{sk}×{c}" for sk,c in sk_dist.items()))
        self.stats2_label.config(text=" | ".join(parts2))

        # accuracy
        mae = skill_acc = drop_acc = None
        if self.eval_history:
            errs      = [abs(r["xp_true"]-r["xp_pred"]) for r in self.eval_history]
            mae       = sum(errs)/len(errs)
            valid_sk  = [r for r in self.eval_history if r["skill_true"]]
            if valid_sk:
                skill_acc = sum(r["skill_true"]==r["skill_pred"] for r in valid_sk)/len(valid_sk)
            drop_acc  = sum(r["drop_true"]==r["drop_pred"] for r in self.eval_history)/len(self.eval_history)
        parts3 = []
        if mae is not None:
            parts3.append(f"XP MAE: {mae:.1f}")
        if skill_acc is not None:
            parts3.append(f"Skill Acc: {skill_acc*100:.0f}%")
        if drop_acc is not None:
            parts3.append(f"Drop Acc: {drop_acc*100:.0f}%")
        self.stats3_label.config(text=" | ".join(parts3))


if __name__ == "__main__":
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

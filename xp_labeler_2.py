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

            # === Dark Theme Colors ===
            self.bg_color = '#1e1e1e';
            self.fg_color = '#ffffff'
            self.button_bg = '#333333';
            self.button_fg = '#ffffff'
            self.entry_bg = '#2b2b2b';
            self.entry_fg = '#ffffff'
            self.highlight_color = '#5e9cff';
            self.shortcut_color = '#CCCC66'

            self.parent.option_add('*Font', ('Helvetica', 12))
            for opt in ['Background', 'Frame.background', 'Label.background']:
                self.parent.option_add(f'*{opt}', self.bg_color)
            for opt in ['Foreground', 'Label.foreground', 'Entry.foreground']:
                self.parent.option_add(f'*{opt}', self.fg_color)
            self.parent.option_add('*Entry.background', self.entry_bg)
            self.parent.option_add('*Spinbox.background', self.entry_bg)
            self.parent.option_add('*Button.background', self.button_bg)
            self.parent.option_add('*Button.foreground', self.button_fg)
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('TCombobox',
                            fieldbackground=self.entry_bg,
                            background=self.entry_bg,
                            foreground=self.entry_fg)
            style.map('TCombobox',
                      fieldbackground=[('readonly', self.entry_bg)],
                      foreground=[('readonly', self.entry_fg)])

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

if __name__ == '__main__':
    root = tk.Tk()
    XPLabeler(root)
    root.mainloop()

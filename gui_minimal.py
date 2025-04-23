#!/usr/bin/env python3
import os
import pandas as pd
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.transforms import Grayscale, ToTensor
from collections import Counter

# ─── ANSI COLORS ──────────────────────────────────────────────────────────────
RED     = '\033[91m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
MAGENTA = '\033[95m'
CYAN    = '\033[96m'
RESET   = '\033[0m'

# ─── Hyperparameters ─────────────────────────────────────────────────────────
LABEL_CSV   = "data/xp_labels.csv"
CROP_DIR    = "data/xp_crops_labeled"
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 10
VAL_FRAC    = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Dataset for XP & Skill ─────────────────────────────────────────────────
class XPSD(Dataset):
    def __init__(self, label_csv, crop_dir, transform=None):
        df = pd.read_csv(label_csv)
        df['drop']  = df['drop'].fillna('no').astype(str).str.lower()
        df = df[df['drop'] == 'yes'].reset_index(drop=True)
        df['skill'] = df['skill'].fillna('').astype(str)
        df = df[df['skill'] != ''].reset_index(drop=True)
        self.df = df

        skills = sorted(df['skill'].unique().tolist())
        self.skill_to_idx = {s:i for i,s in enumerate(skills)}
        self.idx_to_skill = {i:s for s,i in self.skill_to_idx.items()}

        self.crop_dir  = crop_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.crop_dir, row['filename'])).convert("RGB")
        img = self.transform(img) if self.transform else ToTensor()(img)

        xp = torch.tensor(float(row['value']) if row['value'] else 0.0,
                          dtype=torch.float32)
        skill_idx = self.skill_to_idx[row['skill']]
        drop_lbl  = 1
        return img, xp, skill_idx, drop_lbl

# ─── Dataset for Drop Detection ───────────────────────────────────────────────
class DropDataset(Dataset):
    def __init__(self, label_csv, crop_dir, transform=None):
        df = pd.read_csv(label_csv)
        df['drop'] = df['drop'].fillna('no').str.lower()
        self.df = df.reset_index(drop=True)

        self.crop_dir  = crop_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.crop_dir, row['filename'])).convert("RGB")
        img = self.transform(img) if self.transform else ToTensor()(img)

        drop_lbl = 1 if row['drop'] == 'yes' else 0
        return img, drop_lbl

# ─── Models ──────────────────────────────────────────────────────────────────
class XPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128),     nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,1)
        )
    def forward(self,x): return self.net(x).squeeze(1)

class SkillClassifier(nn.Module):
    def __init__(self, n_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128),     nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,n_skills)
        )
    def forward(self,x): return self.net(x)

class DropDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,64),      nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64,1),             nn.Sigmoid()
        )
    def forward(self,x): return self.net(x).squeeze(1)

# ─── Training Routines ───────────────────────────────────────────────────────
def train_regressor(model, loader, opt, loss_fn):
    model.train()
    total_sq, count = 0., 0
    for imgs, xp_true, _, _ in loader:
        imgs, xp_true = imgs.to(DEVICE), xp_true.to(DEVICE)
        preds = model(imgs)
        loss  = loss_fn(preds, xp_true)
        opt.zero_grad(); loss.backward(); opt.step()
        total_sq += ((preds-xp_true)**2).sum().item()
        count    += xp_true.size(0)
    return total_sq/count, (total_sq/count)**0.5

def train_classifier(model, loader, opt, loss_fn):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, _, skill_true, _ in loader:
        imgs, skill_true = imgs.to(DEVICE), skill_true.to(DEVICE)
        logits = model(imgs)
        loss   = loss_fn(logits, skill_true)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()*imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds==skill_true).sum().item()
        total   += imgs.size(0)
    return total_loss/total, correct/total if total>0 else 0.0

def train_detector(model, loader, opt, loss_fn):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, drop_true in loader:
        imgs, drop_true = imgs.to(DEVICE), drop_true.to(DEVICE).float()
        preds = model(imgs)
        loss  = loss_fn(preds, drop_true)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()*imgs.size(0)
        pred_lbl = (preds>0.5).long()
        correct  += (pred_lbl==drop_true.long()).sum().item()
        total    += imgs.size(0)
    return total_loss/total, correct/total if total>0 else 0.0

# ─── Main Pipeline ──────────────────────────────────────────────────────────
if __name__=="__main__":
    print(f"{CYAN}== Starting Training Pipeline =={RESET}\n")

    # build transforms
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        Grayscale(num_output_channels=1),
        ToTensor()
    ])

    # XP+Skill dataset & split
    xpds = XPSD(LABEL_CSV, CROP_DIR, transform)
    n_val = int(len(xpds)*VAL_FRAC); n_trn = len(xpds)-n_val
    xp_tr, xp_vl = random_split(xpds, [n_trn, n_val])
    tr_xp = DataLoader(xp_tr, batch_size=BATCH_SIZE, shuffle=True)
    vl_xp = DataLoader(xp_vl, batch_size=BATCH_SIZE, shuffle=False)

    # Drop detector dataset & split
    dd = DropDataset(LABEL_CSV, CROP_DIR, transform)
    n_val2 = int(len(dd)*VAL_FRAC); n_trn2 = len(dd)-n_val2
    dd_tr, dd_vl = random_split(dd, [n_trn2, n_val2])
    tr_dd = DataLoader(dd_tr, batch_size=BATCH_SIZE, shuffle=True)
    vl_dd = DataLoader(dd_vl, batch_size=BATCH_SIZE, shuffle=False)

    # Diagnostics
    skill_names = list(xpds.skill_to_idx.keys())
    print(f"{YELLOW}XP dataset size:{RESET} train={len(xp_tr)}  val={len(xp_vl)}")
    print(f"{YELLOW}Drop dataset size:{RESET} train={len(dd_tr)}  val={len(dd_vl)}")
    print(f"{YELLOW}Skill classes:{RESET} {skill_names}")
    use_clf = len(skill_names)>=2
    if not use_clf:
        print(f"{RED}⚠️ Skipping skill classifier (need ≥2 skills).{RESET}")
    drop_labels = set(dd.df['drop'].tolist())
    use_dd = drop_labels=={'yes','no'}
    if not use_dd:
        print(f"{RED}⚠️ Skipping drop detector (need both yes/no).{RESET}")
    print()

    # instantiate models
    xp_model = XPRegressor().to(DEVICE)
    xp_opt   = optim.Adam(xp_model.parameters(), lr=LR)
    xp_loss  = nn.MSELoss()
    if use_clf:
        sc_model = SkillClassifier(len(skill_names)).to(DEVICE)
        sc_opt   = optim.Adam(sc_model.parameters(), lr=LR)
        sc_loss  = nn.CrossEntropyLoss()
    if use_dd:
        dd_model = DropDetector().to(DEVICE)
        dd_opt   = optim.Adam(dd_model.parameters(), lr=LR)
        dd_loss  = nn.BCELoss()

    # training loops
    start = time.time()
    for e in range(1, EPOCHS+1):
        mse, rmse = train_regressor(xp_model, tr_xp, xp_opt, xp_loss)
        vmse, vrmse= train_regressor(xp_model, vl_xp, xp_opt, xp_loss)
        print(f"{GREEN}[Regressor]{RESET} Epoch {e}/{EPOCHS} ➤ train RMSE={rmse:.3f}, val MSE={vmse:.3f}")

    if use_clf:
        for e in range(1, EPOCHS+1):
            ce, acc   = train_classifier(sc_model, tr_xp, sc_opt, sc_loss)
            vce, vacc = train_classifier(sc_model, vl_xp, sc_opt, sc_loss)
            print(f"{BLUE}[Classifier]{RESET} Epoch {e}/{EPOCHS} ➤ train Acc={acc*100:.1f}%, val Acc={vacc*100:.1f}%")

    if use_dd:
        for e in range(1, EPOCHS+1):
            dl, da   = train_detector(dd_model, tr_dd, dd_opt, dd_loss)
            vdl, vda = train_detector(dd_model, vl_dd, dd_opt, dd_loss)
            print(f"{MAGENTA}[Detector]{RESET} Epoch {e}/{EPOCHS} ➤ train Acc={da*100:.1f}%, val Acc={vda*100:.1f}%")

    # save
    os.makedirs("models", exist_ok=True)
    torch.save(xp_model.state_dict(),  "models/xp_regressor.pth")
    if use_clf: torch.save(sc_model.state_dict(),  "models/skill_classifier.pth")
    if use_dd:  torch.save(dd_model.state_dict(),  "models/xp_detector.pth")

    elapsed = time.time() - start
    print(f"\n{CYAN}== Training complete in {elapsed/60:.1f} min =={RESET}")

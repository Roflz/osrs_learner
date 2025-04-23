#!/usr/bin/env python3
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Grayscale, ToTensor
from collections import Counter

# ─── Hyperparameters ─────────────────────────────────────────────────────────
LABEL_CSV   = "data/xp_labels.csv"
CROP_DIR    = "data/xp_crops_labeled"
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class XPSD(Dataset):
    def __init__(self, label_csv, crop_dir, transform=None):
        # Load and clean CSV
        df = pd.read_csv(label_csv)
        df['drop']  = df['drop'].fillna('no').astype(str)
        df = df[df['drop'].str.lower() == 'yes'].reset_index(drop=True)
        df['skill'] = df['skill'].fillna('').astype(str)
        df = df[df['skill'] != ''].reset_index(drop=True)
        self.df = df

        # Build skill mapping
        skills = sorted(df['skill'].unique().tolist())
        self.skill_to_idx = {s: i for i, s in enumerate(skills)}
        self.idx_to_skill = {i: s for s, i in self.skill_to_idx.items()}

        self.crop_dir  = crop_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load image
        img = Image.open(os.path.join(self.crop_dir, row['filename'])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        # xp as FloatTensor
        raw_xp = float(row['value']) if row['value'] else 0.0
        xp     = torch.tensor(raw_xp, dtype=torch.float32)

        # skill + drop
        skill_idx = self.skill_to_idx[row['skill']]
        drop_lbl  = 1  # always 1 here

        return img, xp, skill_idx, drop_lbl

# ─── Models ──────────────────────────────────────────────────────────────────
class XPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),   nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class SkillClassifier(nn.Module):
    def __init__(self, n_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),   nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_skills)
        )
    def forward(self, x):
        return self.net(x)

# ─── Training Routines ───────────────────────────────────────────────────────
def train_regressor(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_sq   = 0.0
    count      = 0
    for imgs, xp_true, _, _ in loader:
        imgs, xp_true = imgs.to(DEVICE), xp_true.to(DEVICE)
        preds = model(imgs)
        loss  = loss_fn(preds, xp_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xp_true.size(0)
        total_sq   += ((preds - xp_true) ** 2).sum().item()
        count     += xp_true.size(0)
    mse  = total_sq / count
    rmse = mse ** 0.5
    return mse, rmse

def train_classifier(model, loader, optimizer, loss_fn):
    model.train()
    correct = 0
    total   = 0
    total_loss = 0.0
    for imgs, _, skill_true, _ in loader:
        imgs = imgs.to(DEVICE)
        skill_true = skill_true.to(DEVICE)
        logits = model(imgs)
        loss = loss_fn(logits, skill_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == skill_true).sum().item()
        total   += imgs.size(0)

    ce  = total_loss / total
    acc = correct / total if total>0 else 0.0
    return ce, acc

# ─── Main Pipeline ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("== Starting Training Pipeline ==")

    # transforms & dataset
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    ds = XPSD(LABEL_CSV, CROP_DIR, transform=train_transform)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # diagnostic & skip logic
    skill_names = list(ds.skill_to_idx.keys())
    print(f"Found skill classes: {skill_names}")
    skip_clf = len(skill_names) < 2
    if skip_clf:
        print("⚠️ Need ≥2 distinct skills to train classifier; skipping.")

    # model + optimizer + losses
    xp_model = XPRegressor().to(DEVICE)
    xp_opt   = optim.Adam(xp_model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    if not skip_clf:
        sc_model = SkillClassifier(len(skill_names)).to(DEVICE)
        sc_opt   = optim.Adam(sc_model.parameters(), lr=LR)
        ce_loss  = nn.CrossEntropyLoss()

    # train loop
    tr_mse, tr_rmse = train_regressor(xp_model, train_loader, xp_opt, mse_loss)
    print(f"Regressor -> MSE: {tr_mse:.4f}, RMSE: {tr_rmse:.4f}")

    if not skip_clf:
        tr_ce, tr_acc = train_classifier(sc_model, train_loader, sc_opt, ce_loss)
        print(f"Classifier -> CE: {tr_ce:.4f}, Acc: {tr_acc*100:.1f}%")

    # save models
    os.makedirs("models", exist_ok=True)
    torch.save(xp_model.state_dict(),  "models/xp_regressor.pth")
    if not skip_clf:
        torch.save(sc_model.state_dict(), "models/skill_classifier.pth")

    print("== Training complete ==")

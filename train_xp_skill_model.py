#!/usr/bin/env python3
import os
import math
import copy
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Colored console output
from colorama import Fore, Style, init
init(autoreset=True)
# Detailed metrics
from sklearn.metrics import classification_report, confusion_matrix

# ─── Paths & Constants ─────────────────────────────────────────────────
CSV_PATH      = "data/xp_labels.csv"
IMG_DIR       = "data/xp_crops_labeled"
REGRESSOR_PTH = "models/xp_regressor.pth"
CLASSIFIER_PTH= "models/skill_classifier.pth"
COUNT_PTH     = "models/drop_count_regressor.pth"
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
PATIENCE      = 5
LR            = 1e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Skill Mapping ──────────────────────────────────────────────────────
SKILLS = [
    "None", "woodcutting", "mining", "fishing", "cooking", "firemaking",
    "fletching", "thieving", "agility", "herblore", "crafting",
    "smithing", "runecrafting", "slayer", "farming", "construction", "hunter"
]
SKILL2IDX   = {s:i for i,s in enumerate(SKILLS)}
NUM_SKILLS  = len(SKILLS)

# ─── Transforms ─────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomAffine(10, translate=(0.05,0.05)),
    transforms.ColorJitter(0.2,0.2),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])
val_tf = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

# ─── Datasets ───────────────────────────────────────────────────────────
class XPMultiDataset(Dataset):
    def __init__(self, df, transform, max_drops):
        df = df.copy()
        df['value_list'] = df['value'].fillna("").astype(str).str.split(';').map(
            lambda lst: [float(v) for v in lst if v.strip()]
        )
        df['drop_count'] = df['value_list'].map(len)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.max_drops = max_drops
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(IMG_DIR, row.filename)).convert('RGB')
        x = self.transform(img)
        y = np.zeros(self.max_drops, dtype=np.float32)
        for j,v in enumerate(row.value_list): y[j] = v
        return x, torch.tensor(y)

class SkillMultiDataset(Dataset):
    """
    Per-drop skill labels: repeats skill index for each drop, pads zeros for missing.
    Only rows with drop_count > 0 are kept, and skill_idx is guaranteed integer.
    """
    def __init__(self, df, transform, max_drops):
        df = df.copy()
        df['value_list'] = (
            df['value']
              .fillna("")
              .astype(str)
              .str.split(';')
              .map(lambda lst: [float(v) for v in lst if v.strip()])
        )
        df['drop_count'] = df['value_list'].map(len).astype(int)

        # filter to only examples that actually have drops
        df = df[df['drop_count'] > 0].reset_index(drop=True)

        # map skill names to indices, drop any rows where mapping failed
        df['skill_idx'] = df['skill'].map(SKILL2IDX)
        df = df[df['skill_idx'].notnull()].copy()
        df['skill_idx'] = df['skill_idx'].astype(int)

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.max_drops = max_drops

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(IMG_DIR, row.filename)).convert('RGB')
        x = self.transform(img)

        # build a length-max_drops int array, repeating skill_idx for each drop
        y = np.zeros(self.max_drops, dtype=np.int64)
        for j in range(row.drop_count):
            y[j] = row.skill_idx
        return x, torch.tensor(y)

class DropCountDataset(Dataset):
    def __init__(self, df, transform):
        df = df.copy()
        df['value_list'] = df['value'].fillna("").astype(str).str.split(';').map(
            lambda lst: [float(v) for v in lst if v.strip()]
        )
        df['drop_count'] = df['value_list'].map(len)
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(IMG_DIR, row.filename)).convert('RGB')
        x = self.transform(img)
        return x, torch.tensor(row.drop_count, dtype=torch.float32)

# ─── Model Builders ───────────────────────────────────────────────────────
def make_regressor(output_dim):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features,128),
        nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128,output_dim)
    )
    return m.to(DEVICE)

def make_multi_classifier(max_drops):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
    m.fc = nn.Linear(m.fc.in_features, max_drops * NUM_SKILLS)
    return m.to(DEVICE)

# ─── Training & Eval ──────────────────────────────────────────────────────
def train_one(model, loader, criterion, optimizer, is_class=False, max_drops=None):
    model.train()
    total_loss=0.0
    for x,y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        if is_class:
            B = x.size(0)
            out = out.view(B, max_drops, NUM_SKILLS)
            loss = sum(criterion(out[:,d,:], y[:,d]) for d in range(max_drops))
        else:
            out = out.squeeze(-1)
            loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_one(model, loader, criterion, is_class=False, max_drops=None):
    model.eval()
    total_loss=0.0
    correct = [0]*max_drops if is_class else None
    total = 0
    with torch.no_grad():
        for x,y in loader:
            B = x.size(0)
            x = x.to(DEVICE); y = y.to(DEVICE)
            out = model(x)
            if is_class:
                out = out.view(B, max_drops, NUM_SKILLS)
                loss = sum(criterion(out[:,d,:], y[:,d]) for d in range(max_drops))
                for d in range(max_drops):
                    preds = out[:,d,:].argmax(1)
                    correct[d] += (preds == y[:,d]).sum().item()
            else:
                out = out.squeeze(-1)
                loss = criterion(out, y)
            total_loss += loss.item() * B
            total += B
    avg_loss = total_loss / total
    if is_class:
        return avg_loss, [c/total for c in correct]
    return avg_loss

def run_training(name, model, tr_loader, va_loader, crit, opt, sched, is_class=False, max_drops=None):
    print(Fore.CYAN + f"\n=== {name} ===" + Style.RESET_ALL)
    best_state  = copy.deepcopy(model.state_dict())
    best_metric = -math.inf if is_class else math.inf
    no_imp = 0
    for epoch in range(1, NUM_EPOCHS+1):
        tl = train_one(model, tr_loader, crit, opt, is_class, max_drops)
        if is_class:
            vl, accs = eval_one(model, va_loader, crit, True, max_drops)
            metric = sum(accs)/len(accs)
        else:
            vl = eval_one(model, va_loader, crit, False)
            metric = vl
        improved = (metric > best_metric) if is_class else (vl < best_metric)
        status = (Fore.GREEN+"improved" if improved else Fore.YELLOW+"no change")+Style.RESET_ALL
        if improved:
            best_metric = metric; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
        if is_class:
            print(f"Epoch {epoch} [{status}]: val_loss={vl:.4f}, mean_acc={np.mean(accs)*100:5.1f}%")
            sched.step(np.mean(accs))
        else:
            rmse = math.sqrt(tl)
            vrmse = math.sqrt(vl)
            print(f"Epoch {epoch} [{status}]: MSE={tl:.4f}, RMSE={rmse:.4f} | val MSE={vl:.4f}, val RMSE={vrmse:.4f}")
            sched.step(vl)
        if no_imp >= PATIENCE:
            print(Fore.MAGENTA+"→ Early stopping"+Style.RESET_ALL)
            break
    model.load_state_dict(best_state)
    print(Fore.CYAN + f"=== Done {name} ===\n" + Style.RESET_ALL)
    return model

# ─── Main ──────────────────────────────────────────────────────────────────
if __name__=='__main__':
    # ─── Load & parse labels CSV ────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df['value_list'] = df['value'].fillna("").astype(str).str.split(';').map(
        lambda lst: [float(v) for v in lst if v.strip()])
    df['drop_count'] = df['value_list'].map(len)
    max_drops = int(df['drop_count'].max())

    # ─── Dataset Summary ──────────────────────────────────────────────────────
    skills_present = sorted(df['skill'].dropna().unique())
    print(Fore.YELLOW + "Dataset Summary:" + Style.RESET_ALL)
    print(f" - Skills present ({len(skills_present)}): {', '.join(skills_present)}")
    print(f" - Maximum XP drops in any one crop: {max_drops}\n")

    # ─── Input Data Details ───────────────────────────────────────────────────
    print(Fore.YELLOW + "Input Data Details:" + Style.RESET_ALL)
    print(f" - Total samples: {len(df)}")
    idx = np.arange(len(df));
    np.random.shuffle(idx)
    split = int(0.8 * len(df))
    df_tr, df_va = df.iloc[idx[:split]], df.iloc[idx[split:]]
    for name, split_df in [("Train", df_tr), ("Val", df_va)]:
        dc = split_df['drop_count'].value_counts().sort_index()
        print(f" - {name} drop-count distribution:")
        for cnt, num in dc.items():
            print(f"    {cnt}: {num}")
        all_xp = [v for lst in split_df['value_list'] for v in lst]
        if all_xp:
            print(
                f" - {name} XP values summary: mean={np.mean(all_xp):.1f}, std={np.std(all_xp):.1f}, min={np.min(all_xp):.1f}, max={np.max(all_xp):.1f}")

    # DataLoaders
    xp_tr = DataLoader(XPMultiDataset(df_tr, train_tf, max_drops), batch_size=BATCH_SIZE, shuffle=True)
    xp_va = DataLoader(XPMultiDataset(df_va, val_tf,   max_drops), batch_size=BATCH_SIZE)
    sk_tr = DataLoader(SkillMultiDataset(df_tr, train_tf, max_drops), batch_size=BATCH_SIZE, shuffle=True)
    sk_va = DataLoader(SkillMultiDataset(df_va, val_tf,   max_drops), batch_size=BATCH_SIZE)
    dc_tr = DataLoader(DropCountDataset(df_tr, train_tf), batch_size=BATCH_SIZE, shuffle=True)
    dc_va = DataLoader(DropCountDataset(df_va, val_tf), batch_size=BATCH_SIZE)

    # Informative overview before training
    print(Fore.YELLOW + "\nModels and Data Overview:" + Style.RESET_ALL)
    print(f" - XP Multi-Drop Regressor: train samples={len(xp_tr.dataset)}, val samples={len(xp_va.dataset)}, input shape=1×64×64, output slots={max_drops}")
    print(f" - Skill Multi-Classifier: train samples={len(sk_tr.dataset)}, val samples={len(sk_va.dataset)}, slots={max_drops}, classes={NUM_SKILLS}")
    print(f" - Drop-Count Regressor: train samples={len(dc_tr.dataset)}, val samples={len(dc_va.dataset)}, input shape=1×64×64, output scalar")
    print(Fore.CYAN + "Metrics Legend: MSE/RMSE for regressors; val_loss/mean_acc for classifier; per-slot RMSE & supports; detailed confusion matrices." + Style.RESET_ALL)

    # Train XP Multi-Drop Regressor
    net_xp = make_regressor(max_drops)
    opt_xp = optim.Adam(net_xp.parameters(), lr=LR)
    sched_xp = optim.lr_scheduler.ReduceLROnPlateau(opt_xp, 'min', 0.5, 2)
    crit_xp = nn.MSELoss()
    net_xp = run_training('XP Multi-Drop Regressor', net_xp, xp_tr, xp_va, crit_xp, opt_xp, sched_xp, False)

    # per-slot RMSE & support counts
    print(Fore.MAGENTA + "\nXP Regressor Per-Drop RMSE & Support:" + Style.RESET_ALL)
    errors = np.zeros(max_drops)
    counts = np.zeros(max_drops)
    net_xp.eval()
    with torch.no_grad():
        for x,y in xp_va:
            preds = net_xp(x.to(DEVICE)).cpu().numpy()
            truths = y.numpy()
            for i in range(max_drops):
                errors[i] += ((preds[:,i] - truths[:,i])**2).sum()
                counts[i] += truths.shape[0]
    overall_mse = errors.sum() / counts.sum()
    overall_rmse = math.sqrt(overall_mse)
    for i in range(max_drops):
        mse_i = errors[i]/counts[i]
        rmse_i = math.sqrt(mse_i)
        print(f"  Slot {i+1}: support={int(counts[i])}, RMSE={rmse_i:.2f}")
    print(f"  → Overall across all slots: RMSE={overall_rmse:.2f}")

    # Train Skill Multi-Classifier
    net_sk = make_multi_classifier(max_drops)
    opt_sk = optim.Adam(net_sk.parameters(), lr=LR)
    sched_sk = optim.lr_scheduler.ReduceLROnPlateau(opt_sk, 'max', 0.5, 2)
    crit_sk = nn.CrossEntropyLoss()
    net_sk = run_training('Skill Multi-Classifier', net_sk, sk_tr, sk_va, crit_sk, opt_sk, sched_sk, True, max_drops)
    torch.save(net_sk.state_dict(), CLASSIFIER_PTH)

    # Detailed skill report
    print(Fore.MAGENTA + "\nSkill Classifier Detailed Report:" + Style.RESET_ALL)
    y_true, y_pred = [], []
    net_sk.eval()
    with torch.no_grad():
        for x,y in sk_va:
            B = x.size(0)
            out = net_sk(x.to(DEVICE)).view(B, max_drops, NUM_SKILLS)
            p = out.argmax(2).cpu().numpy()
            y_pred.extend(p.flatten())
            y_true.extend(y.numpy().flatten())
    labels = sorted(set(y_true))
    names  = [SKILLS[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names))
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels)
    print("Normalized Confusion Matrix (rows=true):"); print(np.round(cm_norm,2))
    print("Raw Confusion Matrix (counts):");       print(cm_raw)

    # Train Drop-Count Regressor
    net_dc = make_regressor(1)
    opt_dc = optim.Adam(net_dc.parameters(), lr=LR)
    sched_dc = optim.lr_scheduler.ReduceLROnPlateau(opt_dc, 'min', 0.5, 2)
    crit_dc = nn.MSELoss()
    net_dc = run_training('Drop-Count Regressor', net_dc, dc_tr, dc_va, crit_dc, opt_dc, sched_dc, False)

    # Drop-count as classification report
    print(Fore.MAGENTA + "\nDrop-Count Classification Report:" + Style.RESET_ALL)
    y_true, y_pred = [], []
    net_dc.eval()
    with torch.no_grad():
        for x,y in dc_va:
            trues = y.numpy().astype(int)
            preds = np.round(net_dc(x.to(DEVICE)).cpu().numpy().flatten()).astype(int)
            y_true.extend(trues); y_pred.extend(preds)
    labels = sorted(set(y_true))
    print(classification_report(y_true, y_pred, labels=labels, target_names=[str(l) for l in labels]))
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm_raw  = confusion_matrix(y_true, y_pred, labels=labels)
    print("Normalized Drop-Count Confusion Matrix:"); print(np.round(cm_norm,2))
    print("Raw Drop-Count Confusion Matrix:");       print(cm_raw)

    # save
    torch.save(net_xp.state_dict(), REGRESSOR_PTH)
    torch.save(net_sk.state_dict(), CLASSIFIER_PTH)
    torch.save(net_dc.state_dict(), COUNT_PTH)
    print(Fore.CYAN + "\nFinished training and saved models with detailed metrics." + Style.RESET_ALL)

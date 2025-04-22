import os
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from colorama import init, Fore, Style

# Initialize colorama for colored console output\init(autoreset=True)

# == CONFIG ==
DATA_DIR = "data"
CROP_DIR = os.path.join(DATA_DIR, "xp_crops_labeled")  # labeled crops here
LABEL_CSV = os.path.join(DATA_DIR, "xp_labels.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5  # L2 regularization
EPOCHS = 10
IMG_SIZE = (64, 64)
SKILL_CLASSES = []  # will be populated from dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# == DATASET ==
class XPSD(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        global SKILL_CLASSES
        SKILL_CLASSES = sorted(self.df['skill'].unique().tolist())
        self.skill_to_idx = {s: i for i, s in enumerate(SKILL_CLASSES)}
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row.filename))
        img = self.transform(img)
        xp = torch.tensor(float(row.value), dtype=torch.float32)
        skill = self.skill_to_idx[row.skill]
        presence = torch.tensor(1.0 if xp.item() > 0 else 0.0, dtype=torch.float32)
        return img, xp, skill, presence

# == MODELS ==
class XPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        feat = 32 * (IMG_SIZE[0]//4) * (IMG_SIZE[1]//4)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class SkillClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        feat = 32 * (IMG_SIZE[0]//4) * (IMG_SIZE[1]//4)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class XPDetector(nn.Module):
    def __init__(self):
        super().__init__()
        feat = 32 * (IMG_SIZE[0]//4) * (IMG_SIZE[1]//4)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# == TRAIN/EVAL FUNCTIONS ==
def train_regressor(model, loader, opt, loss_fn):
    model.train(); total_loss=0; total=0
    for imgs, xp, _, _ in loader:
        imgs, xp = imgs.to(DEVICE), xp.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, xp)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    mse = total_loss / total
    rmse = mse**0.5
    return mse, rmse


def eval_regressor(model, loader, loss_fn):
    model.eval(); total_loss=0; total=0
    with torch.no_grad():
        for imgs, xp, _, _ in loader:
            imgs, xp = imgs.to(DEVICE), xp.to(DEVICE)
            preds = model(imgs)
            loss = loss_fn(preds, xp)
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
    mse = total_loss / total
    rmse = mse**0.5
    return mse, rmse


def train_classifier(model, loader, opt, loss_fn):
    model.train(); total_loss=0; total=0; correct=0
    for imgs, _, skill, _ in loader:
        imgs, skill = imgs.to(DEVICE), skill.to(DEVICE)
        logits = model(imgs)
        loss = loss_fn(logits, skill)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == skill).sum().item()
        total += imgs.size(0)
    ce = total_loss / total
    acc = correct / total
    return ce, acc


def eval_classifier(model, loader, loss_fn):
    model.eval(); total_loss=0; total=0; correct=0
    with torch.no_grad():
        for imgs, _, skill, _ in loader:
            imgs, skill = imgs.to(DEVICE), skill.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits, skill)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == skill).sum().item()
            total += imgs.size(0)
    ce = total_loss / total
    acc = correct / total
    return ce, acc


def train_detector(model, loader, opt, loss_fn):
    model.train(); total_loss=0; total=0
    for imgs, _, _, pres in loader:
        imgs, pres = imgs.to(DEVICE), pres.to(DEVICE)
        logits = model(imgs)
        loss = loss_fn(logits, pres)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    bce = total_loss / total
    return bce


def eval_detector(model, loader, loss_fn):
    model.eval(); total_loss=0; total=0; correct=0
    with torch.no_grad():
        for imgs, _, _, pres in loader:
            imgs, pres = imgs.to(DEVICE), pres.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits, pres)
            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == pres).sum().item()
            total += imgs.size(0)
    bce = total_loss / total
    acc = correct / total
    return bce, acc

# == MAIN ==
if __name__ == '__main__':
    print(Fore.CYAN + Style.BRIGHT + "\n== Starting Training Pipeline ==\n")
    ds = XPSD(LABEL_CSV, CROP_DIR)
    n = len(ds)
    val_n = max(1, int(n * 0.2))
    indices = list(range(n))
    val_idx = random.sample(indices, k=val_n)
    train_idx = [i for i in indices if i not in val_idx]

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE)

    # Instantiate models
    xp_model = XPRegressor().to(DEVICE)
    sc_model = SkillClassifier(len(SKILL_CLASSES)).to(DEVICE)
    det_model= XPDetector().to(DEVICE)

    print(Fore.CYAN + Style.BRIGHT + f"Skill classes: {SKILL_CLASSES}\n")

    # Opts, schedulers, losses
    xp_opt, sc_opt, det_opt = [
        optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for m in (xp_model, sc_model, det_model)
    ]
    xp_sched = optim.lr_scheduler.ReduceLROnPlateau(xp_opt, mode='min', patience=2)
    sc_sched = optim.lr_scheduler.ReduceLROnPlateau(sc_opt, mode='max', patience=2)
    det_sched= optim.lr_scheduler.ReduceLROnPlateau(det_opt,mode='min', patience=2)

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # --- Train Loop ---
    for epoch in range(1, EPOCHS+1):
        # Regressor
        tr_mse, tr_rmse = train_regressor(xp_model, train_loader, xp_opt, mse_loss)
        val_mse, val_rmse = eval_regressor(xp_model, val_loader, mse_loss)
        xp_sched.step(val_mse)

        # Classifier
        tr_ce, tr_acc = train_classifier(sc_model, train_loader, sc_opt, ce_loss)
        val_ce, val_acc = eval_classifier(sc_model, val_loader, ce_loss)
        sc_sched.step(val_acc)

        # Detector
        tr_bce = train_detector(det_model, train_loader, det_opt, bce_loss)
        val_bce, val_det_acc = eval_detector(det_model, val_loader, bce_loss)
        det_sched.step(val_bce)

        # Print metrics
        lr = xp_opt.param_groups[0]['lr']
        print(
            Fore.YELLOW + Style.BRIGHT +
            f"Epoch {epoch}/{EPOCHS} | LR: {lr:.1e}\n"
            f"  Reg: train RMSE={tr_rmse:.1f}, val RMSE={val_rmse:.1f}\n"
            f"  Clf: train Acc={tr_acc:.2f}, val Acc={val_acc:.2f}\n"
            f"  Det: train BCE={tr_bce:.4f}, val BCE={val_bce:.4f}, val Acc={val_det_acc:.2f}\n"
        )

    # Save models
    torch.save(xp_model.state_dict(), os.path.join(MODEL_DIR,'xp_regressor.pth'))
    torch.save(sc_model.state_dict(), os.path.join(MODEL_DIR,'skill_classifier.pth'))
    torch.save(det_model.state_dict(),os.path.join(MODEL_DIR,'xp_detector.pth'))

    print(Fore.GREEN + Style.BRIGHT + "\nAll models trained and saved to 'models/'\nInterpreting results:\n"
          f" - Reg RMSE: average XP error (lower is better)\n"
          f" - Clf Acc: skill classification accuracy (> chance indicates learning)\n"
          f" - Det Acc: XP presence detection accuracy.\n")

#!/usr/bin/env python3
"""
Minimal CRNN head + flexible LR scheduling + AdamW + resume training +
OneCycleLR or CosineAnnealingLR + CUDA auto-detection, AMP, single-process DataLoader,
gradient clipping, best-model & last-model checkpointing,
PAD_TOKEN = 10 with real digits 0–9 unchanged, both character- and sequence-level accuracy,
and TorchScript compatibility by forcing max_len to a Python int.
"""
import json
import os
import random
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.utils as utils
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bars

# --- tokens & classes ---
PAD_TOKEN   = 10            # reserve index 10 for padding
NUM_CLASSES = 11            # digits 0–9 plus PAD_TOKEN

class NumberDataset(Dataset):
    # unchanged...
    def __init__(self, df, img_dir, ann_dir, max_len, transform=None):
        self.items = []
        self.anns = {}
        missing_img = missing_ann = 0

        for fn, seq in zip(df['filename'], df['sequence']):
            base   = os.path.splitext(fn)[0].split('_')[0]
            suffix = int(os.path.splitext(fn)[0].split('_')[1]) if '_' in fn else 0
            img_path = os.path.join(img_dir, f"{base}.png")
            ann_path = os.path.join(ann_dir, f"{base}.txt")

            if not os.path.isfile(img_path):
                missing_img += 1
                continue
            if not os.path.isfile(ann_path):
                missing_ann += 1
                continue

            if ann_path not in self.anns:
                with open(ann_path) as f:
                    self.anns[ann_path] = f.read().splitlines()

            digits = [int(c) for c in str(seq).strip()]
            if not digits:
                continue
            L = len(digits)
            self.items.append((img_path, ann_path, torch.tensor(digits, dtype=torch.long), suffix, L))

        print(f"🔍 Loaded {len(self.items)} samples ({missing_img} missing images, {missing_ann} missing anns)")
        self.max_len = int(max_len)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ann_path, seq, suffix, L = self.items[idx]
        img = Image.open(img_path).convert('L')

        line = self.anns[ann_path][suffix]
        _, xc, yc, w, h = line.split()
        xc, yc, w, h = map(float, (xc, yc, w, h))
        W, H = img.size
        x1, y1 = (xc - w/2)*W, (yc - h/2)*H
        x2, y2 = (xc + w/2)*W, (yc + h/2)*H
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

        if self.transform:
            crop = self.transform(crop)

        if L >= self.max_len:
            tgt, tgt_len = seq[:self.max_len], self.max_len
        else:
            pad       = torch.full((self.max_len - L,), PAD_TOKEN, dtype=torch.long)
            tgt, tgt_len = torch.cat((seq, pad), dim=0), L

        return crop, tgt, tgt_len, img_path


class RealDataDataset(Dataset):
    # unchanged...
    def __init__(self, json_dir, img_dir, label_dir, max_len, transform=None):
        self.items = []
        self.transform = transform
        self.max_len = int(max_len)

        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        for jf in json_files:
            jf_path = os.path.join(json_dir, jf)
            with open(jf_path, 'r') as f:
                data = json.load(f)

            img_path = os.path.join(img_dir, data['filename'])
            ann_path = os.path.join(label_dir, data['filename'].replace('.png', '.txt'))

            if not os.path.exists(img_path) or not os.path.exists(ann_path):
                continue

            with open(ann_path, 'r') as f_ann:
                ann_lines = f_ann.read().splitlines()

            for i, xp in enumerate(data.get('xp_values', [])):
                if i >= len(ann_lines):
                    continue
                parts = ann_lines[i].split()
                if len(parts) != 5:
                    continue
                try:
                    _, xc, yc, w, h = map(float, parts)
                except ValueError:
                    continue
                digits = [int(c) for c in str(xp)]
                if not digits:
                    continue
                self.items.append({
                    'img_path': img_path,
                    'bbox':      (xc, yc, w, h),
                    'digits':    torch.tensor(digits, dtype=torch.long),
                    'length':    len(digits)
                })

        print(f"🔍 Loaded {len(self.items)} real samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item['img_path']).convert('L')
        W, H = img.size
        xc, yc, w, h = item['bbox']
        x1, y1 = (xc - w/2)*W, (yc - h/2)*H
        x2, y2 = (xc + w/2)*W, (yc + h/2)*H
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

        if self.transform:
            crop = self.transform(crop)

        digits, L = item['digits'], item['length']
        if L >= self.max_len:
            tgt, tgt_len = digits[:self.max_len], self.max_len
        else:
            pad = torch.full((self.max_len - L,), PAD_TOKEN, dtype=torch.long)
            tgt, tgt_len = torch.cat((digits, pad), dim=0), L

        return crop, tgt, tgt_len, item['img_path']


def pad_collate(batch):
    imgs, targets, lengths, paths = zip(*batch)
    return (
        torch.stack(imgs, 0),
        torch.stack(targets, 0),
        torch.tensor(lengths, dtype=torch.long),
        paths
    )


class CRNNSimple(nn.Module):
    # unchanged...
    def __init__(self, num_classes=NUM_CLASSES, max_len=5):
        super().__init__()
        self.max_len = int(max_len)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.rnn = nn.LSTM(256,128,2, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.rnn_layer_norm = nn.LayerNorm(256)
        self.dropout        = nn.Dropout(0.3)
        self.fc             = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.cnn(x)                    # (B,256,H',W')
        f = f.mean(dim=2, keepdim=True)    # (B,256,1,W')
        f = f.squeeze(2).permute(0,2,1)    # (B,W',256)
        o, _ = self.rnn(f)                 # (B,W',256)
        o = self.rnn_layer_norm(o)
        o = self.dropout(o)
        logits = self.fc(o)                # (B,W',C)

        if logits.size(1) < self.max_len:
            pad = torch.zeros(
                logits.size(0),
                self.max_len - logits.size(1),
                logits.size(2),
                device=logits.device
            )
            logits = torch.cat((logits, pad), dim=1)
        else:
            logits = logits[:, :self.max_len, :]
        return logits

def train_one(model, loader, criterion, optimizer, device, scheduler=None, scaler=None, epoch=0):
    model.train()
    total, count = 0.0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch} ▶ train", leave=False)
    for imgs, targets, _, _ in loop:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler:
            with autocast():
                logits = model(imgs)
                B, L, C = logits.size()
                loss = criterion(logits.reshape(B*L, C), targets.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            B, L, C = logits.size()
            loss = criterion(logits.reshape(B*L, C), targets.reshape(-1))
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total += loss.item() * B
        count += B
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg = total / count
    print(f"—> Epoch {epoch} TRAIN   loss: {avg:.4f}")
    return avg

def validate(model, loader, criterion, device, epoch=0):
    model.eval()
    total_loss = total_chars = correct_chars = seq_correct = count = 0
    loop = tqdm(loader, desc=f"Epoch {epoch} ✅ val  ", leave=False)
    with torch.no_grad():
        for imgs, targets, lengths, _ in loop:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            B, L, C = logits.size()

            loss = criterion(logits.reshape(B*L, C), targets.reshape(-1))
            total_loss += loss.item() * B
            count      += B

            preds   = logits.argmax(2).cpu()
            tgt_cpu = targets.cpu()
            for i in range(B):
                Li = lengths[i].item()
                total_chars   += Li
                correct_chars += (preds[i, :Li] == tgt_cpu[i, :Li]).sum().item()
                if torch.equal(preds[i, :Li], tgt_cpu[i, :Li]):
                    seq_correct += 1

            loop.set_postfix(val_loss=f"{loss.item():.4f}")

    avg_loss = total_loss / count
    char_acc = correct_chars / total_chars
    seq_acc  = seq_correct    / count
    print(f"—> Epoch {epoch} VAL     loss: {avg_loss:.4f} | char_acc: {char_acc:.3f} | seq_acc: {seq_acc:.3f}")
    return avg_loss, char_acc, seq_acc

def visualize_val_preds(model, dataset, device, num_vis=6, out_path='val_preds.png'):
    model.eval()
    idxs = random.sample(range(len(dataset)), num_vis)
    cols, rows = min(3, num_vis), (num_vis + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    with torch.no_grad():
        for ax, idx in zip(axes, idxs):
            crop, tgt, tgt_len, img_path = dataset[idx]
            inp = crop.unsqueeze(0).to(device)
            logits = model(inp)
            preds = logits.argmax(-1)[0]

            true_str = ''.join(str(int(d)) for d in tgt[:tgt_len] if d.item() != PAD_TOKEN)
            pred_str = ''.join(str(int(p)) for p in preds[:tgt_len] if p.item() != PAD_TOKEN)

            ax.imshow(crop.squeeze(0).cpu(), cmap='gray')
            ax.axis('off')
            ax.set_title(f"{os.path.basename(img_path)}\nT:{true_str} P:{pred_str}", fontsize=10)

    for ax in axes[num_vis:]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✅ Saved val predictions to {out_path}")


if __name__ == '__main__':
    # --- argument parsing ---
    p = argparse.ArgumentParser()
    p.add_argument(
        '--yolo_root',
        default='data/yolo',
        help="Root of your YOLO data tree, e.g. 'data/yolo'"
    )
    p.add_argument(
        '--labels',
        default='data/yolo/synth_numbers/synth_map.csv',
        help="Path to the full synthetic map CSV, e.g. data/yolo/synth_numbers/synth_map.csv"
    )
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch',  type=int, default=32)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--out_dir',default='models')
    p.add_argument(
        '--scheduler',
        choices=['onecycle','cosine'],
        default='onecycle'
    )
    args = p.parse_args()

    # --- compute max_len from synth_map.csv ---
    df = pd.read_csv(args.labels)
    max_len = int(df['sequence'].astype(str).map(len).max())
    print(f"ℹ️  Found {len(df)} synthetic blocks, max sequence length = {max_len}")

    # --- build paths off yolo_root ---
    R = args.yolo_root
    synth_train_images = os.path.join(R, 'synth_numbers', 'train', 'images')
    synth_train_labels = os.path.join(R, 'synth_numbers', 'train', 'labels')
    synth_val_images   = os.path.join(R, 'synth_numbers', 'val',   'images')
    synth_val_labels   = os.path.join(R, 'synth_numbers', 'val',   'labels')
    real_train_json    = os.path.join(R, 'real', 'train', 'json')
    real_train_images  = os.path.join(R, 'real', 'train', 'images')
    real_train_labels  = os.path.join(R, 'real', 'train', 'labels')
    real_val_json      = os.path.join(R, 'real', 'val',   'json')
    real_val_images    = os.path.join(R, 'real', 'val',   'images')
    real_val_labels    = os.path.join(R, 'real', 'val',   'labels')

    # --- split the synth df by what actually lives in train/val folders ---
    train_fns = set(os.listdir(synth_train_images))
    val_fns   = set(os.listdir(synth_val_images))
    train_df  = df[df['filename'].isin(train_fns)].reset_index(drop=True)
    val_df    = df[df['filename'].isin(val_fns)].reset_index(drop=True)
    print(f"🔀 Using {len(train_df)} synth‑train and {len(val_df)} synth‑val samples")

    # --- transforms ---
    tf = transforms.Compose([
        transforms.Resize((64,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- datasets ---
    train_ds_synth = NumberDataset(train_df,
                                   synth_train_images,
                                   synth_train_labels,
                                   max_len, transform=tf)
    val_ds_synth   = NumberDataset(val_df,
                                   synth_val_images,
                                   synth_val_labels,
                                   max_len, transform=tf)
    train_ds_real  = RealDataDataset(real_train_json,
                                     real_train_images,
                                     real_train_labels,
                                     max_len, transform=tf)
    val_ds_real    = RealDataDataset(real_val_json,
                                     real_val_images,
                                     real_val_labels,
                                     max_len, transform=tf)

    train_ds = ConcatDataset([train_ds_synth, train_ds_real])
    val_ds   = ConcatDataset([val_ds_synth,   val_ds_real])

    loader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    tr_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                           collate_fn=pad_collate, **loader_kwargs)
    va_loader = DataLoader(val_ds,   batch_size=args.batch,
                           collate_fn=pad_collate, **loader_kwargs)

    # --- model, optimizer, loss & scaler ---
    model = CRNNSimple(num_classes=NUM_CLASSES, max_len=max_len)
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"🖥️  Detected {torch.cuda.device_count()} GPUs, using DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    scaler    = GradScaler() if use_cuda else None
    best_val_loss = float('inf')

    # --- resume-from-checkpoint logic ---
    os.makedirs(args.out_dir, exist_ok=True)
    last_ckpt   = os.path.join(args.out_dir, 'crnn_last.pth')
    start_epoch = 1
    if os.path.exists(last_ckpt):
        ckpt       = torch.load(last_ckpt, map_location=device)
        ckpt_epoch = ckpt.get('epoch', 1)
        print(f"🔄  Resuming from checkpoint: {last_ckpt} (finished epoch {ckpt_epoch})")
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        start_epoch = ckpt_epoch + 1

    # --- compute end epoch for “+N” behavior ---
    end_epoch = start_epoch + args.epochs - 1
    print(f"🔍  Will train epochs {start_epoch} through {end_epoch} (total {args.epochs} new epochs)")

    # --- scheduler setup ---
    if args.scheduler == 'onecycle':
        scheduler   = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(tr_loader),
        )
        batch_sched = True
    else:
        scheduler   = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-5,
        )
        batch_sched = False

    # --- training loop ---
    for epoch in range(start_epoch, end_epoch + 1):
        tr_l = train_one(
            model, tr_loader, criterion,
            optimizer, device,
            scheduler if batch_sched else None,
            scaler, epoch
        )
        va_l, va_char, va_seq = validate(
            model, va_loader, criterion,
            device, epoch
        )

        if not batch_sched:
            scheduler.step()

        # save best
        if va_l < best_val_loss:
            best_val_loss = va_l
            best_pth      = os.path.join(args.out_dir, 'crnn_best.pt')
            torch.jit.script(model).save(best_pth)
            print(f"💾 New best (val_loss={va_l:.4f}) saved to {best_pth}")

        # always save last
        torch.save({
            'epoch':             epoch,
            'model_state_dict':  model.state_dict(),
            'optim_state_dict':  optimizer.state_dict(),
        }, last_ckpt)

        print(
            f"Epoch {epoch:2d}: "
            f"train_loss={tr_l:.4f} | "
            f"char_acc={va_char:.3f} | "
            f"seq_acc={va_seq:.3f}"
        )

    print("✅ Training complete.")

    # --- visualize predictions ---
    visualize_val_preds(
        model, val_ds,      device,
        num_vis=10,
        out_path=os.path.join(args.out_dir, 'val_preds_synth.png')
    )
    visualize_val_preds(
        model, val_ds_real, device,
        num_vis=10,
        out_path=os.path.join(args.out_dir, 'val_preds_real.png')
    )

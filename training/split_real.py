#!/usr/bin/env python3
import os
import argparse
import shutil
import random

def split_real(src_dir: str, split_pct: int):
    """
    Splits labeled crops (png/json/txt triples) in src_dir into train/val.
    Output structure (hard‑coded):
      data/yolo/real/train/{images,json,labels}
      data/yolo/real/val/{images,json,labels}
    """

    # where to put things
    base_train = 'data/yolo/real/train'
    base_val   = 'data/yolo/real/val'

    # ── Clear previous real split ──────────────────────────────────────────
    for base in (base_train, base_val):
        cache = os.path.join(base, 'labels.cache')
        if os.path.isfile(cache):
            os.remove(cache)
        for sub in ('images', 'json', 'labels'):
            path = os.path.join(base, sub)
            if os.path.isdir(path):
                shutil.rmtree(path)

    for sub in ('images','json','labels'):
        os.makedirs(os.path.join(base_train, sub), exist_ok=True)
        os.makedirs(os.path.join(base_val,   sub), exist_ok=True)

    # discover all bases by .png
    all_png = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    bases   = [os.path.splitext(f)[0] for f in all_png]
    random.shuffle(bases)

    n_train = int(len(bases) * split_pct / 100)
    train_bases = bases[:n_train]
    val_bases   = bases[n_train:]

    def copy_group(group, out_base, label):
        total = len(group)
        for idx, base in enumerate(group, start=1):
            for ext, sub in (('.png','images'), ('.json','json'), ('.txt','labels')):
                src = os.path.join(src_dir, base + ext)
                dst = os.path.join(out_base, sub, base + ext)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            print(f"PROGRESS {idx}/{total}")

    # do it!
    print(f"Splitting {len(train_bases)} train, {len(val_bases)} val ({split_pct}% train)…")
    copy_group(train_bases, base_train, "Train")
    copy_group(val_bases,   base_val,   "Val")
    print("Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src_dir',   required=True,
                   help="Folder holding .png, .json, .txt triples")
    p.add_argument('--pct', type=int, default=80,
                   help="Percent of data to assign to train set")
    args = p.parse_args()
    split_real(args.src_dir, args.pct)

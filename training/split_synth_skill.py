#!/usr/bin/env python3
import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Split synthetic skill-icon data into train/val with progress markers"
    )
    parser.add_argument(
        "--pct", type=float, default=80,
        help="Fraction (0–1) of images to put into the train split"
    )
    args = parser.parse_args()
    TRAIN_PCT = args.pct / 100

    SRC_IMG  = "data/yolo/synth_skill/images"    # where your .png images live
    SRC_LBL  = "data/yolo/synth_skill/labels"    # where your .txt annotations live
    DST_ROOT = "data/yolo/synth_skill"           # base output folder

    # ── Clear previous splits ───────────────────────────────────────────────
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = os.path.join(DST_ROOT, split, kind)
            if os.path.isdir(path):
                shutil.rmtree(path)

    # Make train/val dirs
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(DST_ROOT, split, kind), exist_ok=True)

    # Gather and shuffle
    files = [f for f in os.listdir(SRC_IMG) if f.lower().endswith(".png")]
    random.shuffle(files)

    # Split
    total = len(files)
    cut = int(total * TRAIN_PCT)
    splits = {
        "train": files[:cut],
        "val":   files[cut:]
    }

    total = len(files)
    n_train, n_val = len(splits['train']), len(splits['val'])

    # ── 5) Initial summary for GUI log ─────────────────────────────────────────
    print(f"Splitting {total} files -> {n_train} train, {n_val} val.")

    # Copy and emit progress
    done = 0
    for split, fnames in splits.items():
        for fn in fnames:
            # copy image
            shutil.copy(
                os.path.join(SRC_IMG, fn),
                os.path.join(DST_ROOT, split, "images", fn)
            )
            # copy label
            lbl = os.path.splitext(fn)[0] + ".txt"
            src_lbl = os.path.join(SRC_LBL, lbl)
            dst_lbl = os.path.join(DST_ROOT, split, "labels", lbl)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()

            # emit progress marker for GUI
            done += 1
            print(f"PROGRESS {done}/{total}", flush=True)

    # Final summary
    print(
        f"Finished splitting {total} images -> "
        f"{len(splits['train'])} train / {len(splits['val'])} val"
    )

if __name__ == "__main__":
    main()

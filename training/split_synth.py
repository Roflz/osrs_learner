#!/usr/bin/env python3
import os
import random
import shutil
import argparse

def main():
    p = argparse.ArgumentParser(
        description="Split synthetic-number crops into train/val by scene prefix"
    )
    p.add_argument(
        '--pct', '-p',
        type=int,
        default=80,
        help="Percent of *scenes* (prefix groups) to put into train"
    )
    args = p.parse_args()
    TRAIN_PCT = args.pct / 100.0

    SRC_IMG   = "data/yolo/synth_numbers/images"
    SRC_LBL   = "data/yolo/synth_numbers/labels"
    DST_ROOT  = "data/yolo/synth_numbers"

    # ── Clear previous splits ───────────────────────────────────────────────
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = os.path.join(DST_ROOT, split, kind)
            if os.path.isdir(path):
                shutil.rmtree(path)

    # 1) Make output dirs
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            outdir = os.path.join(DST_ROOT, split, kind)
            os.makedirs(outdir, exist_ok=True)

    # 2) Gather all crop‐image filenames
    all_files = [f for f in os.listdir(SRC_IMG) if f.lower().endswith(".png")]

    # 3) Group by scene‐prefix (portion before the first '_')
    groups = {}
    for fn in all_files:
        prefix = fn.split('_', 1)[0]
        groups.setdefault(prefix, []).append(fn)

    # 4) Shuffle prefixes and split them
    prefixes = list(groups.keys())
    random.shuffle(prefixes)
    cut = int(len(prefixes) * TRAIN_PCT)
    train_prefixes = set(prefixes[:cut])
    val_prefixes   = set(prefixes[cut:])

    total_scenes = len(prefixes)
    print(f"Splitting {total_scenes} scenes -> "
          f"{len(train_prefixes)} train, {len(val_prefixes)} val.")

    # 5) Copy all crops & labels, reporting PROGRESS per file
    total_files = len(all_files)
    done = 0
    for prefix, fns in groups.items():
        split = "train" if prefix in train_prefixes else "val"
        for fn in fns:
            # copy image
            shutil.copy(
                os.path.join(SRC_IMG, fn),
                os.path.join(DST_ROOT, split, "images", fn)
            )
            # copy corresponding label
            lbl = fn[:-4] + ".txt"
            shutil.copy(
                os.path.join(SRC_LBL, lbl),
                os.path.join(DST_ROOT, split, "labels", lbl)
            )

            done += 1
            print(f"PROGRESS {done}/{total_files}", flush=True)

    # 6) Final summary
    print(
        f"Finished splitting {total_files} images -> "
        f"{len(train_prefixes)} scene-groups in train, "
        f"{len(val_prefixes)} scene-groups in val."
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Synthetic Skill-Icon Data Generator for YOLO+CRNN with
optional inverse-frequency sampling, console instrumentation, and
CLI progress markers, with neatly formatted side-by-side summary tables
and final distribution table of generated skills.
"""
import os
import random
import argparse
import shutil
import time
from collections import Counter
from itertools import zip_longest
from PIL import Image, ImageEnhance

# ─── DEFAULT CONFIG ─────────────────────────────────────────────────────────
DEFAULT_SKILL_ICON_DIR = "data/skill_icons_clean"   # input PNGs
DEFAULT_BG_IMG_DIR     = "data/backgrounds"         # backgrounds
DEFAULT_OUT_IMG_DIR    = "data/yolo/synth_skill/images"
DEFAULT_OUT_ANN_DIR    = "data/yolo/synth_skill/labels"
REAL_LABEL_DIRS        = [
    "data/yolo/real/train/labels",
    "data/yolo/real/val/labels",
]
DEFAULT_NUM_IMAGES     = 8000
DEFAULT_MAX_ICONS      = 5
DEFAULT_SEED           = 42
# ─── END CONFIG ─────────────────────────────────────────────────────────────

def load_real_counts(label_dirs, skip_classes=None):
    counts = Counter()
    skip = set(skip_classes or [])
    for ld in label_dirs:
        for fn in os.listdir(ld):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(ld, fn)) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid in skip:
                        continue
                    counts[cid] += 1
    return counts

def get_table_lines(headers, rows):
    # Build formatted table lines
    col_widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) + 2
        for i in range(len(headers))
    ]
    header = " | ".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers)))
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        line = " | ".join(str(r[i]).center(col_widths[i]) for i in range(len(headers)))
        lines.append(line)
    lines.append(sep)
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic skill-icon images for YOLO training"
    )
    parser.add_argument("--num_images", "-n", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--max_icons", "-k", type=int, default=DEFAULT_MAX_ICONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_weight", type=float, default=None,
                        help="Optional cap on inverse-frequency weight to avoid extremes")
    parser.add_argument("--no_weights", action="store_true",
                        help="Use uniform sampling instead of inverse-frequency weights")
    args = parser.parse_args()

    use_weights = not args.no_weights
    max_weight  = args.max_weight

    # ── Clear previous outputs ─────────────────────────────────────────────
    if os.path.isdir(DEFAULT_OUT_IMG_DIR):
        shutil.rmtree(DEFAULT_OUT_IMG_DIR)
    if os.path.isdir(DEFAULT_OUT_ANN_DIR):
        shutil.rmtree(DEFAULT_OUT_ANN_DIR)

    print("=== Synth Skills ===")
    print(f"Using real-data labels: {REAL_LABEL_DIRS}")
    print(f"Generating {args.num_images} images; max_icons={args.max_icons}")
    print(f"{'Inverse-frequency' if use_weights else 'Uniform'} sampling of skills")
    print(f"Outputs -> images: {DEFAULT_OUT_IMG_DIR}, labels: {DEFAULT_OUT_ANN_DIR}\n")

    random.seed(args.seed)

    # 1) Load real-data counts
    real_counts = load_real_counts(REAL_LABEL_DIRS, skip_classes=[0])
    max_count   = max(real_counts.values()) if real_counts else 1

    # 2) Load skill names and icons
    skill_files = [f for f in os.listdir(DEFAULT_SKILL_ICON_DIR) if f.lower().endswith(".png")]
    skills      = sorted(os.path.splitext(f)[0] for f in skill_files)
    skill_icons = {
        name: Image.open(os.path.join(DEFAULT_SKILL_ICON_DIR, name + ".png")).convert("RGBA")
        for name in skills
    }

    # 3) Compute sampling weights
    if use_weights:
        skill_weights = []
        for idx, name in enumerate(skills):
            cid = idx + 1
            cnt = real_counts.get(cid, 0)
            weight = max_count / (cnt if cnt > 0 else 1)
            if max_weight is not None and weight > max_weight:
                weight = max_weight
            skill_weights.append(weight)
    else:
        skill_weights = [1.0] * len(skills)

    # Summary tables side by side
    headers1 = ["Skill", "Count", "Weight"]
    rows1 = []
    for idx, name in enumerate(skills):
        cid = idx + 1
        cnt = real_counts.get(cid, 0)
        w   = skill_weights[idx]
        rows1.append([name, cnt, f"{w:.2f}"])
    lines1 = get_table_lines(headers1, rows1)

    headers2 = ["Skill", "Weight"]
    rows2 = [[name, f"{skill_weights[i]:.2f}"] for i, name in enumerate(skills)]
    lines2 = get_table_lines(headers2, rows2)

    print("Summary tables:\n")
    for l1, l2 in zip_longest(lines1, lines2, fillvalue=" " * len(lines1[0])):
        print(l1 + "   " + l2)
    print()

    # 4) Gather backgrounds
    bg_files = [
        os.path.join(DEFAULT_BG_IMG_DIR, f)
        for f in os.listdir(DEFAULT_BG_IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not bg_files:
        raise RuntimeError(f"No backgrounds found in {DEFAULT_BG_IMG_DIR}")

    # 5) Prepare outputs
    os.makedirs(DEFAULT_OUT_IMG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUT_ANN_DIR, exist_ok=True)

    # 6) Generation loop
    start = time.time()
    generated_counts = Counter()
    for i in range(args.num_images):
        bg = Image.open(random.choice(bg_files)).convert("RGBA")
        bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.8, 1.2))
        bg = ImageEnhance.Contrast(bg).enhance(random.uniform(0.8, 1.2))
        W, H = bg.size

        ann_lines = []
        num_icons = random.randint(0, args.max_icons)
        chosen    = random.choices(skills, weights=skill_weights, k=num_icons)
        for name in chosen:
            cid = skills.index(name) + 1
            generated_counts[name] += 1
            icon = skill_icons[name]
            if use_weights and real_counts.get(cid, 0) < max_count * 0.5:
                icon = ImageEnhance.Brightness(icon).enhance(random.uniform(0.7,1.3))
                icon = icon.rotate(random.uniform(-10,10), resample=Image.BILINEAR, expand=True)
            x = random.randint(0, W - icon.width)
            y = random.randint(0, H - icon.height)
            bg.paste(icon, (x, y), icon)
            xc = (x + icon.width/2) / W
            yc = (y + icon.height/2) / H
            w_ = icon.width / W
            h_ = icon.height / H
            ann_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}")

        img_name = f"{i:06d}.png"
        bg.convert("RGB").save(os.path.join(DEFAULT_OUT_IMG_DIR, img_name))
        with open(os.path.join(DEFAULT_OUT_ANN_DIR, img_name.replace(".png", ".txt")), "w") as f:
            if ann_lines:
                f.write("\n".join(ann_lines))

        # emit CLI progress marker
        print(f"PROGRESS {i+1}/{args.num_images}", flush=True)

    elapsed = time.time() - start

    # Final distribution table
    headers3 = ["Skill", "Total"]
    rows3    = [[name, generated_counts.get(name, 0)] for name in skills]
    lines3   = get_table_lines(headers3, rows3)
    print("\n=== Generated Skill Distribution ===")
    for line in lines3:
        print(line)

    print(f"Done! Generated {args.num_images} images in {elapsed:.1f}s.")

if __name__ == "__main__":
    main()

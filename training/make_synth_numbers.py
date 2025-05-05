#!/usr/bin/env python3
"""
Synthetic Number-Block Generator for YOLO+CRNN with
inverse-frequency digit sampling (optional), console instrumentation,
and automatic real‑data JSON parsing—all JSON, TXT, PNG
in the same directory.
"""
import os
import random
import csv
import json
import io
import shutil
import time
import argparse
from collections import Counter
from PIL import Image, ImageEnhance

# ─── DEFAULTS ───────────────────────────────────────────────────────────────
DEFAULT_REAL_JSON_DIR = "data/xp_crops_labeled"
DEFAULT_TEMPLATES_DIR = "data/digits"
DEFAULT_BACK_DIR      = "data/backgrounds"
DEFAULT_OUT_IMG_DIR   = "data/yolo/synth_numbers/images"
DEFAULT_OUT_ANN_DIR   = "data/yolo/synth_numbers/labels"
DEFAULT_MAP_CSV       = "data/yolo/synth_numbers/synth_map.csv"

DEFAULT_NUM_IMAGES    = 8000
DEFAULT_NEG_RATIO     = 0.2
DEFAULT_MIN_SEQ       = 1
DEFAULT_MAX_SEQ       = 5
DEFAULT_MIN_DIGITS    = 1
DEFAULT_MAX_DIGITS    = 5
# ─── END DEFAULTS ───────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_json_dir", default=DEFAULT_REAL_JSON_DIR)
    p.add_argument("--templates_dir", default=DEFAULT_TEMPLATES_DIR)
    p.add_argument("--back_dir",      default=DEFAULT_BACK_DIR)
    p.add_argument("--out_img_dir",   default=DEFAULT_OUT_IMG_DIR)
    p.add_argument("--out_ann_dir",   default=DEFAULT_OUT_ANN_DIR)
    p.add_argument("--map_csv",       default=DEFAULT_MAP_CSV)
    p.add_argument("--num_images",   type=int,   default=DEFAULT_NUM_IMAGES)
    p.add_argument("--neg_ratio",    type=float, default=DEFAULT_NEG_RATIO)
    p.add_argument("--min_seq",      type=int,   default=DEFAULT_MIN_SEQ)
    p.add_argument("--max_seq",      type=int,   default=DEFAULT_MAX_SEQ)
    p.add_argument("--min_digits",   type=int,   default=DEFAULT_MIN_DIGITS)
    p.add_argument("--max_digits",   type=int,   default=DEFAULT_MAX_DIGITS)

    # weighting flags
    p.add_argument("--max_weight",  type=float, default=None,
                   help="Optional cap on inverse-frequency weight")
    p.add_argument("--no_weights",  action="store_true",
                   help="Use uniform sampling instead of inverse-frequency")
    return p.parse_args()


def main():
    args = parse_args()

    REAL_JSON_DIR = args.real_json_dir
    TEMPLATES_DIR = args.templates_dir
    BACK_DIR      = args.back_dir
    OUT_IMG_DIR   = args.out_img_dir
    OUT_ANN_DIR   = args.out_ann_dir
    MAP_CSV       = os.path.abspath(args.map_csv)

    NUM_IMAGES    = args.num_images
    NEG_RATIO     = args.neg_ratio
    MIN_SEQ, MAX_SEQ = args.min_seq, args.max_seq
    DIGIT_LEN     = (args.min_digits, args.max_digits)

    use_weights = not args.no_weights
    max_weight  = args.max_weight

    # ── Clear previous outputs ─────────────────────────────────────────────
    # remove only the specific synth_numbers dirs and CSV parent
    for path in (OUT_IMG_DIR, OUT_ANN_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path)
    csv_dir = os.path.dirname(MAP_CSV)
    if os.path.isdir(csv_dir):
        shutil.rmtree(csv_dir)

    # recreate fresh output dirs
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_ANN_DIR, exist_ok=True)
    os.makedirs(csv_dir,   exist_ok=True)

    random.seed(42)
    print(f"Using JSON from: {REAL_JSON_DIR}")
    print(f"Generating {NUM_IMAGES} images; neg_ratio={NEG_RATIO}, seq={MIN_SEQ}-{MAX_SEQ}, digits={DIGIT_LEN}")
    print(f"{'Inverse-frequency' if use_weights else 'Uniform'} sampling; max_weight={max_weight}")
    print(f"Writing CSV: {MAP_CSV}\n")

    # 1) load real counts
    digit_counts = Counter()
    for fn in os.listdir(REAL_JSON_DIR):
        if not fn.lower().endswith(".json"):
            continue
        data = json.load(open(os.path.join(REAL_JSON_DIR, fn)))
        for xp in data.get("xp_values", []):
            for ch in xp:
                if ch.isdigit():
                    digit_counts[int(ch)] += 1
    max_d = max(digit_counts.values()) if digit_counts else 1

    # 2) build weights
    digit_weights = []
    for d in range(10):
        if use_weights:
            w = max_d / (digit_counts.get(d, 0) or 1)
            if max_weight is not None and w > max_weight:
                w = max_weight
        else:
            w = 1.0
        digit_weights.append(w)

    print("Digit | RealCount |    Weight", flush=True)
    print("-----------------------------", flush=True)
    for d in range(10):
        print(f"  {d:<2}  | {digit_counts.get(d,0):9d} | {digit_weights[d]:8.2f}", flush=True)
    print(flush=True)

    # 3) load templates and backgrounds
    templates = {
        str(d): Image.open(os.path.join(TEMPLATES_DIR, f"{d}_masked.png")).convert("RGBA")
        for d in range(10)
    }
    bg_files = [
        os.path.join(BACK_DIR, f)
        for f in os.listdir(BACK_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    assert bg_files, f"No backgrounds in {BACK_DIR}"

    sampled_digits = Counter()
    total_sequences = 0
    start = time.time()

    # 4) CSV writer
    with open(MAP_CSV, "w", newline="") as map_f:
        writer = csv.writer(map_f)
        writer.writerow(["filename", "sequence"])

        # 5) generation loop
        for i in range(NUM_IMAGES):
            bg = Image.open(random.choice(bg_files)).convert("RGBA")
            bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.8, 1.2))
            bg = ImageEnhance.Contrast(bg).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                buf = io.BytesIO()
                bg.convert("RGB").save(buf, format="JPEG", quality=random.randint(50, 90))
                buf.seek(0)
                bg = Image.open(buf).convert("RGBA")
            W, H = bg.size

            annots = []
            block_idx = 0

            if random.random() > NEG_RATIO:
                K = random.randint(MIN_SEQ, MAX_SEQ)
                total_sequences += K
                for _ in range(K):
                    L = random.randint(*DIGIT_LEN)
                    seq = random.choices([str(d) for d in range(10)], weights=digit_weights, k=L)
                    sampled_digits.update(int(ch) for ch in seq)

                    # assemble the block
                    widths = [templates[ch].width for ch in seq]
                    hgt = max(templates[ch].height for ch in seq)
                    block = Image.new("RGBA", (sum(widths), hgt), (0, 0, 0, 0))
                    xoff = 0
                    for ch in seq:
                        dig = templates[ch]
                        yoff = (hgt - dig.height) // 2
                        block.paste(dig, (xoff, yoff), dig)
                        xoff += dig.width

                    bx = random.randint(0, W - block.width)
                    by = random.randint(0, H - block.height)
                    bg.paste(block, (bx, by), block)

                    xc = (bx + block.width / 2) / W
                    yc = (by + block.height / 2) / H
                    nw = block.width / W
                    nh = block.height / H
                    annots.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                    # write this crop to CSV
                    fn_crop = f"{i:05d}_{block_idx:02d}.png"
                    writer.writerow([fn_crop, "".join(seq)])
                    block_idx += 1

            # save image + label
            img_name = f"{i:05d}.png"
            bg.convert("RGB").save(os.path.join(OUT_IMG_DIR, img_name))
            with open(os.path.join(OUT_ANN_DIR, img_name.replace(".png", ".txt")), "w") as f:
                if annots:
                    f.write("\n".join(annots))

            # progress
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (NUM_IMAGES - (i + 1))
            print(f"PROGRESS {i+1}/{NUM_IMAGES} ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}", flush=True)

    # 6) summary
    total_digits = sum(sampled_digits.values())
    avg_per_block = total_digits / total_sequences if total_sequences else 0
    avg_per_img = total_digits / NUM_IMAGES
    print(f"\nWrote CSV to {MAP_CSV}")
    print(f"Images: {NUM_IMAGES}, Blocks: {total_sequences} ({avg_per_img:.2f} digits/img)")
    print(f"Total digits: {total_digits} ({avg_per_block:.2f} per block)")
    print("\nPer-digit breakdown:")
    for d in range(10):
        print(f" {d}: {sampled_digits[d]}")
    print("Done.")


if __name__ == "__main__":
    main()

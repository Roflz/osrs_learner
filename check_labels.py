#!/usr/bin/env python3
import os
import json
import sys
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# ── adjust these paths to wherever your models live ──
YOLO_MODEL_PATH = "models/best.pt"
CRNN_MODEL_PATH = "models/crnn.pt"
CONF_THRESH     = 0.9    # if avg confidence ≥ this, flag as HIGH‑CONF mismatch

# skill names → must match your YOLO training “names:” order, e.g.:
SKILL_NAMES = [
    "drop","agility","attack","construction","cooking","crafting","defence",
    "farming","firemaking","fishing","fletching","herblore","hitpoints","hunter",
    "magic","mining","prayer","ranged","runecrafting","smithing","slayer",
    "strength","thieving",
]

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO once
yolo_model = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo_model.names

# Load your CRNN (TorchScript) once
crnn_model = torch.jit.load(CRNN_MODEL_PATH, map_location=DEVICE).eval()
PAD_TOKEN = 10

# OCR preprocessing for CRNN
ocr_transform = transforms.Compose([
    transforms.Resize((64,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ANSI color codes for terminal highlighting
RED    = '\033[91m'
YELLOW = '\033[93m'
RESET  = '\033[0m'

# — prediction‐check toggles —
USE_PRED_DROP_COUNT = True    # compare predicted vs JSON drop_count
USE_PRED_XP_VALUES   = True   # compare predicted vs JSON xp_values
USE_PRED_SKILLS      = False  # compare predicted vs JSON skills (disabled)

EXTS = ['.json', '.txt', '.png']

def predict_labels_for_shot(png_path, conf_threshold=CONF_THRESH):
    """
    Runs YOLO → detects xp boxes (class=0) and skill icons (class>0),
    then runs CRNN on each xp‐crop. Returns a dict:
      - drop_count:       int
      - xp_values:        [str,…]
      - skills:           [str,…]
      - xp_confs:         [float,…]
      - skill_confs:      [float,…]
    """
    img = Image.open(png_path)
    res = yolo_model(img, conf=conf_threshold, verbose=False, show=False)[0]
    boxes = res.boxes
    cls   = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xywh  = boxes.xywh.cpu().numpy()

    # filter by confidence if you like:
    xp_idxs    = [i for i,c in enumerate(cls) if c == 0 and confs[i] >= CONF_THRESH]
    skill_idxs = [i for i,c in enumerate(cls) if c > 0 and confs[i] >= CONF_THRESH]

    drop_count = len(xp_idxs)

    xp_values = []
    xp_confs  = []
    for i in xp_idxs:
        cx,cy,w,h = xywh[i]
        x1,y1 = int(cx-w/2), int(cy-h/2)
        x2,y2 = int(cx+w/2), int(cy+h/2)
        crop = img.crop((x1,y1,x2,y2)).convert("L")

        # prepare tensor
        t = ocr_transform(crop).unsqueeze(0).to(DEVICE)

        # get logits from CRNN
        with torch.no_grad():
            logits = crnn_model(t)     # shape (1, T, C)

        # best‑path CTC decode:
        #  1) take argmax at each time step
        #  2) collapse repeats
        #  3) drop PAD_TOKEN (our “blank”)
        pred_indices = logits.argmax(-1)[0].cpu().tolist()
        decoded = []
        prev = None
        for idx in pred_indices:
            if idx == prev:
                continue
            prev = idx
            if idx == PAD_TOKEN:
                break
            decoded.append(str(idx))

        val = "".join(decoded)
        xp_values.append(val)
        xp_confs.append(float(confs[i]))

    # map skill classIDs → names
    skills      = [CLASS_NAMES.get(c, f"cls_{c}") for c in cls[skill_idxs]]
    skill_confs = [float(confs[i]) for i in skill_idxs]

    return {
        "drop_count":  drop_count,
        "xp_values":   xp_values,
        "skills":      skills,
        "xp_confs":    xp_confs,
        "skill_confs": skill_confs
    }

def check_label_file(json_path):
    """
    Returns (parse_error, content_errors) tuple:
    - parse_error: a string if JSON failed to load or is empty, else None
    - content_errors: list of mismatch or missing‐file messages
    """
    base, _   = os.path.splitext(json_path)
    txt_path  = base + ".txt"
    png_path  = base + ".png"
    errors    = []

    # 0) Check for matching PNG
    if not os.path.isfile(png_path):
        errors.append("missing matching .png file")

    # 1) Check JSON is non‐empty on disk
    try:
        if os.path.getsize(json_path) == 0:
            return ("empty JSON file", [])
    except OSError as e:
        return (f"cannot access JSON file: {e}", [])

    # 2) Load & validate JSON syntax
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return (f"JSON parse error: {e}", [])

    # 2.a) Flag truly empty content as a parse‐error
    if not data:
        return ("empty JSON content `{}`", [])

    # 3) Perform the usual content checks
    drop_ct     = data.get('drop_count')
    xp_vals     = data.get('xp_values', [])
    skills      = data.get('skills', [])
    boxes       = data.get('boxes', [])

    # count classes in JSON
    drop_boxes  = sum(1 for b in boxes if b.get('class') == 0)
    skill_boxes = sum(1 for b in boxes if b.get('class', 0) > 0)

    # JSON content checks
    if drop_ct is None:
        errors.append("missing drop_count in JSON")
    else:
        if len(xp_vals) != drop_ct:
            errors.append(f"xp_values: {len(xp_vals)} ≠ drop_count: {drop_ct}")
        if len(skills) != drop_ct:
            errors.append(f"skills: {len(skills)} ≠ drop_count: {drop_ct}")
        if drop_boxes != drop_ct:
            errors.append(f"JSON boxes class=0: {drop_boxes} ≠ drop_count: {drop_ct}")
        if skill_boxes != drop_ct:
            errors.append(f"JSON boxes class>0: {skill_boxes} ≠ drop_count: {drop_ct}")

    # 4) TXT match & counts
    if not os.path.isfile(txt_path):
        errors.append("missing matching .txt file")
    else:
        txt_drop = txt_skill = 0
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    errors.append(f".txt parse error line {line_no}: bad class '{parts[0]}'")
                    continue
                if cls == 0:
                    txt_drop += 1
                else:
                    txt_skill += 1

        # compare TXT counts to drop_count
        if drop_ct is not None:
            if txt_drop != drop_ct:
                errors.append(f".txt class=0 lines: {txt_drop} ≠ drop_count: {drop_ct}")
            if txt_skill != drop_ct:
                errors.append(f".txt class>0 lines: {txt_skill} ≠ drop_count: {drop_ct}")
        # cross-check with JSON box counts
        if txt_drop != drop_boxes:
            errors.append(f".txt class=0 lines: {txt_drop} ≠ JSON boxes class=0: {drop_boxes}")
        if txt_skill != skill_boxes:
            errors.append(f".txt class>0 lines: {txt_skill} ≠ JSON boxes class>0: {skill_boxes}")

    # ── 5) Prediction‑based sanity check ──
    if USE_PRED_DROP_COUNT or USE_PRED_XP_VALUES or USE_PRED_SKILLS:
        try:
            pred = predict_labels_for_shot(png_path)

            # 5.a) Gather only high‑confidence XP detections
            high_xp = [
                (val, conf)
                for val, conf in zip(pred["xp_values"], pred["xp_confs"])
                if conf >= CONF_THRESH
            ]
            xp_vals_high = [val for val, _ in high_xp]

            # 5.b) Gather only high‑confidence skill detections (if enabled)
            high_skills = []
            if USE_PRED_SKILLS:
                high_skills = [
                    (s, c)
                    for s, c in zip(pred["skills"], pred["skill_confs"])
                    if c >= CONF_THRESH
                ]
                skills_high = [s for s, _ in high_skills]

            # 5.c) Compare ground truth vs filtered predictions
            mismatches = []
            if USE_PRED_DROP_COUNT and len(xp_vals_high) != drop_ct:
                mismatches.append(f"drop_count {len(xp_vals_high)}≠{drop_ct}")
            if USE_PRED_XP_VALUES and xp_vals_high != [str(x) for x in xp_vals]:
                mismatches.append(f"xp_values {xp_vals_high}≠{xp_vals}")
            if USE_PRED_SKILLS and skills_high != skills:
                mismatches.append(f"skills {skills_high}≠{skills}")

            # 5.d) Only report if there *are* mismatches AND
            #     every confidence used is ≥ CONF_THRESH
            if mismatches:
                # collect all confidences we actually compared
                confs_used = [c for _, c in high_xp]
                if USE_PRED_SKILLS:
                    confs_used += [c for _, c in high_skills]

                # ensure no low‑confidence got in
                if confs_used and all(c >= CONF_THRESH for c in confs_used):
                    errors.append("PREDICT‑HIGH‑CONF MISMATCH: " + "; ".join(mismatches))

        except Exception as e:
            errors.append(f"PREDICTION ERROR: {e}")

    return (None, errors)

def find_problems(folder):
    parse_failures = []
    content_failures = []

    # Gather all JSON filenames
    json_files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.json'))
    total = len(json_files)

    # 1) Cross‑file existence check for .json, .txt, .png
    basenames = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(folder)
        if os.path.splitext(fn)[1].lower() in EXTS
    }
    for base in sorted(basenames):
        missing = [
            ext for ext in EXTS
            if not os.path.isfile(os.path.join(folder, base + ext))
        ]
        if missing:
            parse_failures.append((
                base,
                f"missing file(s): {', '.join(missing)}"
            ))

    # 2) JSON‑based checks, with live progress
    for idx, fn in enumerate(json_files, 1):
        # Overwrite the same line with progress
        print(f"[{idx}/{total}] Checking {fn}...", end='\r', flush=True)

        path = os.path.join(folder, fn)
        parse_err, errs = check_label_file(path)
        if parse_err:
            parse_failures.append((fn, parse_err))
        elif errs:
            content_failures.append((fn, errs))

    # Finally move down a line so subsequent prints don’t overwrite
    print()

    return parse_failures, content_failures

def prompt_choice(prompt, choices):
    """
    Prompt until the user types one of the choices (case-insensitive).
    Returns the choice in lowercase.
    """
    choices = [c.lower() for c in choices]
    choice = ''
    while True:
        choice = input(f"{prompt} [{'/'.join(choices)}]: ").strip().lower()
        if choice in choices:
            return choice
        print(f"Please choose one of: {', '.join(choices)}")

def delete_files(folder, basename):
    """
    Remove .json, .txt, and .png for the given basename in folder.
    """
    for ext in EXTS:
        p = os.path.join(folder, basename + ext)
        if os.path.isfile(p):
            try:
                os.remove(p)
                print(f"  deleted {basename + ext}")
            except Exception as e:
                print(f"  failed to delete {basename + ext}: {e}")

def main(folder):
    parse_failures, content_failures = find_problems(folder)
    # build quick lookup
    parse_map   = {fn: msg for fn, msg in parse_failures}
    content_map = {fn: errs for fn, errs in content_failures}

    error_files = list(parse_map.keys()) + list(content_map.keys())

    # --- initial reporting (same as before) ---
    if parse_failures:
        print("\nJSON PARSE/EMPTY ERRORS:")
        for fn, msg in parse_failures:
            print(f"  • {fn}: {msg}")
    if content_failures:
        print("\nCONTENT MISMATCHES:")
        for fn, errs in content_failures:
            print(f"  • {fn}:")
            for e in errs:
                print(f"      - {e}")
    total_files = len([f for f in os.listdir(folder) if f.lower().endswith('.json')])
    print(f"\nSummary: {len(error_files)} files with errors, {total_files - len(error_files)} OK.\n")

    if not error_files:
        return

    choice = prompt_choice(
        "(r) Remove all error files?\n"
        "(s) Step through individually?\n"
        "(x) Exit?",
        ['r', 's', 'x']
    )

    if choice == 'x':
        print("Exiting without deleting.")
        return
    if choice == 'r':
        print("Deleting all error files:")
        for fn in error_files:
            base, _ = os.path.splitext(fn)
            delete_files(folder, base)
        return

    # — Step through individually —
    for fn in error_files:
        base      = os.path.splitext(fn)[0]
        png_path  = os.path.join(folder, base + '.png')
        json_path = os.path.join(folder, fn)

        # 1) Show the PNG
        if os.path.isfile(png_path):
            try:
                Image.open(png_path).show()
            except Exception as e:
                print(f"{RED}[!] Failed to open image {png_path}: {e}{RESET}")
        else:
            print(f"{RED}[!] PNG not found: {png_path}{RESET}")

        # 2) Load & pretty‑print JSON, highlighting bad keys
        print(f"\n── JSON content for {fn} ──")
        try:
            raw = open(json_path, 'r', encoding='utf-8').read()
            data = json.loads(raw)
        except Exception as e:
            print(f"{RED}Cannot parse JSON: {e}{RESET}")
            print(raw)
        else:
            pretty = json.dumps(data, indent=2)
            errs   = content_map.get(fn, [])

            # figure out which JSON keys to highlight
            highlight_keys = set()
            for err in errs:
                if "drop_count" in err:
                    highlight_keys.add('"drop_count"')
                if err.startswith("xp_values"):
                    highlight_keys.add('"xp_values"')
                if err.startswith("skills"):
                    highlight_keys.add('"skills"')

            # print with highlights
            for line in pretty.splitlines():
                if any(k in line for k in highlight_keys):
                    print(f"{YELLOW}{line}{RESET}")
                else:
                    print(line)

            # finally, print the error messages
            if errs:
                print(f"\nErrors for {fn}:")
                for e in errs:
                    print(f"  {RED}- {e}{RESET}")

        # 3) Prompt to delete
        answer = prompt_choice(f"\nDelete {fn} and its .txt/.png?", ['y', 'n', 'x'])
        if answer == 'x':
            print("Exiting early.")
            return
        if answer == 'y':
            delete_files(folder, base)

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_labels.py /path/to/labeled_shots_folder")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}")
        sys.exit(1)

    main(folder)

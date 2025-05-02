#!/usr/bin/env python3
import os
import json
import sys

EXTS = ['.json', '.txt', '.png']

def check_label_file(json_path):
    """
    Returns (parse_error, content_errors) tuple:
    - parse_error: a string if JSON failed to load, else None
    - content_errors: list of mismatch messages
    """
    base, _ = os.path.splitext(json_path)
    txt_path = base + ".txt"
    errors = []

    # 1) Load & validate JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return (f"JSON parse error: {e}", [])

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

    # 2) TXT match & counts
    if not os.path.isfile(txt_path):
        errors.append("missing matching .txt file")
    else:
        txt_drop = txt_skill = 0
        with open(txt_path, 'r') as f:
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

    return (None, errors)

def find_problems(folder):
    parse_failures = []
    content_failures = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.json'):
            continue
        path = os.path.join(folder, fn)
        parse_err, errs = check_label_file(path)
        if parse_err:
            parse_failures.append((fn, parse_err))
        elif errs:
            content_failures.append((fn, errs))
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
    error_files = [fn for fn, _ in parse_failures] + [fn for fn, _ in content_failures]

    # Reporting
    if parse_failures:
        print("\nJSON PARSE ERRORS:")
        for fn, msg in parse_failures:
            print(f"  • {fn}: {msg}")

    if content_failures:
        print("\nCONTENT MISMATCHES:")
        for fn, errs in content_failures:
            print(f"  • {fn}:")
            for e in errs:
                print(f"      - {e}")

    total_files = len([f for f in os.listdir(folder) if f.lower().endswith('.json')])
    print(f"\nSummary: {len(error_files)} files with errors, {total_files - len(error_files)} OK.")

    if not error_files:
        return

    # Prompt user for deletion strategy
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

    # Step through each error file
    print("\nStep through each error file:")
    for fn in error_files:
        base, _ = os.path.splitext(fn)
        c = prompt_choice(f"Delete {fn} and its .txt/.png?", ['y', 'n', 'x'])
        if c == 'x':
            print("Exiting early.")
            return
        if c == 'y':
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

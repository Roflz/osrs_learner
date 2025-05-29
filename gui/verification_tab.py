import os
import json
import glob
import yaml
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class VerificationTab(ttk.Frame):
    """Tab that verifies JSON↔TXT↔skills consistency for labeled crops."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._build_ui()

    def _build_ui(self):
        ttk.Label(self, text="Verification Tab", font=("Helvetica", 16, "bold")) \
            .pack(padx=20, pady=(20, 10), anchor="w")

        btn = ttk.Button(self, text="Run Verification", command=self._on_verify)
        btn.pack(padx=20, pady=(0,10), anchor="w")

        # add a scrolled text widget to display errors & stats
        self.log = scrolledtext.ScrolledText(self, width=80, height=20, state='disabled')
        self.log.pack(padx=20, pady=(0,20), fill='both', expand=True)

    def _on_verify(self):
        LABELED_DIR = 'data/xp_crops_labeled'
        YAML_FILE   = 'data.yaml'

        # load names map
        try:
            with open(YAML_FILE, 'r') as yf:
                cfg = yaml.safe_load(yf)
            names_map = cfg.get('names', {})
        except Exception as e:
            messagebox.showerror("Verification Error", f"Failed to load {YAML_FILE}: {e}")
            return

        errors = []
        checked = 0

        for json_path in glob.glob(os.path.join(LABELED_DIR, '*.json')):
            stem = os.path.splitext(os.path.basename(json_path))[0]
            txt_path = os.path.join(LABELED_DIR, stem + '.txt')
            checked += 1

            # load JSON
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                errors.append((stem, f'JSON load error: {e}'))
                continue

            # 0) drop_count vs actual zeros
            exp = data.get('drop_count', 0)
            real0 = sum(1 for b in data.get('boxes',[]) if b.get('class')==0)
            if real0 != exp:
                errors.append((stem, f'drop_count mismatch: {exp} vs {real0}'))

            # collect JSON IDs
            json_ids = {b.get('class') for b in data.get('boxes',[]) if isinstance(b.get('class'), int)}

            # load TXT
            if not os.path.isfile(txt_path):
                errors.append((stem, 'Missing TXT file'))
                continue
            txt_ids = set(); txt0 = 0
            with open(txt_path, 'r', encoding='utf-8') as tf:
                for ln in tf:
                    parts = ln.strip().split()
                    if parts and parts[0].isdigit():
                        cid = int(parts[0])
                        txt_ids.add(cid)
                        if cid==0: txt0 += 1

            # 1) JSON vs TXT IDs
            if json_ids != txt_ids:
                errors.append((stem, f'ID mismatch: {sorted(json_ids)} vs {sorted(txt_ids)}'))
                continue

            # 2) skills mapping
            skill_ids = sorted(cid for cid in json_ids if cid!=0)
            mapped = {names_map.get(cid) for cid in skill_ids}
            js = set(data.get('skills', []))
            if mapped != js:
                errors.append((stem, f"Skills mismatch: {sorted(js)} vs {sorted(mapped)}"))

            # 3) TXT zeros vs drop_count
            if txt0 != exp:
                errors.append((stem, f'TXT zeros mismatch: {exp} vs {txt0}'))

        # now display
        self.log.config(state='normal')
        self.log.delete('1.0', tk.END)

        if errors:
            self.log.insert(tk.END, "❌ WARNING: Found inconsistencies:\n\n", 'err')
            for stem, msg in errors:
                self.log.insert(tk.END, f" • {stem}: {msg}\n")
            summary = f"\nChecked {checked} files, {len(errors)} errors.\n"
            self.log.insert(tk.END, summary)
            self.log.tag_config('err', foreground='red', font=('Helvetica', 12, 'bold'))
        else:
            self.log.insert(tk.END, f"✅ All {checked} files passed consistency checks.\n", 'ok')
            self.log.tag_config('ok', foreground='green', font=('Helvetica', 12, 'bold'))

        self.log.config(state='disabled')
        # still fail pipeline if you want:
        if errors:
            self.master.event_generate('<<VerificationFailed>>')

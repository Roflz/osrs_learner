import os
import json
import csv
from collections import Counter

from config import SKILLS


def color_for_ratio(r):
    if r < 0.5:
        r_, g = 200, int((r / 0.5) * 200)
    else:
        r_, g = int((1 - (r - 0.5) / 0.5) * 200), 200
    return f'#{r_:02x}{g:02x}00'


def make_table(st, table_id, rows, headers, max_vals):
    # Compute column widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    # Header style
    st.tag_configure('hdr', font=('Courier', 10, 'bold'), foreground='#ffffff')

    # Header row
    for h in headers:
        st.insert('end', h.ljust(widths[h]) + '  ', 'hdr')
    st.insert('end', '\n', 'hdr')

    # Separator row
    for h in headers:
        st.insert('end', '-' * widths[h] + '  ', 'hdr')
    st.insert('end', '\n', 'hdr')

    # Data rows
    for i, row in enumerate(rows, start=1):
        is_total = (str(row[headers[0]]) == 'Total')
        for j, h in enumerate(headers):
            text = str(row[h]).ljust(widths[h])
            if j == 0 or is_total:
                st.insert('end', text + '  ')
            else:
                val = row[h]
                ratio = (val / max_vals[h]) if max_vals[h] else 0
                fg = color_for_ratio(ratio)
                tag = f'{table_id}_cell_{h}_{i}'
                st.tag_configure(tag, foreground=fg)
                st.insert('end', text + '  ', tag)
        st.insert('end', '\n')


def _compute_synth_balance():
    """
    Count real vs. synthetic images and return
    (real_count, synth_count, needed_synth)
    """
    # Count real JSON labels
    real_dirs = [
        'data/yolo/real/train/json',
        'data/yolo/real/val/json',
    ]
    real_count = 0
    for d in real_dirs:
        if os.path.isdir(d):
            real_count += len([f for f in os.listdir(d) if f.lower().endswith('.json')])

    # Count synthetic images
    synth_dirs = [
        'data/yolo/synth_numbers/train/images',
        'data/yolo/synth_numbers/val/images',
    ]
    synth_count = 0
    for d in synth_dirs:
        if os.path.isdir(d):
            synth_count += len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))])

    # Compute how many more synthetic are needed
    needed = max(0, real_count - synth_count)
    return real_count, synth_count, needed


def update_stats(self):
    st = self.stats_text
    st.config(state='normal', font=('Courier', 10), bg='#1e1e1e', fg='#e0e0e0')
    st.delete('1.0', 'end')

    # --- Precompute Data Balance ---
    real_cnt, synth_cnt, need_cnt = _compute_synth_balance()

    # === SKILLS header & Data Balance in one line ===
    skill_headers = ['Skill', 'RealTrain', 'RealVal', 'RealTotal', 'SynthTrain', 'SynthVal', 'SynthTotal']
    # Compute a rough width for the Skills column block
    # We use header lengths plus two spaces each
    total_width = sum(len(h) for h in skill_headers) + 2 * len(skill_headers)

    # Print combined header
    st.tag_configure('hdr', font=('Courier', 10, 'bold'), foreground='#ffffff')
    st.insert('end', 'Skills'.ljust(total_width) + '  ' + 'Data Balance\n', 'hdr')

    # === Data Balance & Real Negative Ratio ===
    real_cnt, synth_cnt, need_cnt = _compute_synth_balance()
    pad = ' ' * (total_width + 2)

    # Data Balance
    st.insert('end', pad + f"Real images:      {real_cnt}\n")
    st.insert('end', pad + f"Synthetic images: {synth_cnt}\n")
    st.insert('end', pad + f"To match real:    generate {need_cnt}\n")

    # Real Negative Ratio (from xp_crops_labeled)
    zero_count = 0
    src_dir = 'data/xp_crops_labeled'
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.lower().endswith('.json'):
                data = json.load(open(os.path.join(src_dir, fn)))
                if data.get('drop_count', 0) == 0:
                    zero_count += 1
    total_real = real_cnt
    neg_ratio = (zero_count / total_real) if total_real else 0.0
    pct = neg_ratio * 100

    st.insert('end', pad + f"Zero‑drop images: {zero_count}/{total_real}\n")
    st.insert('end', pad + f"Negative ratio:   {pct:.2f}%\n\n")

    # === SKILLS table ===
    # Real train/val counts
    rt, rv = Counter(), Counter()
    for split, ctr in [
        ('data/yolo/real/train/json', rt),
        ('data/yolo/real/val/json', rv),
    ]:
        if os.path.isdir(split):
            for fn in os.listdir(split):
                if fn.lower().endswith('.json'):
                    data = json.load(open(os.path.join(split, fn)))
                    for skl in data.get('skills', []):
                        cid = SKILLS.index(skl)
                        ctr[cid] += 1

    # Real total via labeled JSON
    rt_json = Counter()
    src_dir = 'data/xp_crops_labeled'
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.endswith('.json'):
                data = json.load(open(os.path.join(src_dir, fn)))
                for skl in data.get('skills', []):
                    rt_json[skl] += 1

    # Synthetic counts train/val
    skl_tr, skl_val = Counter(), Counter()
    for path, ctr in [
        ('data/yolo/synth_skill/train/labels', skl_tr),
        ('data/yolo/synth_skill/val/labels', skl_val),
    ]:
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith('.txt'):
                    for ln in open(os.path.join(path, fn)):
                        parts = ln.split()
                        if parts:
                            ctr[int(parts[0])] += 1

    # Synthetic total
    skl_tot = Counter()
    all_lbl = 'data/yolo/synth_skill/labels'
    if os.path.isdir(all_lbl):
        for fn in os.listdir(all_lbl):
            if fn.endswith('.txt'):
                for ln in open(os.path.join(all_lbl, fn)):
                    parts = ln.split()
                    if parts:
                        skl_tot[int(parts[0])] += 1

    # Build skill rows
    skill_rows = []
    for cid, name in enumerate(SKILLS):
        if cid == 0:
            continue
        skill_rows.append({
            'Skill':      name,
            'RealTrain':  rt[cid],
            'RealVal':    rv[cid],
            'RealTotal':  rt_json[name],
            'SynthTrain': skl_tr[cid],
            'SynthVal':   skl_val[cid],
            'SynthTotal': skl_tot[cid],
        })
    # Append totals row
    total_skill = {
        'Skill':      'Total',
        'RealTrain':  sum(r['RealTrain'] for r in skill_rows),
        'RealVal':    sum(r['RealVal'] for r in skill_rows),
        'RealTotal':  sum(r['RealTotal'] for r in skill_rows),
        'SynthTrain': sum(r['SynthTrain'] for r in skill_rows),
        'SynthVal':   sum(r['SynthVal'] for r in skill_rows),
        'SynthTotal': sum(r['SynthTotal'] for r in skill_rows),
    }
    skill_rows.append(total_skill)

    # Compute max vals for coloring
    data_skill = [r for r in skill_rows if r['Skill'] != 'Total']
    skill_max = {h: max(r[h] for r in data_skill) or 1 for h in skill_headers[1:]}

    # Render the skills table
    make_table(st, 'skills', skill_rows, skill_headers, skill_max)

    st.insert('end', '\n')

    # === DIGIT INSTANCES table ===
    # Real train/val digit counts
    digit_tr_real, digit_val_real = Counter(), Counter()
    for path, ctr in [
        ('data/yolo/real/train/json', digit_tr_real),
        ('data/yolo/real/val/json', digit_val_real),
    ]:
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith('.json'):
                    data = json.load(open(os.path.join(path, fn)))
                    for xp in data.get('xp_values', []):
                        for ch in str(xp):
                            if ch.isdigit():
                                ctr[int(ch)] += 1

    # Real total
    digit_total_real = Counter()
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.endswith('.json'):
                data = json.load(open(os.path.join(src_dir, fn)))
                for xp in data.get('xp_values', []):
                    for ch in str(xp):
                        if ch.isdigit():
                            digit_total_real[int(ch)] += 1

    # Synthetic via synth_map.csv
    csv_path = os.path.join('data', 'yolo', 'synth_numbers', 'synth_map.csv')
    train_dir = os.path.join('data', 'yolo', 'synth_numbers', 'train', 'images')
    val_dir   = os.path.join('data', 'yolo', 'synth_numbers', 'val', 'images')
    train_screens = set(os.listdir(train_dir)) if os.path.isdir(train_dir) else set()
    val_screens   = set(os.listdir(val_dir))   if os.path.isdir(val_dir)   else set()

    digit_tr_syn, digit_val_syn, digit_total_syn = Counter(), Counter(), Counter()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for crop, seq in rdr:
                base = crop.split('_', 1)[0] + '.png'
                for ch in seq:
                    if ch.isdigit():
                        digit_total_syn[int(ch)] += 1
                if base in train_screens:
                    for ch in seq:
                        if ch.isdigit():
                            digit_tr_syn[int(ch)] += 1
                elif base in val_screens:
                    for ch in seq:
                        if ch.isdigit():
                            digit_val_syn[int(ch)] += 1

    # Build digit rows
    digit_rows = []
    for d in range(10):
        digit_rows.append({
            'Digit':      d,
            'RealTrain':  digit_tr_real[d],
            'RealVal':    digit_val_real[d],
            'RealTotal':  digit_total_real[d],
            'SynthTrain': digit_tr_syn[d],
            'SynthVal':   digit_val_syn[d],
            'SynthTotal': digit_total_syn[d],
        })

    # Append totals row
    total_digits = {
        'Digit': 'Total',
        **{h: sum(r[h] for r in digit_rows) for h in [
            'RealTrain', 'RealVal', 'RealTotal',
            'SynthTrain', 'SynthVal', 'SynthTotal'
        ]}
    }
    digit_rows.append(total_digits)

    # Compute max vals for coloring
    data_digits = digit_rows[:-1]
    digit_max = {h: max(r[h] for r in data_digits) or 1 for h in [
        'RealTrain', 'RealVal', 'RealTotal',
        'SynthTrain', 'SynthVal', 'SynthTotal'
    ]}

    # Render the digit table
    st.insert('end', 'Digit Instances\n', 'hdr')
    make_table(
        st, 'digits', digit_rows,
        ['Digit', 'RealTrain', 'RealVal', 'RealTotal', 'SynthTrain', 'SynthVal', 'SynthTotal'],
        digit_max
    )

    st.config(state='disabled')

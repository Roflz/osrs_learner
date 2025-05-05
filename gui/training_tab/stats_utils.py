import os, json, csv
from collections import Counter

from config import SKILLS


def color_for_ratio(r):
    if r < 0.5:
        r_, g = 200, int((r/0.5)*200)
    else:
        r_, g = int((1-(r-0.5)/0.5)*200), 200
    return f'#{r_:02x}{g:02x}00'


def make_table(st, table_id, rows, headers, max_vals):
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))
    st.tag_configure('hdr', font=('Courier',10,'bold'), foreground='#ffffff')
    # header row
    for h in headers:
        st.insert('end', h.ljust(widths[h]) + '  ', 'hdr')
    st.insert('end', '\n', 'hdr')
    # separator row
    for h in headers:
        st.insert('end', '-'*widths[h] + '  ', 'hdr')
    st.insert('end', '\n', 'hdr')
    # data rows
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


def update_stats(self):
    st = self.stats_text
    st.config(state='normal', font=('Courier',10), bg='#1e1e1e', fg='#e0e0e0')
    st.delete('1.0', 'end')

    from collections import Counter

    # 1) SKILLS table
    rt, rv = Counter(), Counter()
    for split, ctr in [
        ('data/yolo/real/train/json', rt),
        ('data/yolo/real/val/json',   rv),
    ]:
        if os.path.isdir(split):
            for fn in os.listdir(split):
                if not fn.lower().endswith('.json'):
                    continue
                data = json.load(open(os.path.join(split, fn)))
                for skill_name in data.get('skills', []):
                    cid = SKILLS.index(skill_name)
                    ctr[cid] += 1

    # RealTotal via labeled JSON
    rt_json = Counter()
    src_dir = 'data/xp_crops_labeled'
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.endswith('.json'):
                data = json.load(open(os.path.join(src_dir, fn)))
                for skill in data.get('skills', []):
                    rt_json[skill] += 1

    # Synthetic counts
    skl_tr, skl_val = Counter(), Counter()
    for path, ctr in [
        ('data/yolo/synth_skill/train/labels', skl_tr),
        ('data/yolo/synth_skill/val/labels',   skl_val),
    ]:
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith('.txt'):
                    for ln in open(os.path.join(path, fn)):
                        parts = ln.split()
                        if parts:
                            ctr[int(parts[0])] += 1

    skl_tot = Counter()
    all_lbl = 'data/yolo/synth_skill/labels'
    if os.path.isdir(all_lbl):
        for fn in os.listdir(all_lbl):
            if fn.endswith('.txt'):
                for ln in open(os.path.join(all_lbl, fn)):
                    parts = ln.split()
                    if parts:
                        skl_tot[int(parts[0])] += 1

    # Build rows
    skill_rows = []
    for cid, name in enumerate(SKILLS):
        if cid == 0:
            continue
        rt_val = rt[cid]
        rv_val = rv[cid]
        st_val = skl_tr[cid]
        sv_val = skl_val[cid]
        skill_rows.append({
            'Skill': name,
            'RealTrain': rt_val,
            'RealVal': rv_val,
            'RealTotal': rt_json[name],
            'SynthTrain': st_val,
            'SynthVal': sv_val,
            'SynthTotal': skl_tot[cid],
        })

    total_skill = {
        'Skill': 'Total',
        'RealTrain': sum(r['RealTrain'] for r in skill_rows),
        'RealVal': sum(r['RealVal'] for r in skill_rows),
        'RealTotal': sum(r['RealTotal'] for r in skill_rows),
        'SynthTrain': sum(r['SynthTrain'] for r in skill_rows),
        'SynthVal': sum(r['SynthVal'] for r in skill_rows),
        'SynthTotal': sum(r['SynthTotal'] for r in skill_rows),
    }
    skill_rows.append(total_skill)

    data_skill = [r for r in skill_rows if r['Skill'] != 'Total']
    skill_max = {h: max(r[h] for r in data_skill) or 1 for h in ['RealTrain','RealVal','RealTotal','SynthTrain','SynthVal','SynthTotal']}

    st.insert('end', 'Skills\n', 'hdr')
    make_table(st, 'skills', skill_rows,
               ['Skill','RealTrain','RealVal','RealTotal','SynthTrain','SynthVal','SynthTotal'],
               skill_max)

    st.insert('end', '\n')

    # 2) DIGIT INSTANCES table
    digit_tr_real, digit_val_real = Counter(), Counter()
    for path, ctr in [('data/yolo/real/train/json', digit_tr_real),('data/yolo/real/val/json', digit_val_real)]:
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith('.json'):
                    data = json.load(open(os.path.join(path, fn)))
                    for xp in data.get('xp_values', []):
                        for ch in str(xp):
                            if ch.isdigit():
                                ctr[int(ch)] += 1

    digit_total_real = Counter()
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.endswith('.json'):
                data = json.load(open(os.path.join(src_dir, fn)))
                for xp in data.get('xp_values', []):
                    for ch in str(xp):
                        if ch.isdigit():
                            digit_total_real[int(ch)] += 1

    csv_path = os.path.join('data','yolo','synth_numbers','synth_map.csv')
    train_dir = os.path.join('data', 'yolo', 'synth_numbers', 'train', 'images')
    train_screens = set(os.listdir(train_dir)) if os.path.isdir(train_dir) else set()

    val_dir = os.path.join('data', 'yolo', 'synth_numbers', 'val', 'images')
    val_screens = set(os.listdir(val_dir)) if os.path.isdir(val_dir) else set()

    digit_tr_syn, digit_val_syn, digit_total_syn = Counter(), Counter(), Counter()
    if os.path.isfile(csv_path):
        with open(csv_path,newline='') as f:
            rdr = csv.reader(f)
            next(rdr,None)
            for crop, seq in rdr:
                base = crop.split('_',1)[0] + '.png'
                for ch in seq:
                    if ch.isdigit():
                        digit_total_syn[int(ch)] +=1
                if base in train_screens:
                    for ch in seq:
                        if ch.isdigit(): digit_tr_syn[int(ch)] +=1
                elif base in val_screens:
                    for ch in seq:
                        if ch.isdigit(): digit_val_syn[int(ch)] +=1

    digit_rows = [{'Digit':d,'RealTrain':digit_tr_real[d],
                   'RealVal':digit_val_real[d],
                   'RealTotal':digit_total_real[d],
                   'SynthTrain':digit_tr_syn[d],
                   'SynthVal':digit_val_syn[d],
                   'SynthTotal':digit_total_syn[d]} for d in range(10)]
    total_digits = {'Digit':'Total',**{h:sum(r[h] for r in digit_rows) for h in ['RealTrain','RealVal','RealTotal','SynthTrain','SynthVal','SynthTotal']}}
    digit_rows.append(total_digits)
    data_digits = digit_rows[:-1]
    digit_max = {h:max(r[h] for r in data_digits) or 1 for h in ['RealTrain','RealVal','RealTotal','SynthTrain','SynthVal','SynthTotal']}

    st.insert('end','Digit Instances\n','hdr')
    make_table(st,'digits',digit_rows,['Digit','RealTrain','RealVal','RealTotal','SynthTrain','SynthVal','SynthTotal'],digit_max)

    st.config(state='disabled')

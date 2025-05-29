# === NEW training_utils.py ===
import os
import sys
import shutil

def build_split_real_cmd(src, pct):
    return [sys.executable, 'training/split_real.py', '--src', src, '--pct', str(pct)]

def build_synth_number_cmd(cfg):
    cmd = [sys.executable, 'training/make_synth_numbers.py',
           '--num_images', str(cfg['num_images']),
           '--neg_ratio', str(cfg['neg_ratio']),
           '--min_seq', str(cfg['min_seq']),
           '--max_seq', str(cfg['max_seq']),
           '--min_digits', str(cfg['min_digits']),
           '--max_digits', str(cfg['max_digits'])]
    if cfg.get('max_weight'):
        cmd += ['--max_weight', str(cfg['max_weight'])]
    if not cfg.get('use_weights', True):
        cmd.append('--no_weights')
    return cmd

def build_synth_skill_cmd(cfg):
    cmd = [sys.executable, 'training/make_synth_skill.py',
           '--num_images', str(cfg['num_images']),
           '--max_icons', str(cfg['max_icons'])]
    if cfg.get('max_weight'):
        cmd += ['--max_weight', str(cfg['max_weight'])]
    if not cfg.get('use_weights', True):
        cmd.append('--no_weights')
    return cmd

def build_split_synth_cmd(pct, mode):
    script = 'training/split_synth.py' if mode == 'numbers' else 'training/split_synth_skill.py'
    return [sys.executable, script, '--pct', str(pct)]

def build_yolo_cmd(cfg):
    """
    Returns a cmd list that runs exactly:
      cmd.exe /c yolo detect train model=… data=… epochs=… imgsz=… batch=… rect=True verbose=False
    """
    cmd = [
        "cmd.exe", "/c",
        "yolo", "detect", "train",
        f"model={cfg['model_path']}",
        f"data={cfg['data_yaml']}",
        f"epochs={cfg['epochs']}",
        f"imgsz={cfg['imgsz']}",
        f"batch={cfg['batch']}"
    ]
    if cfg.get("rect"):
        cmd.append("rect=True")
    # replace --quiet with the proper verbose=False flag
    cmd.append("verbose=False")
    return cmd



def build_crnn_cmd(cfg):
    return [sys.executable, 'training/train_crnn.py',
            '--epochs', str(cfg['epochs']),
            '--batch', str(cfg['batch']),
            '--lr', str(cfg['lr']),
            '--sched', cfg['sched']]

def clear_dirs(paths, recreate=False):
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
    if recreate:
        for p in paths:
            os.makedirs(p, exist_ok=True)

def clear_real_data():
    paths = [
        'data/yolo/real/train/images',
        'data/yolo/real/train/json',
        'data/yolo/real/train/labels',
        'data/yolo/real/val/images',
        'data/yolo/real/val/json',
        'data/yolo/real/val/labels'
    ]
    clear_dirs(paths, recreate=True)

import os
import shutil

def clear_synth_numbers():
    # all the dirs you already clear
    paths = [
        'data/yolo/synth_numbers/images',
        'data/yolo/synth_numbers/train/images',
        'data/yolo/synth_numbers/train/labels',
        'data/yolo/synth_numbers/labels',
        'data/yolo/synth_numbers/val/images',
        'data/yolo/synth_numbers/val/labels',
    ]
    # remove & recreate each
    clear_dirs(paths, recreate=True)

    # now also delete the mapping CSV if it exists
    csv_path = os.path.join('data', 'yolo', 'synth_numbers', 'synth_map.csv')
    if os.path.isfile(csv_path):
        os.remove(csv_path)


def clear_synth_skills():
    paths = [
        'data/yolo/synth_skill/images',
        'data/yolo/synth_skill/train/images',
        'data/yolo/synth_skill/train/labels',
        'data/yolo/synth_skill/labels',
        'data/yolo/synth_skill/val/images',
        'data/yolo/synth_skill/val/labels'
    ]
    clear_dirs(paths, recreate=True)

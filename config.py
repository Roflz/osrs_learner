import os

import yaml

# Base data directories
CROP_DIR = os.path.join('data', 'xp_crops')
LABELED_DIR = os.path.join('data', 'xp_crops_labeled')
SKIP_DIR = os.path.join('data', 'xp_crops_skipped')

# Model paths
YOLO_MODEL_PATH = os.path.join('models', 'best.pt')
CRNN_MODEL_PATH = os.path.join('models', 'crnn_best.pt')

# Threshold defaults
DROP_THRESHOLD = 80.0
XP_THRESHOLD = 80.0
SKILL_THRESHOLD = 80.0

# Skills classes: load dynamically from data.yaml, including class 0
DATA_YAML = 'data.yaml'
with open(DATA_YAML, 'r') as yf:
    cfg = yaml.safe_load(yf)
SKILLS = [cfg['names'][i] for i in sorted(cfg['names'].keys())]

# UI Theme colors
BG_COLOR = '#1e1e1e'
FG_COLOR = '#ffffff'
BUTTON_BG = '#333333'
BUTTON_FG = '#ffffff'
ENTRY_BG = '#2b2b2b'
ENTRY_FG = '#ffffff'
HIGHLIGHT_COLOR = '#5e9cff'
SHORTCUT_COLOR = '#CCCC66'
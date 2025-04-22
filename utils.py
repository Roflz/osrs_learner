import os
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

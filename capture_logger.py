import os
import csv
import time
import numpy as np
import cv2
from pynput import keyboard
from mss import mss
from pynput import mouse
from PIL import Image
from utils import get_timestamp, ensure_dir

# === CONFIG ===
SAVE_DIR = "data"
IMG_DIR = os.path.join(SAVE_DIR, "screenshots")
CSV_PATH = os.path.join(SAVE_DIR, "actions.csv")
SCREEN_REGION = {"top": 100, "left": 100, "width": 800, "height": 600}
CAPTURE_INTERVAL = 0.2  # seconds

# Create folders if not exist
ensure_dir(IMG_DIR)

# Open CSV for appending
csv_file = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)
if os.stat(CSV_PATH).st_size == 0:
    csv_writer.writerow(["timestamp", "img_path", "mouse_x", "mouse_y", "button", "action_type"])

# Shared state
click_queue = []
key_queue = []

def on_click(x, y, button, pressed):
    if pressed:
        ts = get_timestamp()
        click_queue.append((ts, x, y, str(button), "click"))

def on_press(key):
    try:
        k = key.char if hasattr(key, 'char') else str(key)
    except:
        k = str(key)
    ts = get_timestamp()
    key_queue.append((ts, k, "press"))

def on_release(key):
    try:
        k = key.char if hasattr(key, 'char') else str(key)
    except:
        k = str(key)
    ts = get_timestamp()
    key_queue.append((ts, k, "release"))

listener = mouse.Listener(on_click=on_click)
listener.start()
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()

with mss() as sct:
    print("Starting capture loop... Press Ctrl+C to stop.")
    try:
        while True:
            # Capture screen
            img = np.array(sct.grab(SCREEN_REGION))
            ts = get_timestamp()
            img_path = os.path.join("screenshots", f"{ts}.png")
            full_img_path = os.path.join(IMG_DIR, f"{ts}.png")
            Image.fromarray(img).save(full_img_path)

            # Check for clicks
            while click_queue:
                click_ts, x, y, button, action_type = click_queue.pop(0)
                csv_writer.writerow([click_ts, img_path, x, y, button, action_type])

            # Check for keyboard input
            while key_queue:
                key_ts, key, action_type = key_queue.pop(0)
                csv_writer.writerow([key_ts, img_path, "-", "-", key, action_type])


            time.sleep(CAPTURE_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping...")
        csv_file.close()
        listener.stop()
        keyboard_listener.stop()


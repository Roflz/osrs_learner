import os
import csv
import time
import numpy as np
import cv2
from PIL import Image
from mss import mss
from pynput import mouse, keyboard
from ultralytics import YOLO
from utils import get_timestamp, ensure_dir

class OSRSCapture:
    def __init__(self, save_dir="data", screen_region=None, interval=0.2):
        self.save_dir = save_dir
        self.screen_region = screen_region or {"top": 100, "left": 100, "width": 800, "height": 600}
        self.interval = interval
        self.running = False
        self.click_queue = []
        self.key_queue = []
        self.action_callback = None

        self.img_dir = os.path.join(save_dir, "screenshots")
        self.csv_path = os.path.join(save_dir, "actions.csv")
        ensure_dir(self.img_dir)

        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(["timestamp", "img_path", "mouse_x", "mouse_y", "key_or_button", "action_type"])

        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.yolo_model = YOLO("yolo/osrs_custom.pt")  # path to your YOLOv8 model
        self.sct = mss()

    def on_click(self, x, y, button, pressed):
        if pressed:
            ts = get_timestamp()
            self.click_queue.append((ts, x, y, str(button)))

    def on_press(self, key):
        try:
            k = key.char if hasattr(key, 'char') else str(key)
        except:
            k = str(key)
        ts = get_timestamp()
        self.key_queue.append((ts, k, "press"))

    def on_release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else str(key)
        except:
            k = str(key)
        ts = get_timestamp()
        self.key_queue.append((ts, k, "release"))

    def classify_click_action(self, x, y, img):
        results = self.yolo_model.predict(img, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls[0])]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return f"click_{label}"
        return "click"

    def start(self):
        self.running = True
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self._capture_loop()

    def stop(self):
        self.running = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.csv_file.close()

    def _capture_loop(self):
        while self.running:
            img = np.array(self.sct.grab(self.screen_region))
            ts = get_timestamp()
            img_path = os.path.join("screenshots", f"{ts}.png")
            full_img_path = os.path.join(self.img_dir, f"{ts}.png")
            Image.fromarray(img).save(full_img_path)

            while self.click_queue:
                click_ts, x, y, button = self.click_queue.pop(0)
                action_type = self.classify_click_action(x, y, img)
                self.csv_writer.writerow([click_ts, img_path, x, y, button, action_type])
                if self.action_callback:
                    self.action_callback(action_type)

            while self.key_queue:
                key_ts, key, action_type = self.key_queue.pop(0)
                self.csv_writer.writerow([key_ts, img_path, "-", "-", key, action_type])
                if self.action_callback:
                    self.action_callback(action_type)

            time.sleep(self.interval)

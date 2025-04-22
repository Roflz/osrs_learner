import os
import csv
import time
import re
import json
import numpy as np
import cv2
from PIL import Image
from mss import mss
from pynput import mouse, keyboard
from ultralytics import YOLO
from utils import get_timestamp, ensure_dir
import pygetwindow as gw
from tkinter import messagebox
import pytesseract

SAVE_DIR = "data"
IMG_DIR = os.path.join(SAVE_DIR, "screenshots")
CSV_PATH = os.path.join(SAVE_DIR, "actions.csv")
CAPTURE_INTERVAL = 0.2

class OSRSCapture:
    def __init__(self,
                 save_dir=SAVE_DIR,
                 window_title="Old School RuneScape",
                 screen_region=None,
                 interval=CAPTURE_INTERVAL):
        wins = gw.getWindowsWithTitle(window_title)
        self.window_found = bool(wins)
        if self.window_found:
            w = wins[0]
            self.screen_region = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
        else:
            self.screen_region = screen_region or {"top":100,"left":100,"width":800,"height":600}
        self.interval = interval
        self.running = False
        self.click_queue = []
        self.key_queue = []
        self.action_callback = None

        ensure_dir(save_dir)
        ensure_dir(IMG_DIR)
        self.csv_file = open(CSV_PATH, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if os.stat(CSV_PATH).st_size == 0:
            self.csv_writer.writerow(["timestamp","img_path","mouse_x","mouse_y","button","action_type"])

        xp_path = os.path.join(save_dir, "xp_events.csv")
        ensure_dir(save_dir)
        self.xp_csv_file = open(xp_path, "a", newline="")
        self.xp_csv_writer = csv.writer(self.xp_csv_file)
        if os.stat(xp_path).st_size == 0:
            self.xp_csv_writer.writerow(["timestamp","xp_text","action_type"])

        popup_file = os.path.join(save_dir, "popup_region.json")
        if os.path.exists(popup_file):
            with open(popup_file, "r") as f:
                self.popup_region = json.load(f)
        else:
            reg = self.screen_region
            self.popup_region = {"top":reg["top"]+int(reg["height"]*0.6),
                                 "left":reg["left"]+int(reg["width"]*0.3),
                                 "width":int(reg["width"]*0.4),
                                 "height":int(reg["height"]*0.2)}

        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        try:
            self.yolo_model = YOLO("yolo/osrs_custom.pt")
            self.model_loaded = True
        except:
            self.yolo_model = None
            self.model_loaded = False

    def select_popup_region(self):
        messagebox.showinfo("Select XP Region","Click top-left corner.")
        coords = []
        def tl(x,y,btn,press):
            if press: coords.append((x,y)); return False
        mouse.Listener(on_click=tl).run()
        messagebox.showinfo("Select XP Region","Click bottom-right corner.")
        def br(x,y,btn,press):
            if press: coords.append((x,y)); return False
        mouse.Listener(on_click=br).run()
        tlx,tly = coords[0]; brx,bry = coords[1]
        self.popup_region = {"top":tly,"left":tlx,
                             "width":brx-tlx,"height":bry-tly}
        with open(popup_file, "w") as f:
            json.dump(self.popup_region, f)

    def on_click(self, x, y, button, pressed):
        if pressed:
            ts = get_timestamp()
            act = getattr(self, 'last_action', f"click_{button}")
            self.last_action = act
            self.click_queue.append((ts, x, y, act))

    def on_press(self, key):
        k = getattr(key, 'char', str(key))
        ts = get_timestamp()
        act = f"key_press_{k}"
        self.last_action = act
        self.key_queue.append((ts, k, act))

    def classify_click_action(self, x, y, img):
        if not self.model_loaded:
            self.last_action = "click"
            return "click"
        res = self.yolo_model.predict(img, verbose=False)
        for r in res:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = r.names[int(box.cls[0])]
                if x1<=x<=x2 and y1<=y<=y2:
                    act = f"click_{label}"
                    self.last_action = act
                    return act
        self.last_action = "click"
        return "click"

    def start(self):
        if not self.window_found:
            messagebox.showerror("OSRS Not Found","Open OSRS before recording.")
            return
        self.running = True
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self._capture_loop()

    def stop(self):
        self.running = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.csv_file.close()
        self.xp_csv_file.close()

    def _capture_loop(self):
        from mss import mss as _mss
        with _mss() as sct:
            while self.running:
                win = gw.getActiveWindow()
                if not (win and "Old School RuneScape" in win.title):
                    time.sleep(self.interval)
                    continue
                frame = np.array(sct.grab(self.screen_region))
                ts = get_timestamp()
                fname = f"{ts}.png"
                Image.fromarray(frame).save(os.path.join(IMG_DIR, fname))
                # process clicks
                while self.click_queue:
                    cts, x, y, act = self.click_queue.pop(0)
                    self.csv_writer.writerow([cts, fname, x, y, act, act])
                    if self.action_callback: self.action_callback(act)
                # process keys
                while self.key_queue:
                    kts, k, act = self.key_queue.pop(0)
                    self.csv_writer.writerow([kts, fname, '-', '-', act, act])
                    if self.action_callback: self.action_callback(act)
                # OCR XP popup
                pr = self.popup_region
                crop = frame[pr['top']-self.screen_region['top']:pr['top']-self.screen_region['top']+pr['height'],
                             pr['left']-self.screen_region['left']:pr['left']-self.screen_region['left']+pr['width']]
                text = pytesseract.image_to_string(crop)
                clean = re.sub(r"\s+", "", text).lower()
                m = re.search(r"\+?\d+xp", clean)
                if m:
                    xp = m.group(0)
                    self.last_xp_text = xp
                    self.xp_csv_writer.writerow([ts, xp, self.last_action])
                    if self.action_callback: self.action_callback('xp_popup')
                time.sleep(self.interval)

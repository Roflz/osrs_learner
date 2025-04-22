import tkinter as tk
from tkinter import messagebox
import json
import os
import csv
import threading
import time
from datetime import datetime
from mss import mss, tools
import pygetwindow as gw
from pynput import mouse, keyboard
from PIL import Image
import numpy as np

# == CONFIG ==
DATA_DIR = "data"
REGION_FILE = os.path.join(DATA_DIR, "popup_region.json")
SCREENSHOT_DIR = os.path.join(DATA_DIR, "screenshots")
XP_CROP_DIR = os.path.join(DATA_DIR, "xp_crops")
ACTION_LOG = os.path.join(DATA_DIR, "actions.csv")

class XPRecorder:
    def __init__(self, root):
        self.root = root
        root.title("OSRS Data Recorder")
        root.geometry("360x280")

        # Interval slider
        slider_frame = tk.Frame(root)
        slider_frame.pack(pady=5)
        tk.Label(slider_frame, text="Capture interval (s):").pack(side="left")
        self.interval_var = tk.DoubleVar(value=0.6)
        self.interval_slider = tk.Scale(
            slider_frame,
            variable=self.interval_var,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient="horizontal",
            length=200,
        )
        self.interval_slider.pack(side="left", padx=5)

        # Region selector
        tk.Button(root, text="Select XP Region", command=self.select_region).pack(pady=3)
        self.show_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            root,
            text="Show XP Overlay",
            variable=self.show_var,
            command=self.toggle_overlay
        ).pack(pady=3)

        # Recording control
        self.recording = False
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=5)

        # Status counters frame
        status_frame = tk.Frame(root)
        status_frame.pack(fill="x", pady=5)
        self.click_count = 0
        self.key_count = 0
        self.shot_count = 0
        self.crop_count = 0
        self.click_label = tk.Label(status_frame, text="Clicks: 0")
        self.key_label = tk.Label(status_frame, text="Keys: 0")
        self.shot_label = tk.Label(status_frame, text="Screenshots: 0")
        self.crop_label = tk.Label(status_frame, text="XP Crops: 0")
        for lbl in (self.click_label, self.key_label, self.shot_label, self.crop_label):
            lbl.pack(side="left", padx=5)

        # Internal queues and control
        self.popup_region = None
        self.overlay = None
        self.click_queue = []
        self.key_queue = []
        self.stop_event = threading.Event()

        # Ensure storage
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(XP_CROP_DIR, exist_ok=True)
        if os.path.exists(REGION_FILE):
            with open(REGION_FILE, 'r') as f:
                self.popup_region = json.load(f)

    def select_region(self):
        wins = gw.getWindowsWithTitle("Old School RuneScape")
        if not wins:
            messagebox.showerror("Error", "OSRS window not found.")
            return
        w = wins[0]
        selector = tk.Toplevel(self.root)
        selector.overrideredirect(True)
        selector.geometry(f"{w.width}x{w.height}+{w.left}+{w.top}")
        selector.attributes("-alpha", 0.25)
        selector.attributes("-topmost", True)
        canvas = tk.Canvas(selector, cursor="cross")
        canvas.pack(fill="both", expand=True)

        coords = {}
        def on_press(event):
            coords['x1'], coords['y1'] = event.x, event.y
        def on_drag(event):
            canvas.delete('rect')
            canvas.create_rectangle(
                coords['x1'], coords['y1'], event.x, event.y,
                outline='red', width=2, tag='rect'
            )
        def on_release(event):
            coords['x2'], coords['y2'] = event.x, event.y
            selector.destroy()
            x1, y1 = coords['x1'], coords['y1']
            x2, y2 = coords['x2'], coords['y2']
            self.popup_region = {
                'left': min(x1, x2),
                'top': min(y1, y2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1)
            }
            with open(REGION_FILE, 'w') as f:
                json.dump(self.popup_region, f)
            messagebox.showinfo("Saved", f"Region: {self.popup_region}")
        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)

    def toggle_overlay(self):
        if not self.show_var.get():
            if self.overlay:
                self.overlay.destroy()
                self.overlay = None
            return
        if not self.popup_region:
            messagebox.showwarning("No region", "Select region first.")
            self.show_var.set(False)
            return
        wins = gw.getWindowsWithTitle("Old School RuneScape")
        if not wins:
            messagebox.showerror("Error", "OSRS window not found.")
            self.show_var.set(False)
            return
        w = wins[0]
        pr = self.popup_region
        abs_left = w.left + pr['left']
        abs_top  = w.top  + pr['top']
        if self.overlay:
            self.overlay.destroy()
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.geometry(f"{pr['width']}x{pr['height']}+{abs_left}+{abs_top}")
        self.overlay.attributes('-alpha',0.3)
        self.overlay.attributes('-topmost',True)
        tk.Frame(self.overlay,bg='red').pack(fill='both',expand=True)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.interval = self.interval_var.get()
        newf = not os.path.exists(ACTION_LOG)
        self.csvf = open(ACTION_LOG, 'a', newline='')
        self.writer = csv.writer(self.csvf)
        if newf:
            self.writer.writerow(['timestamp','img','mx','my','btn','key','xp_crop'])
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.record_loop, daemon=True)
        self.thread.start()
        self.recording = True
        self.record_button.config(text='Stop Recording')

    def stop_recording(self):
        self.stop_event.set()
        self.thread.join()
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.csvf.close()
        self.recording = False
        self.record_button.config(text='Start Recording')

    def on_click(self, x, y, button, pressed):
        wins = gw.getWindowsWithTitle('Old School RuneScape')
        if not pressed or not wins:
            return
        w = wins[0]
        if w.left <= x <= w.left + w.width and w.top <= y <= w.top + w.height:
            ts = datetime.now().isoformat()
            self.click_queue.append((ts, x, y, str(button)))
            self.click_count += 1
            self.root.after(0, lambda: self.click_label.config(text=f"Clicks: {self.click_count}"))

    def on_press(self, key):
        wins = gw.getWindowsWithTitle('Old School RuneScape')
        active = gw.getActiveWindow()
        if wins and active and 'Old School RuneScape' in active.title:
            ts = datetime.now().isoformat()
            self.key_queue.append((ts, str(key)))
            self.key_count += 1
            self.root.after(0, lambda: self.key_label.config(text=f"Keys: {self.key_count}"))

    def record_loop(self):
        with mss() as sct:
            while not self.stop_event.is_set():
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                wins = gw.getWindowsWithTitle('Old School RuneScape')
                imgpath = ''
                croppath = ''
                if wins:
                    w = wins[0]
                    region = {'top': w.top, 'left': w.left, 'width': w.width, 'height': w.height}
                    frame = sct.grab(region)
                    imgpath = os.path.join(SCREENSHOT_DIR, f"{ts}.png")
                    tools.to_png(frame.rgb, frame.size, output=imgpath)
                    self.shot_count += 1
                    self.root.after(0, lambda: self.shot_label.config(text=f"Screenshots: {self.shot_count}"))
                    if self.popup_region:
                        # Convert to PIL and crop
                        pil_full = Image.frombytes('RGB', frame.size, frame.rgb)
                        pr = self.popup_region
                        crop = pil_full.crop((pr['left'], pr['top'], pr['left']+pr['width'], pr['top']+pr['height']))
                        croppath = os.path.join(XP_CROP_DIR, f"{ts}.png")
                        crop.save(croppath)
                        self.crop_count += 1
                        self.root.after(0, lambda: self.crop_label.config(text=f"XP Crops: {self.crop_count}"))
                while self.click_queue:
                    c_ts, mx, my, btn = self.click_queue.pop(0)
                    self.writer.writerow([c_ts, imgpath, mx, my, btn, '', croppath])
                while self.key_queue:
                    k_ts, k = self.key_queue.pop(0)
                    self.writer.writerow([k_ts, imgpath, '', '', '', k, croppath])
                self.csvf.flush()
                time.sleep(self.interval)

if __name__ == '__main__':
    root = tk.Tk()
    app = XPRecorder(root)
    root.mainloop()

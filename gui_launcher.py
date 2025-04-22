import threading
import time
import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from PIL import Image, ImageTk
import cv2
import numpy as np
from pynput import mouse, keyboard

from capture_core import OSRSCapture
import os
import pyautogui
from mss import mss as mss_module
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import pytesseract
import re

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OSRS Learner Recorder")
        self.root.geometry("800x600")

        # Root layout: Graph expands, controls stay minimal
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        # Initialize capture core
        self.capture = OSRSCapture()
        self.capture.action_callback = self.track_action
        self.capture_thread = None

        # Input counters
        self.click_count = 0
        self.key_count = 0

        # --- Graph Area ---
        graph_frame = tk.Frame(root)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.tooltip = tk.StringVar()
        tk.Label(graph_frame, textvariable=self.tooltip, fg="gray").grid(row=1, column=0, sticky="w", pady=(4,0))

        # --- Bottom Panel ---
        bottom = tk.Frame(root)
        bottom.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        ctrl = tk.Frame(bottom)
        ctrl.pack(fill="x", pady=2)
        self.toggle_button = tk.Button(ctrl, text="Start Recording", command=self.toggle_recording)
        self.toggle_button.pack(side="left", padx=3)
        tk.Button(ctrl, text="Select XP Popup Region", command=self.capture.select_popup_region).pack(side="left", padx=3)
        self.show_popup_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Show XP Region", variable=self.show_popup_var).pack(side="left", padx=3)
        self.status_label = tk.Label(ctrl, text="Status: Idle")
        self.status_label.pack(side="left", padx=3)

        self.stats_label = tk.Label(bottom, text="Actions: 0 | XP: 0 | Clicks: 0 | Keys: 0")
        self.stats_label.pack(fill="x", pady=2)

        logsum = tk.Frame(bottom)
        logsum.pack(fill="x", pady=2)

        left = tk.Frame(logsum)
        left.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(left, text="Recent XP Events:").pack(anchor="w")
        self.xp_log_box = tk.Listbox(left, height=4)
        self.xp_log_box.pack(fill="both", expand=True)

        right = tk.Frame(logsum)
        right.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(right, text="XP Summary:").pack(anchor="w")
        self.xp_summary_box = tk.Listbox(right, height=4)
        self.xp_summary_box.pack(fill="both", expand=True)
        tk.Button(right, text="Reset Summary", command=self.reset_summary).pack(pady=2)

        goal_frame = tk.Frame(bottom)
        goal_frame.pack(fill="x", pady=5)
        self.goal_ui = GoalSettingUI(goal_frame, recommender=None, autoplayer=None)

        # Internal state
        self.total_actions = 0
        self.xp_popup_count = 0
        self.xp_history = []

        os.makedirs("data", exist_ok=True)
        self.xp_log_path = os.path.join("data", "xp_events.csv")
        if not os.path.exists(self.xp_log_path):
            with open(self.xp_log_path, "w") as f:
                f.write("timestamp,xp_text,action_type\n")

    def toggle_recording(self):
        if not self.capture.window_found:
            messagebox.showwarning("Window Not Found", "Please open OSRS before recording.")
            return
        if self.capture.running:
            self.capture.stop()
            self.mouse_listener.stop()
            self.keyboard_listener.stop()
            self.toggle_button.config(text="Start Recording")
            self.status_label.config(text="Status: Idle")
        else:
            self.mouse_listener = mouse.Listener(on_click=self.on_click)
            self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
            self.mouse_listener.start()
            self.keyboard_listener.start()
            self.capture_thread = threading.Thread(target=self.capture.start, daemon=True)
            self.capture_thread.start()
            self.toggle_button.config(text="Stop Recording")
            self.status_label.config(text="Status: Recording...")

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.click_count += 1
            self.capture.last_action = f"click_{button}"
            self.total_actions += 1
            self.update_stats_label()

    def on_key_press(self, key):
        k = getattr(key, 'char', str(key))
        self.key_count += 1
        self.capture.last_action = f"key_press_{k}"
        self.total_actions += 1
        self.update_stats_label()

    def track_action(self, action_type):
        if action_type == 'xp_popup':
            self.xp_popup_count += 1
            xp_text = self.capture.last_xp_text
            ts = datetime.now().strftime('%H:%M:%S')
            last_act = getattr(self.capture, 'last_action', 'none')
            self.xp_log_box.insert(0, f"{ts} {xp_text}  [last: {last_act}]")
            if self.xp_log_box.size() > 20:
                self.xp_log_box.delete(tk.END)
            try:
                val = int(re.sub(r"\D+", "", xp_text))
            except:
                val = 0
            self.xp_history.append(val)
            self.ax.clear()
            self.ax.plot(self.xp_history, marker='o')
            self.ax.set_title('XP Over Time')
            self.canvas.draw()
            self.click_count = 0
            self.key_count = 0
        self.update_stats_label()

    def update_stats_label(self):
        self.stats_label.config(
            text=f"Actions: {self.total_actions} | XP: {self.xp_popup_count} | Clicks: {self.click_count} | Keys: {self.key_count}"
        )

    def reset_summary(self):
        self.xp_history.clear()
        self.ax.clear()
        self.canvas.draw()
        self.xp_log_box.delete(0, tk.END)

class ActionRecommender:
    def __init__(self, xp_data_path="data/clean_xp_data.csv"):
        try:
            self.model = joblib.load("models/xp_predictor.pkl")
            self.model_loaded = True
        except FileNotFoundError:
            messagebox.showerror("Model Not Found", "XP predictor model not found.\nPlease train the model first.")
            self.model_loaded = False
            self.action_map, self.inverse_xp_map = {}, {}
            return
        df = pd.read_csv(xp_data_path)
        cats = df["action_type"].astype("category")
        self.action_map = dict(enumerate(cats.cat.categories))
        cats2 = df["xp_type"].astype("category")
        self.inverse_xp_map = {v: k for k, v in enumerate(cats2.cat.categories)}

    def recommend(self, xp_type):
        if not getattr(self, 'model_loaded', False):
            return 'click', 0.0
        xp_id = self.inverse_xp_map.get(xp_type, 0)
        preds = [(aid, self.model.predict([[aid, xp_id]])[0]) for aid in self.action_map]
        best = max(preds, key=lambda x: x[1])
        return self.action_map[best[0]], best[1]

class AutoPlayer:
    class MultiStepGoal:
        def __init__(self, steps):
            self.steps = steps
            self.current_index = 0
            self.current_count = 0
        def get_current_goal(self):
            return self.steps[self.current_index][0]
        def advance(self):
            self.current_count += 1
            if self.current_count >= self.steps[self.current_index][1]:
                self.current_index += 1
                self.current_count = 0
                return self.current_index < len(self.steps)
            return True
        def is_done(self):
            return self.current_index >= len(self.steps)
    def __init__(self, recommender):
        self.recommender = recommender
        self.goal = None
        self.multi_goal = None
    def set_goal(self, skill):
        self.goal = skill
    def set_multi_goal(self, steps):
        self.multi_goal = self.MultiStepGoal(steps)
    def perform_best_action(self):
        import time
        if self.multi_goal:
            self.goal = self.multi_goal.get_current_goal()
        if not self.goal:
            print("No skill goal set.")
            return
        for _ in range(5):
            action, xp = self.recommender.recommend(self.goal)
            print(f"Performing {action} (est. {xp:.1f} XP)")
            if action.startswith("click_"):
                self.fake_click(action)
            time.sleep(5)
            if self.multi_goal and not self.multi_goal.advance():
                print("Multi-step goal complete!")
                break
    def fake_click(self, label):
        from ultralytics import YOLO
        import mss
        yolo = YOLO("yolo/osrs_custom.pt")
        sct = mss.mss()
        region = {"top":100,"left":100,"width":800,"height":600}
        img = np.array(sct.grab(region))
        res = yolo.predict(img, verbose=False)
        for r in res:
            for b in r.boxes:
                name = r.names[int(b.cls[0])]
                if label == f"click_{name}":
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cx = (x1+x2)//2 + region["left"]
                    cy = (y1+y2)//2 + region["top"]
                    pyautogui.moveTo(cx, cy, duration=0.2)
                    pyautogui.click()
                    return

class GoalSettingUI:
    def __init__(self, root, recommender, autoplayer):
        self.recommender = recommender
        self.autoplayer = autoplayer
        frame = tk.Frame(root)
        frame.pack(fill="x", pady=5)
        tk.Label(frame, text="Skill Goal:").pack(anchor="w")
        self.skill_entry = tk.Entry(frame)
        self.skill_entry.pack(fill="x")
        tk.Button(frame, text="Recommend", command=self.recommend_action).pack(pady=2)
        tk.Button(frame, text="Auto-Play", command=self.auto_play_action).pack(pady=2)
        tk.Label(frame, text="Multi-Step Goal (Skill:Count,...):").pack(anchor="w")
        self.multi_entry = tk.Entry(frame)
        self.multi_entry.pack(fill="x")
        tk.Button(frame, text="Start Multi-Step Goal", command=self.start_multi_step_goal).pack(pady=2)
        self.step_tracker_label = tk.Label(frame, text="")
        self.step_tracker_label.pack(pady=2)
        self.recommendation = tk.StringVar()
        tk.Label(frame, textvariable=self.recommendation, fg="blue").pack()

    def recommend_action(self):
        skill = self.skill_entry.get().strip()
        if not skill:
            self.recommendation.set("Enter a skill.")
            return
        self.autoplayer.set_goal(skill)
        action, xp = self.recommender.recommend(skill)
        self.recommendation.set(f"Try: {action} (est. {xp:.1f} XP)")

    def auto_play_action(self):
        self.autoplayer.perform_best_action()
        self.update_step_tracker()

    def start_multi_step_goal(self):
        s = self.multi_entry.get().strip()
        try:
            steps = [(ss.split(":")[0].strip(), int(ss.split(":")[1])) for ss in s.split(",")]
            self.autoplayer.set_multi_goal(steps)
            self.recommendation.set(f"Started multi-step: {steps}")
            self.update_step_tracker()
        except Exception as e:
            self.recommendation.set(f"Invalid format: {e}")

    def update_step_tracker(self):
        mg = self.autoplayer.multi_goal
        if mg:
            idx = mg.current_index + 1
            total = len(mg.steps)
            skill, target = mg.steps[mg.current_index]
            cur = mg.current_count
            self.step_tracker_label.config(text=f"Step {idx}/{total}: {skill} [{cur}/{target}]")
        else:
            self.step_tracker_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    recommender = ActionRecommender()
    autoplayer = AutoPlayer(recommender)
    app.goal_ui.recommender = recommender
    app.goal_ui.autoplayer = autoplayer
    root.mainloop()

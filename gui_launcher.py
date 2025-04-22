import threading
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from capture_core import OSRSCapture
import os
import pygetwindow as gw  # new import for window detection
from datetime import datetime
import json
from tkinter import messagebox  # for warning dialogs
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OSRS Learner Recorder")
        self.root.geometry("600x600")
        self.capture = OSRSCapture()
        self.capture.action_callback = self.track_action
        self.capture_thread = None

        # Recording controls
        self.status_var = tk.StringVar(value="Status: Idle")
        self.toggle_button = tk.Button(root, text="Start Recording", command=self.toggle_recording, height=2, width=20)
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.toggle_button.pack(pady=5)
        self.status_label.pack()

        # Preview and stats
        self.preview_canvas = tk.Label(root)
        self.preview_canvas.pack(pady=5)
        self.stats_label = tk.Label(root, text="Frames: 0 | Actions: 0 | XP popups: 0")
        self.stats_label.pack(pady=5)

        # Recent XP log
        self.xp_log_label = tk.Label(root, text="Recent XP Events:")
        self.xp_log_label.pack()
        self.xp_log_box = tk.Listbox(root, height=5, width=50)
        self.xp_log_box.pack(pady=5)

        # XP graph and tooltip
        self.fig, self.ax = plt.subplots(figsize=(5, 2.5))
        self.tooltip = tk.StringVar(value="")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=5)
        self.tooltip_label = tk.Label(root, textvariable=self.tooltip, fg="gray")
        self.tooltip_label.pack()

        # XP summary panel
        self.xp_summary_label = tk.Label(root, text="Total XP by Skill:", font=("Arial", 10, "bold"))
        self.xp_summary_label.pack()
        self.xp_summary_box = tk.Listbox(root, height=5, width=40)
        self.xp_summary_box.pack(pady=5)
        self.reset_summary_button = tk.Button(root, text="Reset Summary", command=self.reset_summary)
        self.reset_summary_button.pack(pady=5)

        # Internal state
        self.preview_running = False
        self.total_frames = 0
        self.total_actions = 0
        self.xp_popup_count = 0
        self.action_counts = {}
        self.xp_history = []  # (skill, xp_value, timestamp)

        # Data paths
        self.goal_path = os.path.join("data", "goal.json")
        self.xp_log_path = os.path.join("data", "xp_events.csv")
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.xp_log_path):
            with open(self.xp_log_path, "w") as f:
                f.write("timestamp,xp_text,action_type,xp_type\n")

        self.last_action = None
        self.current_goal = self.load_goal()

    def toggle_recording(self):
        # Ensure the OSRS window is open
        if not getattr(self.capture, 'window_found', False):
            messagebox.showwarning(
                "Window Not Found",
                "Old School RuneScape window not detected. Please open the game before starting recording."
            )
            return
        if self.capture.running:
            self.capture.stop()
            self.preview_running = False
            self.status_var.set("Status: Idle")
            self.toggle_button.config(text="Start Recording")
        else:
            self.capture_thread = threading.Thread(target=self.capture.start, daemon=True)
            self.capture_thread.start()
            self.preview_running = True
            threading.Thread(target=self.update_preview_loop, daemon=True).start()
            self.status_var.set("Status: Recording...")
            self.toggle_button.config(text="Stop Recording")

    def update_preview_loop(self):
        while self.preview_running:
            frame = np.array(self.capture.sct.grab(self.capture.screen_region))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.preview_canvas.configure(image=img_tk)
            self.preview_canvas.image = img_tk
            self.total_frames += 1
            self.stats_label.config(
                text=f"Frames: {self.total_frames} | Actions: {self.total_actions} | XP popups: {self.xp_popup_count} | Most common: {self.most_common_action()}"
            )
            self.root.update_idletasks()
            self.root.after(100)

    def track_action(self, action_type):
        self.total_actions += 1
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
        self.last_action = action_type
        if action_type == "xp_popup":
            self.xp_popup_count += 1
            xp_text = self.capture.last_xp_text
            xp_type = self.classify_xp_type_from_action(self.last_action)
            self.log_xp_event(xp_text, self.last_action, xp_type)
            self.update_xp_log_display(f"{xp_text} ({xp_type})")
            self.update_xp_graph(xp_text, xp_type)

    def classify_xp_type_from_action(self, action):
        mapping = {"tree": "Woodcutting", "fishing_spot": "Fishing", "rock": "Mining", "range": "Cooking", "bank": "Banking"}
        for key, skill in mapping.items():
            if key in action:
                return skill
        return "Unknown"

    def log_xp_event(self, xp_text, action_type, xp_type):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.xp_log_path, "a") as f:
            f.write(f"{ts},{xp_text},{action_type},{xp_type}\n")

    def update_xp_log_display(self, xp_entry):
        self.xp_log_box.insert(0, xp_entry)
        if self.xp_log_box.size() > 5:
            self.xp_log_box.delete(5)

    def update_xp_graph(self, xp_text, xp_type):
        try:
            xp_value = int(''.join(filter(str.isdigit, xp_text)))
            ts = datetime.now()
            self.xp_history.append((xp_type, xp_value, ts))
            self.ax.clear()
            skills = list({t for t, _, _ in self.xp_history[-100:]})
            for skill in skills:
                pts = [(i, v) for i, (t, v, _) in enumerate(self.xp_history[-100:]) if t == skill]
                xs, ys = zip(*pts)
                dur = (self.xp_history[-1][2] - self.xp_history[-len(pts)][2]).total_seconds()
                rate = (sum(ys) / dur) * 3600 if dur > 0 else 0
                lbl = f"{skill} ({rate:.0f} XP/hr)"
                line, = self.ax.plot(xs, ys, marker='o', label=lbl)
                for x, y in pts:
                    self.ax.annotate(str(y), (x, y), fontsize=8, textcoords='offset points', xytext=(0,5), ha='center')
            self.ax.set_title("XP Over Time by Skill")
            self.ax.set_ylabel("XP")
            self.ax.set_xlabel("Event #")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.mpl_connect("motion_notify_event", self.on_hover)
            self.canvas.draw()
            self.update_xp_summary()
        except ValueError:
            pass

    def on_hover(self, event):
        if event.inaxes == self.ax:
            for line in self.ax.get_lines():
                cont, ind = line.contains(event)
                if cont:
                    i = ind['ind'][0]
                    lbl = line.get_label()
                    x, y = line.get_xdata()[i], line.get_ydata()[i]
                    self.tooltip.set(f"{lbl}: {int(y)} XP at Event {int(x)}")
                    return
        self.tooltip.set("")

    def load_goal(self):
        if os.path.exists(self.goal_path):
            with open(self.goal_path) as f:
                return json.load(f).get("skill")
        return None

    def save_goal(self, skill):
        with open(self.goal_path, 'w') as f:
            json.dump({"skill": skill}, f)

    def reset_summary(self):
        self.xp_history.clear()
        self.ax.clear()
        self.canvas.draw()
        self.update_xp_summary()

    def update_xp_summary(self):
        self.xp_summary_box.delete(0, tk.END)
        totals = {}
        for t, v, _ in self.xp_history:
            totals[t] = totals.get(t, 0) + v
        for skill, total in sorted(totals.items(), key=lambda x: -x[1]):
            elapsed = max((datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.xp_log_path))).total_seconds(), 1)
            rate = (total / elapsed) * 3600
            self.xp_summary_box.insert(tk.END, f"{skill}: {total} XP ({rate:.1f} XP/hr)")

class ActionRecommender:
    def __init__(self, xp_data_path="data/clean_xp_data.csv"):
        import pandas as pd, joblib
        self.model = joblib.load("models/xp_predictor.pkl")
        df = pd.read_csv(xp_data_path)
        cats = df["action_type"].astype("category")
        self.action_map = dict(enumerate(cats.cat.categories))
        cats2 = df["xp_type"].astype("category")
        self.xp_type_map = dict(enumerate(cats2.cat.categories))
        self.inverse_action_map = {v:k for k,v in self.action_map.items()}
        self.inverse_xp_map = {v:k for k,v in self.xp_type_map.items()}

    def recommend(self, xp_type):
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
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    cx=(x1+x2)//2+region["left"]
                    cy=(y1+y2)//2+region["top"]
                    pyautogui.moveTo(cx, cy, duration=0.2)
                    pyautogui.click()
                    return

class GoalSettingUI:
    def __init__(self, root, recommender, autoplayer):
        self.recommender = recommender
        self.autoplayer = autoplayer
        self.frame = tk.Frame(root)
        self.frame.pack(pady=10)
        tk.Label(self.frame, text="Skill Goal:").pack()
        self.skill_entry = tk.Entry(self.frame)
        self.skill_entry.pack()
        tk.Button(self.frame, text="Recommend", command=self.recommend_action).pack()
        tk.Button(self.frame, text="Auto-Play", command=self.auto_play_action).pack()
        tk.Label(self.frame, text="Multi-Step Goal (Skill:Count,...):").pack()
        self.multi_entry = tk.Entry(self.frame, width=40)
        self.multi_entry.pack()
        tk.Button(self.frame, text="Start Multi-Step Goal", command=self.start_multi_step_goal).pack()
        self.step_tracker_label = tk.Label(self.frame, text="")
        self.step_tracker_label.pack(pady=5)
        self.recommendation = tk.StringVar()
        tk.Label(self.frame, textvariable=self.recommendation, fg="blue").pack()
    def recommend_action(self):
        skill = self.skill_entry.get().strip()
        if not skill:
            self.recommendation.set("Enter a skill.")
            return
        self.autoplayer.set_goal(skill)
        action,xp = self.recommender.recommend(skill)
        self.recommendation.set(f"Try: {action} (est. {xp:.1f} XP)")
    def auto_play_action(self):
        self.autoplayer.perform_best_action()
        self.update_step_tracker()
    def start_multi_step_goal(self):
        s = self.multi_entry.get().strip()
        try:
            steps=[(ss.split(":")[0].strip(),int(ss.split(":")[1])) for ss in s.split(",")]
            self.autoplayer.set_multi_goal(steps)
            self.recommendation.set(f"Started multi-step: {steps}")
            self.update_step_tracker()
        except Exception as e:
            self.recommendation.set(f"Invalid format: {e}")
    def update_step_tracker(self):
        mg=self.autoplayer.multi_goal
        if mg:
            idx=mg.current_index+1
            total=len(mg.steps)
            skill,target=mg.steps[mg.current_index]
            cur=mg.current_count
            self.step_tracker_label.config(text=f"Step {idx}/{total}: {skill} [{cur}/{target}]")
        else:
            self.step_tracker_label.config(text="")

if __name__ == "__main__":
    root=tk.Tk()
    app=RecorderApp(root)
    recommender=ActionRecommender()
    autoplayer=AutoPlayer(recommender)
    gui=GoalSettingUI(root,recommender,autoplayer)
    root.mainloop()

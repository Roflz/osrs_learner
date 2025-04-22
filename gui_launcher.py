import threading
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from capture_core import OSRSCapture
import os
from datetime import datetime
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

        self.status_var = tk.StringVar(value="Status: Idle")
        self.toggle_button = tk.Button(root, text="Start Recording", command=self.toggle_recording, height=2, width=20)
        self.status_label = tk.Label(root, textvariable=self.status_var)

        self.preview_canvas = tk.Label(root)
        self.stats_label = tk.Label(root, text="Frames: 0 | Actions: 0 | XP popups: 0")

        self.xp_log_label = tk.Label(root, text="Recent XP Events:")
        self.xp_log_box = tk.Listbox(root, height=5, width=50)

        self.toggle_button.pack(pady=10)
        self.status_label.pack()
        self.preview_canvas.pack()
        self.stats_label.pack()
        self.xp_log_label.pack()
        self.xp_log_box.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 2.5))
        self.tooltip = tk.StringVar(value="")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.tooltip_label = tk.Label(root, textvariable=self.tooltip, fg="gray")
        self.tooltip_label.pack()
        self.xp_summary_label = tk.Label(root, text="Total XP by Skill:", font=("Arial", 10, "bold"))
        self.xp_summary_label.pack()
        self.xp_summary_box = tk.Listbox(root, height=5, width=40)
        self.xp_summary_box.pack(pady=5)

        self.preview_running = False
        self.screen_preview_thread = None
        self.total_frames = 0
        self.total_actions = 0
        self.xp_popup_count = 0
        self.action_counts = {}
        self.xp_history = []
        self.xp_labels = []

        self.xp_log_path = os.path.join("data", "xp_events.csv")
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.xp_log_path):
            with open(self.xp_log_path, "w") as f:
                f.write("timestamp,xp_text,action_type,xp_type\n")

        self.last_action = None

    def toggle_recording(self):
        if self.capture.running:
            self.capture.stop()
            self.preview_running = False
            self.status_var.set("Status: Idle")
            self.toggle_button.config(text="Start Recording")
        else:
            self.capture_thread = threading.Thread(target=self.capture.start)
            self.capture_thread.start()
            self.preview_running = True
            self.screen_preview_thread = threading.Thread(target=self.update_preview_loop)
            self.screen_preview_thread.start()
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
        if action_type not in self.action_counts:
            self.action_counts[action_type] = 0
        self.action_counts[action_type] += 1
        self.last_action = action_type

        if action_type == "xp_popup":
            self.xp_popup_count += 1
            xp_text = self.capture.last_xp_text
            xp_type = self.classify_xp_type_from_action(self.last_action)
            self.log_xp_event(xp_text, self.last_action, xp_type)
            self.update_xp_log_display(f"{xp_text} ({xp_type})")
            self.update_xp_graph(xp_text, xp_type)

    def classify_xp_type_from_action(self, action):
        if "tree" in action or "click_tree" in action:
            return "Woodcutting"
        elif "fish" in action or "click_fishing_spot" in action:
            return "Fishing"
        elif "mine" in action or "click_rock" in action:
            return "Mining"
        elif "cook" in action or "click_range" in action:
            return "Cooking"
        elif "bank" in action or "click_bank" in action:
            return "Banking"
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
        self.update_xp_summary()
        try:
            xp_value = int(''.join(filter(str.isdigit, xp_text)))
            self.xp_history.append((xp_type, xp_value))
            self.xp_labels.append(xp_type)
            self.ax.clear()
            skill_types = list(set(t for t, _ in self.xp_history[-100:]))
            for skill in skill_types:
                values = [v for t, v in self.xp_history[-100:] if t == skill]
                indices = [i for i, (t, _) in enumerate(self.xp_history[-100:]) if t == skill]
                line, = self.ax.plot(indices, values, marker='o', label=skill)
                for i, val in zip(indices, values):
                    self.ax.annotate(f"{val}", (i, val), fontsize=8, textcoords="offset points", xytext=(0,5), ha='center')
            self.ax.set_title("XP Over Time by Skill")
            self.ax.set_ylabel("XP")
            self.ax.set_xlabel("Event #")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.mpl_connect("motion_notify_event", self.on_hover)
            self.canvas.draw()
            self.update_xp_summary()
            pass

    def on_hover(self, event):
        if event.inaxes == self.ax:
            for line in self.ax.get_lines():
                cont, ind = line.contains(event)
                if cont:
                    label = line.get_label()
                    x_val = line.get_xdata()[ind['ind'][0]]
                    y_val = line.get_ydata()[ind['ind'][0]]
                    self.tooltip.set(f"{label}: {int(y_val)} XP at Event {int(x_val)}")
                    return
        self.tooltip.set("")

        def update_xp_summary(self):
        totals = {}
        for t, v in self.xp_history:
            totals[t] = totals.get(t, 0) + v
        self.xp_summary_box.delete(0, tk.END)
        for skill, total in sorted(totals.items(), key=lambda x: -x[1]):
            self.xp_summary_box.insert(tk.END, f"{skill}: {total} XP")

    def most_common_action(self):
        if not self.action_counts:
            return "None"
        return max(self.action_counts, key=self.action_counts.get)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()

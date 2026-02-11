import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import vlc

# App Setup
app = tk.Tk()
app.geometry("600x600")
app.title("Drowsy Boi 4.0")
ctk.set_appearance_mode("dark")

# Frame for Video
vidFrame = tk.Frame(app, height=400, width=600)
vidFrame.pack()

vid = ctk.CTkLabel(vidFrame)
vid.pack()

# Global Counter
counter = 0
def restart_counter():
    global counter
    counter = 0

# Reset Button
resetButton = ctk.CTkButton(
    app,
    height=40,
    width=120,
    font=("Arial", 20),
    text="Reset",
    text_color="white",
    fg_color="#333333",
    command=restart_counter
)
resetButton.pack(pady=20)

# Load Model
model = torch.hub.load("ultralytics/yolov5", 'custom', path="yolov5/runs/train/exp9/weights/last.pt", force_reload=True)

# Video Capture
cap = cv2.VideoCapture(0)

# Video Loop
def loop():
    global counter

    ret, frame = cap.read()
    if not ret:
        app.after(10, loop)
        return

    # Process and render
    results = model(frame)
    rendered_frame = np.squeeze(results.render())

    rgb_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    vid.imgtk = imgtk
    vid.configure(image=imgtk)

    counter += 1
    app.after(10, loop)

# Start loop
loop()
app.mainloop()
cap.release()

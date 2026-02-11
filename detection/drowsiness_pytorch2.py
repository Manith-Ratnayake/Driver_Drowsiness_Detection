import torch
import cv2
import os
import uuid
import time
import numpy as np

# === CONFIG ===
CUSTOM_MODEL_PATH = 'yolov5/runs/train/exp15/weights/last.pt'  # ✅ Change to your actual path
IMAGES_PATH = os.path.join('data', 'images')
LABELS = ['awake', 'drowsy']  # 🧠 Customize your class labels
NUM_IMAGES = 5

# === Load Custom YOLOv5 Model ===
print("[INFO] Loading custom YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=CUSTOM_MODEL_PATH, force_reload=True)
print("[INFO] Model loaded successfully!")

# === Live Detection ===
cap = cv2.VideoCapture(0)
print("[INFO] Starting live detection with custom model. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to read frame.")
        break

    results = model(frame)
    results.render()  # updates results.ims
    cv2.imshow('Custom YOLOv5 Detection', np.squeeze(results.ims[0]))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# === Image Collection for Custom Classes ===
os.makedirs(IMAGES_PATH, exist_ok=True)
cap = cv2.VideoCapture(0)

for label in LABELS:
    print(f'[INFO] Collecting images for class: {label}')
    time.sleep(3)  # Let user prepare

    for img_num in range(NUM_IMAGES):
        print(f'[INFO] Capturing image {img_num+1}/{NUM_IMAGES} for {label}')
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame.")
            continue

        filename = f"{label}.{uuid.uuid1()}.jpg"
        img_path = os.path.join(IMAGES_PATH, filename)
        cv2.imwrite(img_path, frame)
        cv2.imshow('Image Collection', frame)

        time.sleep(1.5)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# === Optional: Test One Image from Collected Set ===
test_img_path = os.path.join(IMAGES_PATH, os.listdir(IMAGES_PATH)[0])
print(f"[INFO] Running inference on collected image: {test_img_path}")
result = model(test_img_path)
result.print()
result.render()
cv2.imshow("Custom Model Test Inference", np.squeeze(result.ims[0]))
cv2.waitKey(0)
cv2.destroyAllWindows()
